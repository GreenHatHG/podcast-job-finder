from __future__ import annotations

# This script runs in the dedicated FireRed environment and imports sibling modules.
# pylint: disable=import-error

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from worker_ctc import (  # type: ignore[import-not-found]
    ENCODER_FRAME_SHIFT_MS,
    FireRedCtcModel,
    load_onnx_session,
)


SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKEN_PREFIX = "<"
SPECIAL_TOKEN_SUFFIX = ">"
SENTENCEPIECE_SPACE = "▁"


@dataclass(slots=True, frozen=True)
class AlignedToken:
    token_id: int
    text: str
    start_ms: int
    end_ms: int


class FireRedOnnxAsr:  # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        model_dir: Path,
        *,
        provider: str,
        intra_op_threads: int,
    ) -> None:
        """准备语音识别所需的文件和模型。

        创建 ``FireRedOnnxAsr`` 对象时会调用这个函数。它只负责完成识别前的
        准备工作，真正把音频转换成文字的操作由 ``transcribe`` 完成。

        Args:
            model_dir: 模型文件所在的文件夹。
            provider: 指定使用哪种设备运行模型，例如 CPU。
            intra_op_threads: 模型运行时最多使用的 CPU 线程数。
        """
        self._ctc_model = FireRedCtcModel(
            model_dir,
            provider=provider,
            intra_op_threads=intra_op_threads,
        )
        self._tokens = self._ctc_model.tokens

        # 记住三个特殊标记的编号。它们分别表示“没有文字”“一句话开始”和
        # “一句话结束”，识别过程中会用这些编号判断结果是否有效或已经结束。
        self._blank_id = self._ctc_model.blank_id
        self._sos_id = self._tokens.id(SOS_TOKEN)
        self._eos_id = self._tokens.id(EOS_TOKEN)

        self._decoder = load_onnx_session(
            model_dir / "decoder.int8.onnx",
            provider=provider,
            intra_op_threads=intra_op_threads,
        )

        # 找出解码模型需要的所有临时记忆项，并按编号排列。识别较长音频时，
        # 这些临时记忆能让模型接着上一轮结果继续处理。
        self._cache_names = sorted(
            (
                input_node.name
                for input_node in self._decoder.get_inputs()
                if input_node.name.startswith("cache_")
            ),
            key=lambda name: int(name.removeprefix("cache_")),
        )

        # 从第一项临时记忆中读取它的大小。后续开始识别时，会按照这个大小
        # 创建空的临时记忆，供解码模型逐步填写。
        first_cache = next(
            node for node in self._decoder.get_inputs() if node.name == "cache_0"
        )
        self._decoder_dimension = int(first_cache.shape[-1])

    def transcribe(
        self,
        audio_path: Path,
        *,
        discard_before_ms: int = 0,
    ) -> list[AlignedToken]:
        """把一个音频文件转换成带起止时间的文字片段。

        Args:
            audio_path: 需要识别的音频文件路径。
            discard_before_ms: 丢弃在这个时间点之前已经结束的文字，单位为毫秒。
                分段音频前面带有重复声音时，可以用它避免重复返回上一段文字。

        Returns:
            按出现顺序排列的文字片段，每项都带有文字、编号和起止时间。
        """
        encoded_audio = self._ctc_model.encode(audio_path)

        # 根据编码器整理出的声音信息逐步生成文字编号。例如模型判断结果是
        # “你”“好”，这里得到的可能是它们在 tokens.txt 中对应的两个数字。
        token_ids = self._decode(
            encoded_audio.outputs,
            encoded_audio.mask.astype(np.bool_),
        )

        # 解码结果里可能包含“开始”“结束”等控制识别流程的特殊标记。
        # 时间对齐只需要真正显示给用户的文字，所以在这里去掉特殊标记。
        text_token_ids = [
            token_id
            for token_id in token_ids
            if not _is_non_text_token(self._tokens.text(token_id))
        ]

        # 整段音频没有识别出可显示的文字时，直接返回空列表。
        if not text_token_ids:
            return []

        # 把已经识别出的文字按顺序放回最合适的时间片，得到每项文字对应的
        # 开始位置和结束位置。
        spans = _force_align(
            self._ctc_model.build_logits(encoded_audio),
            text_token_ids,
            blank_id=self._blank_id,
        )

        # 把文字编号和时间片整理成调用方容易使用的结果。编码器的一个时间片
        # 代表 40 毫秒，因此将时间片编号乘以 40，换算成毫秒时间。
        aligned_tokens = [
            AlignedToken(
                token_id=token_id,
                text=_normalize_token_text(self._tokens.text(token_id)),
                start_ms=start_frame * ENCODER_FRAME_SHIFT_MS,
                end_ms=end_frame * ENCODER_FRAME_SHIFT_MS,
            )
            for token_id, (start_frame, end_frame) in zip(text_token_ids, spans)
        ]

        # 丢弃在指定时间点之前已经结束的文字。跨过这个时间点的文字会保留，
        # 避免从字词中间截断结果。
        return [token for token in aligned_tokens if token.end_ms > discard_before_ms]

    def _decode(
        self,
        encoder_outputs: np.ndarray,
        encoder_mask: np.ndarray,
    ) -> list[int]:
        sequence = np.asarray([[self._sos_id]], dtype=np.int64)
        caches = [
            np.empty((1, 0, self._decoder_dimension), dtype=np.float32)
            for _ in self._cache_names
        ]
        token_ids = []
        for _ in range(encoder_outputs.shape[1]):
            inputs: dict[str, np.ndarray] = {
                "ys": sequence,
                "encoder_outputs": encoder_outputs.astype(np.float32),
                "src_mask": encoder_mask,
            }
            inputs.update(dict(zip(self._cache_names, caches)))
            output_names = ["output"] + [
                f"new_cache_{index}" for index in range(len(caches))
            ]
            outputs = self._decoder.run(output_names, inputs)
            next_token = int(np.argmax(outputs[0], axis=-1).item())
            if next_token == self._eos_id:
                break
            token_ids.append(next_token)
            sequence = np.concatenate(
                [sequence, np.asarray([[next_token]], dtype=np.int64)],
                axis=1,
            )
            caches = outputs[1:]
        return token_ids


def _force_align(  # pylint: disable=too-many-locals
    logits: np.ndarray,
    token_ids: list[int],
    *,
    blank_id: int,
) -> list[tuple[int, int]]:
    """找出每个字或词在音频中大约出现在哪些时间片。

    语音识别完成后，我们已经知道模型听到了哪些字或词，但还不知道它们分别
    在什么时候出现。这个函数会把模型对每个短时间片的判断，与已经识别出的
    文字顺序进行比较，找出整体上最合适的一组对应关系。

    Args:
        logits: 模型对每个时间片的原始判断。每一行代表一个时间片，每一列
            代表一个可能的文字编号。
        token_ids: 已经识别出的文字编号，顺序与实际说话顺序一致。
        blank_id: “这个时间片没有出现新文字”所使用的特殊编号。

    Returns:
        每个文字对应的开始时间片和结束时间片。结束位置不包含在范围内，
        因此调用方可以直接用“结束位置减开始位置”得到持续时间。

    Raises:
        ValueError: 音频时间片太少，或者某个文字无法在音频中找到对应位置。
    """
    # 把模型的原始判断整理成可以连续相加、方便比较的分数。后面会把一条路径
    # 上所有时间片的分数加起来，分数越高，说明这条对应关系越合理。
    log_probs = _log_softmax(logits)

    # 在每两个文字之间插入 blank。例如文字编号是 [A, B]，这里会整理成
    # [blank, A, blank, B, blank]。blank 可以表示停顿、拖音，或者两个文字
    # 之间没有产生新文字的时间片。
    expanded_tokens = np.full(len(token_ids) * 2 + 1, blank_id, dtype=np.int64)
    expanded_tokens[1::2] = token_ids

    # state_count 是上面这条扩展序列共有多少个位置；frame_count 是整段音频
    # 被模型切成了多少个短时间片。
    state_count = len(expanded_tokens)
    frame_count = log_probs.shape[0]

    # 每个文字至少需要占用一个时间片。时间片数量更少时一定无法完成对应，
    # 直接报错可以让调用方看到明确原因。
    if frame_count < len(token_ids):
        raise ValueError(
            f"CTC 帧数少于待对齐 token 数：frames={frame_count} tokens={len(token_ids)}"
        )

    # scores 记录“处理到当前时间片时，到达每个位置的最佳总分”。负无穷表示
    # 这个位置目前不可能到达。第一个时间片只能落在开头的 blank 或首个文字。
    scores = np.full(state_count, -np.inf, dtype=np.float32)
    scores[0] = log_probs[0, blank_id]
    scores[1] = log_probs[0, expanded_tokens[1]]

    # backpointers 会记住每个时间片最终选择了哪种走法，方便处理完全部时间片后，
    # 从结尾倒着还原出完整对应关系。0、1、2 分别表示停在原位、前进一步、
    # 在允许时跨过中间的 blank 前进两步。
    backpointers = np.zeros((frame_count, state_count), dtype=np.int8)

    # 从第二个时间片开始，依次判断当前时间片最适合放在哪个位置。
    for frame_index in range(1, frame_count):
        # 到达一个位置有三种可能：继续停在这个位置、从前一个位置走过来，
        # 或者跨过一个 blank 从前两个位置走过来。
        candidates = np.stack(
            [
                scores,
                np.concatenate(([-np.inf], scores[:-1])),
                _build_skip_scores(scores, expanded_tokens, blank_id=blank_id),
            ]
        )

        # 为扩展序列中的每个位置选出总分最高的走法，并取出这条走法之前已经
        # 累积的分数。
        moves = np.argmax(candidates, axis=0).astype(np.int8)
        scores = candidates[moves, np.arange(state_count)]

        # 再加上模型对“当前时间片属于这个文字或 blank”的判断分数，得到处理
        # 完当前时间片后的新总分。
        scores = scores + log_probs[frame_index, expanded_tokens]

        # 保存本次选择，后面倒推路径时会用到。
        backpointers[frame_index] = moves

    # 合法路径可以结束在最后一个文字，也可以结束在它后面的 blank。比较两者
    # 的总分，选择模型认为更合适的结束位置。
    final_state = state_count - 1
    if scores[state_count - 2] > scores[final_state]:
        final_state = state_count - 2

    # states 记录每个时间片最后对应扩展序列中的哪个位置。先放入结束位置，
    # 再根据之前保存的走法，从后往前逐个还原。
    states = np.empty(frame_count, dtype=np.int32)
    states[-1] = final_state
    for frame_index in range(frame_count - 1, 0, -1):
        states[frame_index - 1] = states[frame_index] - int(
            backpointers[frame_index, states[frame_index]]
        )

    # 扩展序列中的奇数位置才是真正的文字。依次找出每个文字占用了哪些时间片，
    # 再把第一个和最后一个时间片整理成调用方需要的范围。
    spans = []
    for token_index in range(len(token_ids)):
        token_state = token_index * 2 + 1
        token_frames = np.flatnonzero(states == token_state)

        # 正常情况下，每个已识别文字都应至少对应一个时间片。找不到时说明模型
        # 给出的时间信息无法支持当前识别结果。
        if token_frames.size == 0:
            raise ValueError(f"CTC token 对齐失败：token_index={token_index}")

        # 结束位置加 1，把范围整理成左闭右开的形式，例如 (3, 6) 表示使用
        # 第 3、4、5 三个时间片。
        spans.append((int(token_frames[0]), int(token_frames[-1]) + 1))
    return spans


def _build_skip_scores(
    scores: np.ndarray,
    expanded_tokens: np.ndarray,
    *,
    blank_id: int,
) -> np.ndarray:
    skip_scores = np.full_like(scores, -np.inf)
    for state_index in range(2, len(scores)):
        token_id = expanded_tokens[state_index]
        if token_id not in (blank_id, expanded_tokens[state_index - 2]):
            skip_scores[state_index] = scores[state_index - 2]
    return skip_scores


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    maximums = np.max(logits, axis=-1, keepdims=True)
    shifted_logits = logits - maximums
    return shifted_logits - np.log(
        np.sum(np.exp(shifted_logits), axis=-1, keepdims=True)
    )


def _is_non_text_token(token: str) -> bool:
    return token.startswith(SPECIAL_TOKEN_PREFIX) and token.endswith(
        SPECIAL_TOKEN_SUFFIX
    )


def _normalize_token_text(token: str) -> str:
    return token.replace(SENTENCEPIECE_SPACE, " ")
