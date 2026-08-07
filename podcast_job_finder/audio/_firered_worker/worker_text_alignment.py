from __future__ import annotations

# This script runs in the dedicated FireRed environment and imports sibling modules.
# The alignment worker intentionally shares model-loading steps with the ASR worker.
# pylint: disable=duplicate-code,import-error

from dataclasses import dataclass
from pathlib import Path
import unicodedata

import numpy as np

from worker_asr import (  # type: ignore[import-not-found]
    BLANK_TOKEN,
    ENCODER_FRAME_SHIFT_MS,
    TokenDictionary,
    _force_align,
    _load_session,
    _log_softmax,
)
from worker_features import FeatureExtractor  # type: ignore[import-not-found]


UNKNOWN_TOKEN = "<unk>"
DIGIT_PRONUNCIATIONS = {
    "0": "零",
    "1": "一",
    "2": "二",
    "3": "三",
    "4": "四",
    "5": "五",
    "6": "六",
    "7": "七",
    "8": "八",
    "9": "九",
}


@dataclass(slots=True, frozen=True)
class TextAlignment:
    text: str
    source_start: int
    source_end: int
    start_ms: int
    end_ms: int
    confidence: float


@dataclass(slots=True, frozen=True)
class _TargetToken:
    text: str
    source_start: int
    source_end: int
    token_id: int


class FireRedTextAligner:
    """使用 FireRed 编码器和 CTC 给外部文字分配时间。"""

    def __init__(
        self,
        model_dir: Path,
        *,
        provider: str,
        intra_op_threads: int,
    ) -> None:
        self._tokens = TokenDictionary(model_dir / "tokens.txt")
        self._blank_id = self._tokens.id(BLANK_TOKEN)
        self._unknown_id = self._tokens.id(UNKNOWN_TOKEN)
        self._features = FeatureExtractor(model_dir / "cmvn.ark")
        self._encoder = _load_session(
            model_dir / "encoder.int8.onnx",
            provider=provider,
            intra_op_threads=intra_op_threads,
        )
        self._ctc = _load_session(
            model_dir / "ctc.int8.onnx",
            provider=provider,
            intra_op_threads=intra_op_threads,
        )

    def align(self, audio_path: Path, text: str) -> list[TextAlignment]:
        targets = self._tokenize(text)
        if not targets:
            raise ValueError("CTC 外部文本没有可匹配字符。")
        logits = self._infer_logits(audio_path)
        spans = _force_align(
            logits,
            [target.token_id for target in targets],
            blank_id=self._blank_id,
        )
        log_probs = _log_softmax(logits)
        return [
            _build_text_alignment(target, span, log_probs)
            for target, span in zip(targets, spans)
        ]

    def _infer_logits(self, audio_path: Path) -> np.ndarray:
        features, feature_lengths = self._features.extract(audio_path)
        encoder_outputs, encoder_lengths, _ = self._encoder.run(
            ["output", "output_lengths", "mask"],
            {"input": features, "input_lengths": feature_lengths},
        )
        logits = self._ctc.run(
            ["logits"],
            {"encoder_outputs": encoder_outputs.astype(np.float32)},
        )[0]
        return logits[0, : int(encoder_lengths[0])]

    def _tokenize(self, text: str) -> list[_TargetToken]:
        targets = []
        for index, character in enumerate(text):
            acoustic_text = _normalize_acoustic_character(character)
            if acoustic_text is None:
                continue
            targets.append(
                _TargetToken(
                    text=character,
                    source_start=index,
                    source_end=index + 1,
                    token_id=self._token_id(acoustic_text),
                )
            )
        return targets

    def _token_id(self, acoustic_text: str) -> int:
        try:
            return self._tokens.id(acoustic_text)
        except KeyError:
            return self._unknown_id


def _normalize_acoustic_character(character: str) -> str | None:
    if character in DIGIT_PRONUNCIATIONS:
        return DIGIT_PRONUNCIATIONS[character]
    category = unicodedata.category(character)
    if character.isspace() or category.startswith(("P", "S")):
        return None
    if character.isascii() and character.isalpha():
        return character.upper()
    if character.isalnum() or category.startswith(("L", "N")):
        return character
    return None


def _build_text_alignment(
    target: _TargetToken,
    span: tuple[int, int],
    log_probs: np.ndarray,
) -> TextAlignment:
    start_frame, end_frame = span
    confidence = float(
        np.exp(np.max(log_probs[start_frame:end_frame, target.token_id]))
    )
    return TextAlignment(
        text=target.text,
        source_start=target.source_start,
        source_end=target.source_end,
        start_ms=start_frame * ENCODER_FRAME_SHIFT_MS,
        end_ms=end_frame * ENCODER_FRAME_SHIFT_MS,
        confidence=confidence,
    )
