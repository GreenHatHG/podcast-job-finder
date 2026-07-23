from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from ten_vad import TenVad  # type: ignore[import-untyped]

from podcast_job_finder.audio._segment_split import (
    MIN_CUT_POSITION_RATIO,
    LongSegmentSplitConfig,
    split_long_segments,
)
from podcast_job_finder.audio.normalized_audio import NormalizedAudio
from podcast_job_finder.progress import PercentageProgressLogger


# 检测前统一使用的音频采样率。16000 表示每秒读取 16000 个声音数据点。
# TEN VAD 按这个采样率工作，输入音频需要先转换成相同格式。
VAD_SAMPLE_RATE: Final = 16_000

# 每次拿多少个声音数据点判断是否有人说话。256 个点约等于 16 毫秒。
# 数值越小反应越快，数值越大判断范围越长；TEN VAD 推荐保持 256。
VAD_FRAME_SAMPLES: Final = 256
VAD_FRAME_DURATION_MS: Final = VAD_FRAME_SAMPLES / VAD_SAMPLE_RATE * 1_000
VAD_DETECTION_OPERATION_NAME: Final = "逐帧语音检测"

# 整段音频的平均音量高于这个值时，通常包含较响的背景声。
# 此时会提高判断标准，减少把背景声当成人声的情况。
HIGH_ENERGY_THRESHOLD: Final = 10_000

# 整段音频的平均音量低于这个值时，通常说话声音比较轻。
# 此时会降低判断标准，增加识别出轻声说话的机会。
LOW_ENERGY_THRESHOLD: Final = 1_000

# 音频整体较响时，把判断标准提高到原来的多少倍。
# 1.2 表示提高 20%；数值越大，越不容易把背景声当成人声。
HIGH_ENERGY_THRESHOLD_MULTIPLIER: Final = 1.2

# 音频整体较轻时，把判断标准降低到原来的多少倍。
# 0.8 表示降低 20%；数值越小，越容易识别出轻声，也更容易带入背景声。
LOW_ENERGY_THRESHOLD_MULTIPLIER: Final = 0.8

# 强制切分时推荐保留的真实音频重叠范围；0 表示关闭重叠。
MIN_FORCED_SPLIT_OVERLAP_MS: Final = 500
MAX_FORCED_SPLIT_OVERLAP_MS: Final = 800

# 自动降低判断标准时允许达到的最低值，避免把大量背景声算作人声。
MIN_ADAPTIVE_THRESHOLD: Final = 0.2

# 自动提高判断标准时允许达到的最高值，避免漏掉大部分正常说话声。
MAX_ADAPTIVE_THRESHOLD: Final = 0.9

logger = logging.getLogger(__name__)


def _milliseconds_to_frames(duration_ms: int, frame_duration_ms: float) -> int:
    return max(1, int(np.ceil(duration_ms / frame_duration_ms)))


def _milliseconds_to_frames_allow_zero(
    duration_ms: int,
    frame_duration_ms: float,
) -> int:
    return int(np.ceil(duration_ms / frame_duration_ms))


def _validate_forced_split_overlap_frames(
    overlap_frames: int,
    *,
    max_speech_frames: int,
) -> None:
    if overlap_frames == 0:
        return
    minimum_cut_advance_frames = int(max_speech_frames * MIN_CUT_POSITION_RATIO)
    if overlap_frames >= minimum_cut_advance_frames:
        raise ValueError(
            "forced_split_overlap_ms 换算后的帧数必须小于最长片段四分之一的帧数。"
        )


@dataclass(slots=True, frozen=True)
class SpeechSegment:
    # 片段第一个采样点在规范化音频中的位置。
    start_sample: int

    # 片段结束位置；该位置对应的采样点不包含在片段内。
    end_sample: int

    @property
    def start_ms(self) -> int:
        """返回用于展示的开始毫秒；四舍五入可能产生轻微精度差异。"""
        return _samples_to_milliseconds(self.start_sample)

    @property
    def end_ms(self) -> int:
        """返回用于展示的结束毫秒；四舍五入可能产生轻微精度差异。"""
        return _samples_to_milliseconds(self.end_sample)

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def to_dict(self) -> dict[str, int]:
        return {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
        }


@dataclass(slots=True, frozen=True)
class VadConfig:
    # 判断“这段声音像不像人在说话”的严格程度，取值在 0 与 1 之间。
    # 数值越小越容易把轻声和背景声算作说话，数值越大越容易漏掉轻声，默认 0.5。
    threshold: float = 0.5

    # 每个片段期望保留的最短说话时间，单位是毫秒，1000 表示 1 秒。
    # 短于这个时间的内容会与相邻片段合并。
    # 数值越大，零碎片段越少；数值越小，短句越容易单独保留。
    min_speech_duration_ms: int = 1_000

    # 每个片段允许的最长说话时间，单位是毫秒，30000 表示 30 秒。
    # 超过后会优先在停顿处切开，找不到停顿时也会切开。
    # 数值越大，单个片段越长；数值越小，生成的片段越多。
    max_speech_duration_ms: int = 30_000

    # 片段因最长时长限制而被强制切开时，相邻片段重复保留的真实音频时长。
    # 640 毫秒对应 40 个 VAD 帧，用于给识别模型保留切点附近的完整发音。
    forced_split_overlap_ms: int = 640

    # 一段安静持续多久才算一句话结束。单位是毫秒，600 表示 0.6 秒。
    # 数值越大，会忽略普通换气；数值越小，片段会更频繁地在短暂停顿处切开。
    min_silence_duration_ms: int = 600

    def __post_init__(self) -> None:
        if not 0 < self.threshold < 1:
            raise ValueError("threshold 必须大于 0 且小于 1。")
        if self.min_speech_duration_ms <= 0:
            raise ValueError("min_speech_duration_ms 必须大于 0。")
        if self.max_speech_duration_ms < self.min_speech_duration_ms:
            raise ValueError(
                "max_speech_duration_ms 必须大于等于 min_speech_duration_ms。"
            )
        if self.forced_split_overlap_ms < 0:
            raise ValueError("forced_split_overlap_ms 必须大于等于 0。")
        if self.forced_split_overlap_ms and not (
            MIN_FORCED_SPLIT_OVERLAP_MS
            <= self.forced_split_overlap_ms
            <= MAX_FORCED_SPLIT_OVERLAP_MS
        ):
            raise ValueError(
                "forced_split_overlap_ms 必须在 500 到 800 毫秒之间，或设置为 0 关闭。"
            )
        _validate_forced_split_overlap_frames(
            _milliseconds_to_frames_allow_zero(
                self.forced_split_overlap_ms,
                VAD_FRAME_DURATION_MS,
            ),
            max_speech_frames=_milliseconds_to_frames(
                self.max_speech_duration_ms,
                VAD_FRAME_DURATION_MS,
            ),
        )
        if self.min_silence_duration_ms <= 0:
            raise ValueError("min_silence_duration_ms 必须大于 0。")


def _detect_speech_segments(
    audio: NormalizedAudio,
    *,
    config: VadConfig = VadConfig(),
) -> list[SpeechSegment]:
    """从规范化 WAV 中流式检测说话片段。"""
    # 少于一个完整判断单位的声音无法识别人声，主要处理空音频或异常短输入。
    if audio.sample_count < VAD_FRAME_SAMPLES:
        logger.debug(
            "音频不足一个 VAD 帧：sample_count=%d required_samples=%d",
            audio.sample_count,
            VAD_FRAME_SAMPLES,
        )
        return []

    # 计算每个小段代表多少毫秒，供时间配置和最终结果换算使用。
    frame_duration_ms = VAD_FRAME_DURATION_MS

    # 第一步：逐个检查约 16 毫秒的小段，记录每一小段是否有人说话。
    speech_frames = _classify_speech_frames(audio, config.threshold)

    # 第二步：从开始说话起持续往后检查，短暂停顿前后的说话内容会合并成一个片段；
    # 遇到足够长的安静后，结束并保存这个片段。
    raw_segments = _build_natural_segments(
        speech_frames,
        min_silence_frames=_milliseconds_to_frames(
            config.min_silence_duration_ms,
            frame_duration_ms,
        ),
    )
    logger.debug("自然停顿分段完成：segment_count=%d", len(raw_segments))

    # 第三步：把时间太短的内容并入相邻片段，减少零碎片段。
    merged_segments = _merge_short_segments(
        raw_segments,
        min_speech_frames=_milliseconds_to_frames(
            config.min_speech_duration_ms,
            frame_duration_ms,
        ),
    )
    logger.debug("短片段合并完成：segment_count=%d", len(merged_segments))

    # 第四步：把时间太长的内容优先从停顿处切开，控制单个片段长度。
    max_speech_frames = _milliseconds_to_frames(
        config.max_speech_duration_ms,
        frame_duration_ms,
    )
    forced_split_overlap_frames = _milliseconds_to_frames_allow_zero(
        config.forced_split_overlap_ms,
        frame_duration_ms,
    )
    split_segments = split_long_segments(
        merged_segments,
        speech_frames=speech_frames,
        audio=audio,
        config=LongSegmentSplitConfig(
            max_speech_frames=max_speech_frames,
            overlap_frames=forced_split_overlap_frames,
            frame_samples=VAD_FRAME_SAMPLES,
        ),
    )
    logger.debug("长片段切分完成：segment_count=%d", len(split_segments))

    # 第五步：把内部使用的帧位置转换成精确的采样位置。
    return _convert_to_segments(split_segments)


def _classify_speech_frames(
    audio: NormalizedAudio,
    configured_threshold: float,
) -> NDArray[np.bool_]:
    """把整段声音切成约 16 毫秒的小段，逐段判断是否有人说话。

    返回结果与这些小段一一对应：True 表示有人说话，False 表示安静或背景声。
    后续处理会根据这份结果寻找连续安静的位置，让音频尽量在自然停顿处切开。
    过长片段也会参考这份结果选择更合适的切分位置。
    """
    # 只处理长度完整的小段，结尾不足 16 毫秒的部分无法单独完成判断。
    frame_count = audio.sample_count // VAD_FRAME_SAMPLES

    # 根据整段音频的平均音量调整判断标准，兼顾轻声录音和较响的背景声。
    threshold = _adapt_threshold(
        audio,
        configured_threshold,
        sample_count=frame_count * VAD_FRAME_SAMPLES,
    )
    logger.debug(
        "开始%s：frame_count=%d frame_duration_ms=%.1f "
        "configured_threshold=%.2f effective_threshold=%.2f",
        VAD_DETECTION_OPERATION_NAME,
        frame_count,
        VAD_FRAME_DURATION_MS,
        configured_threshold,
        threshold,
    )
    vad = TenVad(VAD_FRAME_SAMPLES, threshold)
    speech_frames = np.zeros(frame_count, dtype=np.bool_)
    progress_logger = PercentageProgressLogger(
        logger=logger,
        operation_name=VAD_DETECTION_OPERATION_NAME,
        total_items=frame_count,
    )

    # 按播放顺序检查每个小段，并记下这一小段里是否有人说话。
    frames = audio.iter_samples(chunk_samples=VAD_FRAME_SAMPLES)
    for frame_index, frame in enumerate(frames):
        if frame_index == frame_count:
            break
        _, is_speech = vad.process(frame)
        speech_frames[frame_index] = is_speech == 1
        progress_logger.update(frame_index + 1)

    speech_frame_count = int(np.count_nonzero(speech_frames))
    logger.debug(
        "%s完成：speech_frames=%d total_frames=%d speech_ratio=%.1f%%",
        VAD_DETECTION_OPERATION_NAME,
        speech_frame_count,
        frame_count,
        speech_frame_count / frame_count * 100,
    )
    return speech_frames


def _adapt_threshold(
    audio: NormalizedAudio,
    threshold: float,
    *,
    sample_count: int,
) -> float:
    """根据整段音频的平均音量，微调判断“是否有人说话”的严格程度。

    较响的录音会采用更严格的标准，减少背景声干扰；较轻的录音会采用更宽松的
    标准，增加识别出轻声说话的机会；音量适中时保持用户设置的原值。
    """
    audio_energy = _calculate_mean_energy(audio, sample_count=sample_count)

    # 整体音量较响时提高判断标准，并限制最高值，避免漏掉大量正常说话声。
    if audio_energy > HIGH_ENERGY_THRESHOLD:
        return min(
            MAX_ADAPTIVE_THRESHOLD,
            max(threshold * HIGH_ENERGY_THRESHOLD_MULTIPLIER, threshold),
        )

    # 整体音量较轻时降低判断标准，并限制最低值，避免带入大量背景声。
    if audio_energy < LOW_ENERGY_THRESHOLD:
        return max(
            MIN_ADAPTIVE_THRESHOLD,
            min(threshold * LOW_ENERGY_THRESHOLD_MULTIPLIER, threshold),
        )
    return threshold


def _calculate_mean_energy(
    audio: NormalizedAudio,
    *,
    sample_count: int,
) -> float:
    absolute_sum = 0
    processed_samples = 0
    chunks = audio.iter_samples()
    for samples in chunks:
        remaining_samples = sample_count - processed_samples
        current_samples = samples[:remaining_samples]
        absolute_sum += int(
            np.abs(current_samples.astype(np.int32)).sum(dtype=np.int64)
        )
        processed_samples += int(current_samples.size)
        if processed_samples == sample_count:
            break
    return absolute_sum / processed_samples


def _build_natural_segments(
    speech_frames: NDArray[np.bool_],
    *,
    min_silence_frames: int,
) -> list[tuple[int, int]]:
    """把连续的“有人说话”判断整理成自然片段的开始和结束位置。

    遇到人声时记录片段开始位置。之后只有连续安静达到指定长度，才认为这段
    话已经结束，这样可以保留说话过程中的短暂停顿和正常换气。
    返回值中的每一项都是“开始位置、结束位置”，位置单位是约 16 毫秒的小段。
    """
    segments: list[tuple[int, int]] = []
    speech_start: int | None = None
    silence_frames = 0
    for frame_index, is_speech in enumerate(speech_frames):
        # 当前还没有开始记录片段，遇到第一小段人声时记下开始位置。
        if speech_start is None:
            if is_speech:
                speech_start = frame_index
            continue

        # 再次听到人声说明上一段安静只是停顿，从头计算连续安静的长度。
        if is_speech:
            silence_frames = 0
            continue

        # 连续安静较短时继续等待，给换气和短暂停顿留出空间。
        silence_frames += 1
        if silence_frames < min_silence_frames:
            continue

        # 连续安静达到要求，将安静开始前的内容保存为一个完整片段。
        silence_start = frame_index - silence_frames + 1
        segments.append((speech_start, silence_start))
        speech_start = None
        silence_frames = 0

    # 音频结束时仍有未保存的人声，将它作为最后一个片段保留下来。
    if speech_start is not None:
        segments.append((speech_start, len(speech_frames) - silence_frames))
    return segments


def _merge_short_segments(
    segments: list[tuple[int, int]],
    *,
    min_speech_frames: int,
) -> list[tuple[int, int]]:
    """把时间太短的片段并入相邻片段，减少零碎的小文件。

    上一个片段太短时，会把它和当前片段连在一起；当前片段太短时，会把它并入
    上一个片段。合并后的范围会包含两个片段之间的安静时间。
    """
    merged: list[tuple[int, int]] = []
    for start_frame, end_frame in segments:
        # 第一个片段暂时没有相邻片段可以选择，先保存下来。
        if not merged:
            merged.append((start_frame, end_frame))
            continue

        previous_start, previous_end = merged[-1]

        # 上一个片段太短时，将它的开始位置延续到当前片段的结束位置。
        if previous_end - previous_start < min_speech_frames:
            merged[-1] = (previous_start, end_frame)
            continue

        # 当前片段太短时，延长上一个片段，让当前内容归入其中。
        if end_frame - start_frame < min_speech_frames:
            merged[-1] = (previous_start, end_frame)
            continue

        # 两个片段都达到最短要求，保留当前片段原有的范围。
        merged.append((start_frame, end_frame))
    return merged


def _convert_to_segments(
    segments: list[tuple[int, int]],
) -> list[SpeechSegment]:
    return [
        SpeechSegment(
            start_sample=start_frame * VAD_FRAME_SAMPLES,
            end_sample=end_frame * VAD_FRAME_SAMPLES,
        )
        for start_frame, end_frame in segments
        if end_frame > start_frame
    ]


def _samples_to_milliseconds(sample_count: int) -> int:
    return round(sample_count * 1_000 / VAD_SAMPLE_RATE)
