from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Final

import numpy as np
from numpy.typing import NDArray

from podcast_job_finder.audio.segmentation._pcm import milliseconds_to_frames
from podcast_job_finder.audio.segmentation.normalized_audio import NormalizedAudio


# 少于 48 毫秒的静音通常属于 VAD 瞬时波动，不作为自然切点参与评分。
MIN_SILENCE_DURATION_MS: Final = 48

# 比较停顿与前和后各 320 毫秒音频的能量，排除被 VAD 误判成静音的低能量发音，
# 有时候 VAD 识别成静音了，但是这个时间片段可能是某个字的发音开始没多久，
# 如果从这里切割，可能模模糊糊会听出是某个字，会影响识别，所以这里需要额外的检测
SILENCE_ENERGY_CONTEXT_DURATION_MS: Final = 320

# 用来确认一段由 vad 识别出来的静音区间是否真的适合作为音频切分位置。
# 0.15 表示这段停顿的平均声音大小，最多只能是前后正常声音的 15%；
# 超过这个比例时，它更可能是轻声说话，因此不会在这里切开，避免把一句话或一个字切断。
MAX_SILENCE_TO_CONTEXT_ENERGY_RATIO: Final = 0.15


@dataclass(slots=True, frozen=True)
class SilenceInterval:
    start_frame: int
    end_frame: int

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame


@dataclass(slots=True, frozen=True)
class CutCandidate:
    cut_frame: int
    silence_duration_frames: int
    mean_energy: float


def build_silence_candidates(
    speech_frames: NDArray[np.bool_],
    audio: NormalizedAudio,
    *,
    start_frame: int,
    end_frame: int,
    frame_samples: int,
) -> list[CutCandidate]:
    """从指定语音片段中找出适合作为切点的停顿。

    函数先找出持续时间足够长的停顿，再结合停顿附近的声音逐个检查，
    最后只返回确实适合切开的停顿位置。
    """
    # 配置中的停顿长度使用毫秒表示。这里把它换算成当前音频使用的时间格数，
    # 方便与 start_frame、end_frame 直接比较。
    min_silence_frames = milliseconds_to_frames(
        MIN_SILENCE_DURATION_MS,
        sample_rate=audio.sample_rate,
        frame_samples=frame_samples,
    )

    # 检查停顿时，还要查看它前和后一小段声音。这个值表示需要向两边查看多少毫秒的音频。
    energy_context_frames = milliseconds_to_frames(
        SILENCE_ENERGY_CONTEXT_DURATION_MS,
        sample_rate=audio.sample_rate,
        frame_samples=frame_samples,
    )

    # 先根据 vad 识别出来的“有没有人在讲话”的标记，找出片段内所有静音区间。
    intervals = _find_silence_intervals(
        speech_frames,
        search_start=start_frame,
        search_end=end_frame,
        min_silence_frames=min_silence_frames,
    )

    # `candidates` 只保存通过后续声音检查的停顿。
    candidates = []

    # 逐个检查找到的停顿。某些地方虽然被标成停顿，实际仍可能包含较轻的讲话声。
    for interval in intervals:
        energy_start = max(
            start_frame,
            interval.start_frame - energy_context_frames,
        )
        energy_end = min(
            end_frame,
            interval.end_frame + energy_context_frames,
        )
        candidate = _build_silence_candidate(
            interval,
            audio=audio,
            energy_start=energy_start,
            energy_end=energy_end,
            frame_samples=frame_samples,
        )

        # 返回 None 表示这个停顿没有通过检查，不适合作为切点。
        if candidate is not None:
            candidates.append(candidate)

    return candidates


def _build_silence_candidate(
    interval: SilenceInterval,
    *,
    audio: NormalizedAudio,
    energy_start: int,
    energy_end: int,
    frame_samples: int,
) -> CutCandidate | None:
    """
    语音识别结果标出的“停顿”有时仍包含轻声讲话。这个函数会读取停顿及其
    前后的一小段声音，确认停顿处确实明显更安静。检查通过时返回一个候选切点，
    检查失败时返回 `None`，让上层逻辑忽略这个位置。
    """
    # 读取整个检查范围内每个短时间段的声音大小。数值越小，声音越安静。
    frame_energies = _read_frame_energies(
        audio,
        energy_start,
        energy_end,
        frame_samples=frame_samples,
    )

    # frame_energies 从 energy_start 开始记录，下面两个相对位置用于找到
    # 停顿在这组数据中的实际范围。
    relative_start = interval.start_frame - energy_start
    relative_end = interval.end_frame - energy_start

    # 取出停顿本身的声音数据。
    silence_energies = frame_energies[relative_start:relative_end]

    # 把停顿前后的声音放在一起，作为正常讲话音量的参考。
    context_energies = np.concatenate(
        (frame_energies[:relative_start], frame_energies[relative_end:])
    )

    # 用平均值表示整个停顿大致有多安静，避免某一个瞬间的声音影响判断。
    mean_energy = float(np.mean(silence_energies))

    # 停顿处需要比前后声音明显更安静。差距不够大时，它可能只是轻声讲话，
    # 直接忽略可以减少从词语中间切开的情况。
    if not _has_sufficient_energy_drop(mean_energy, context_energies):
        return None

    # 检查通过后，在整个停顿中选择最安静的位置作为切点，并保留停顿长度和
    # 平均声音大小，供后续逻辑比较多个候选位置。
    return CutCandidate(
        cut_frame=_find_quietest_cut_frame(interval, silence_energies),
        silence_duration_frames=interval.duration_frames,
        mean_energy=mean_energy,
    )


def _has_sufficient_energy_drop(
    silence_mean_energy: float,
    context_energies: NDArray[np.int64],
) -> bool:
    if context_energies.size == 0:
        return True
    context_mean_energy = float(np.mean(context_energies))
    if context_mean_energy == 0:
        return True
    return (
        silence_mean_energy / context_mean_energy <= MAX_SILENCE_TO_CONTEXT_ENERGY_RATIO
    )


def build_fallback_candidates(
    audio: NormalizedAudio,
    *,
    start_frame: int,
    end_frame: int,
    max_speech_frames: int,
    frame_samples: int,
) -> list[CutCandidate]:
    """为过长且缺少明显停顿的连续语音准备备用切分位置。

    当一大段话长时间没有自然停顿时，仍然需要把它分成长度合适的片段。
    这里会先估算需要分成几段，再围绕每个大致的分界点寻找候选位置，
    供后续流程选择，避免生成过长或长短差别太大的音频片段。
    """
    # 计算这段音频大约需要分成几段。长度已经合适时，无需准备备用切点。
    segment_frames = end_frame - start_frame
    segment_count = ceil(segment_frames / max_speech_frames)
    if segment_count <= 1:
        return []

    # 尽量把整段音频均匀分开，并给每个理想分界点留出一小段寻找空间。
    # 这样后续可以在附近选择更合适的位置，同时避免各段长度相差太多。
    balanced_segment_frames = ceil(segment_frames / segment_count)
    search_radius = (max_speech_frames - balanced_segment_frames) // 2
    candidates = []

    # 逐个处理相邻片段之间的分界点，并收集附近可用的备用位置。
    for cut_index in range(1, segment_count):
        target_frame = start_frame + segment_frames * cut_index // segment_count
        search_start = max(start_frame + 1, target_frame - search_radius)
        search_end = min(end_frame - 1, target_frame + search_radius)
        candidates.extend(
            _build_fallback_candidates_for_target(
                audio,
                target_frame=target_frame,
                search_start=search_start,
                search_end=search_end,
                frame_samples=frame_samples,
            )
        )
    return candidates


def _build_fallback_candidates_for_target(
    audio: NormalizedAudio,
    *,
    target_frame: int,
    search_start: int,
    search_end: int,
    frame_samples: int,
) -> list[CutCandidate]:
    """在一个预计分界点附近准备两个备用切分位置。

    连续说话中可能找不到自然停顿，直接切开又容易显得突兀。这里会同时找出
    附近最安静的位置，并保留原本预计的均匀分界点，交给后续流程统一比较。
    前者更照顾听感，后者可以避免切分后的音频片段长短相差太多。
    """
    # 切分位置位于两小段音频之间。为了比较每个位置有多安静，需要读取
    # 搜索范围内各个切分位置前面的声音。
    frame_energies = _read_frame_energies(
        audio,
        search_start - 1,
        search_end,
        frame_samples=frame_samples,
    )

    # 找出范围内最安静的位置。多个位置同样安静时，选择更靠近搜索范围
    # 中间的一个，减少切点过度偏向某一侧的情况。
    minimum_energy = frame_energies.min()
    quietest_offsets = np.flatnonzero(frame_energies == minimum_energy)
    middle_offset = (len(frame_energies) - 1) / 2
    quietest_offset = min(
        (int(offset) for offset in quietest_offsets),
        key=lambda offset: abs(offset - middle_offset),
    )

    # 原本预计的分界点也要保留，方便后续在“更安静”和“分得更均匀”之间选择。
    target_offset = target_frame - search_start
    return [
        _fallback_candidate(
            cut_frame=search_start + quietest_offset,
            energy=frame_energies[quietest_offset],
        ),
        _fallback_candidate(
            cut_frame=target_frame,
            energy=frame_energies[target_offset],
        ),
    ]


def _fallback_candidate(*, cut_frame: int, energy: np.int64) -> CutCandidate:
    return CutCandidate(
        cut_frame=cut_frame,
        silence_duration_frames=0,
        mean_energy=float(energy),
    )


def deduplicate_candidates(
    candidates: list[CutCandidate],
) -> list[CutCandidate]:
    """合并位于同一个音频切分位置的重复候选项。

    自然停顿、最安静位置和均匀分段位置可能碰巧指向同一处，因此每个位置
    最终只需要保留一个候选项。遇到重复时，优先保留停顿时间更长的候选；
    停顿时间相同时，再保留声音更安静的候选，让最终切分听起来更自然。
    """
    # 以切分位置作为唯一编号，方便快速找到这个位置之前是否已有候选项。
    candidates_by_frame: dict[int, CutCandidate] = {}
    for candidate in candidates:
        previous = candidates_by_frame.get(candidate.cut_frame)

        # 第一次遇到这个位置时直接保存。再次遇到时，只有新的候选更合适，
        # 才替换已经保存的候选；具体比较标准由 _candidate_rank 统一提供。
        if previous is None or _candidate_rank(candidate) > _candidate_rank(previous):
            candidates_by_frame[candidate.cut_frame] = candidate

    # 按照音频从前到后的顺序返回，方便后续流程依次处理所有切分位置。
    return [candidates_by_frame[frame] for frame in sorted(candidates_by_frame)]


def _candidate_rank(candidate: CutCandidate) -> tuple[int, float]:
    return candidate.silence_duration_frames, -candidate.mean_energy


def _find_silence_intervals(
    speech_frames: NDArray[np.bool_],
    *,
    search_start: int,
    search_end: int,
    min_silence_frames: int,
) -> list[SilenceInterval]:
    intervals: list[SilenceInterval] = []
    silence_start: int | None = None
    for frame_index in range(search_start, search_end):
        if not speech_frames[frame_index]:
            if silence_start is None:
                silence_start = frame_index
            continue
        _append_silence_interval(
            intervals,
            silence_start,
            frame_index,
            min_silence_frames=min_silence_frames,
        )
        silence_start = None

    _append_silence_interval(
        intervals,
        silence_start,
        search_end,
        min_silence_frames=min_silence_frames,
    )
    return intervals


def _append_silence_interval(
    intervals: list[SilenceInterval],
    start_frame: int | None,
    end_frame: int,
    *,
    min_silence_frames: int,
) -> None:
    if start_frame is not None and end_frame - start_frame >= min_silence_frames:
        intervals.append(SilenceInterval(start_frame, end_frame))


def _read_frame_energies(
    audio: NormalizedAudio,
    search_start_frame: int,
    search_end_frame: int,
    *,
    frame_samples: int,
) -> NDArray[np.int64]:
    """从音频数据中提取指定区间内每一帧的总能量值，返回一维能量数组"""
    samples = audio.read_samples(
        search_start_frame * frame_samples,
        search_end_frame * frame_samples,
    )
    frames = samples.reshape(-1, frame_samples).astype(np.int32)
    return np.abs(frames).sum(axis=1, dtype=np.int64)


def _find_quietest_cut_frame(
    interval: SilenceInterval,
    frame_energies: NDArray[np.int64],
) -> int:
    minimum_energy = frame_energies.min()
    quietest_offsets = np.flatnonzero(frame_energies == minimum_energy)
    middle_offset = (interval.duration_frames - 1) / 2
    quietest_offset = min(
        (int(offset) for offset in quietest_offsets),
        key=lambda offset: abs(offset - middle_offset),
    )
    return interval.start_frame + quietest_offset + 1
