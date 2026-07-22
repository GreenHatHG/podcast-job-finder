from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Final

import numpy as np
from numpy.typing import NDArray

from podcast_job_finder.audio.normalized_audio import NormalizedAudio


# 一段内容太长时，至少保留最长时长的四分之一，避免生成只有几秒的小片段。
MIN_CUT_POSITION_RATIO: Final = 0.25

# 靠近最长时长的 80% 时，片段长度最合适。更早或更晚的停顿仍可参与评分。
PREFERRED_CUT_POSITION_RATIO: Final = 0.8

# 少于 48 毫秒的静音通常属于 VAD 瞬时波动，不作为完整停顿参与评分。
MIN_SILENCE_DURATION_MS: Final = 48

# 停顿达到 300 毫秒后，继续变长不会增加“停顿完整度”的得分。
FULL_SILENCE_DURATION_MS: Final = 300

# 停顿是否完整占主要权重；安静程度和片段长度用于区分质量接近的停顿。
SILENCE_DURATION_SCORE_WEIGHT: Final = 0.5
SILENCE_ENERGY_SCORE_WEIGHT: Final = 0.25
SEGMENT_LENGTH_SCORE_WEIGHT: Final = 0.25


@dataclass(slots=True, frozen=True)
class _SilenceInterval:
    start_frame: int
    end_frame: int

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame


@dataclass(slots=True, frozen=True)
class _CutCandidate:
    interval: _SilenceInterval
    cut_frame: int
    mean_energy: float


def split_long_segments(
    segments: list[tuple[int, int]],
    *,
    speech_frames: NDArray[np.bool_],
    audio: NormalizedAudio,
    max_speech_frames: int,
    frame_samples: int,
) -> list[tuple[int, int]]:
    """把超过最长时间的片段优先从完整停顿处切开。"""
    split_segments: list[tuple[int, int]] = []
    for start_frame, end_frame in segments:
        current_start = start_frame
        while end_frame - current_start > max_speech_frames:
            cut_frame = _find_natural_cut_frame(
                current_start,
                speech_frames=speech_frames,
                audio=audio,
                max_speech_frames=max_speech_frames,
                frame_samples=frame_samples,
            )
            split_segments.append((current_start, cut_frame))
            current_start = cut_frame

        if end_frame > current_start:
            split_segments.append((current_start, end_frame))
    return split_segments


def _find_natural_cut_frame(
    segment_start: int,
    *,
    speech_frames: NDArray[np.bool_],
    audio: NormalizedAudio,
    max_speech_frames: int,
    frame_samples: int,
) -> int:
    """在片段允许的最长时间内，找出听起来最自然的切点。

    函数会收集范围内的连续静音，比较停顿长度、安静程度和切分后的片段长度，
    再从得分最高的停顿中选出具体切点。没有找到可靠停顿时，会选择整个搜索
    范围内声音最轻的一帧，尽量降低从词语中间切开的概率。
    """
    # 从片段播放到最长时长的四分之一后开始寻找，避免生成只有几秒的小片段。
    # 搜索终点就是这个片段允许达到的最长时间，切点不会超过该位置。
    search_start = segment_start + int(max_speech_frames * MIN_CUT_POSITION_RATIO)
    search_end = min(segment_start + max_speech_frames, len(speech_frames))

    # 配置使用毫秒方便理解，VAD 结果使用帧记录位置，因此需要先换算成帧数。
    # min_silence_frames 用来排除过短的静音，减少单帧误判参与评分。
    min_silence_frames = _milliseconds_to_frames(
        MIN_SILENCE_DURATION_MS,
        sample_rate=audio.sample_rate,
        frame_samples=frame_samples,
    )

    # full_silence_frames 表示多长的停顿可以获得完整的“停顿长度”得分。
    # 更长的停顿仍可参与选择，但不会继续增加这一项得分。
    full_silence_frames = _milliseconds_to_frames(
        FULL_SILENCE_DURATION_MS,
        sample_rate=audio.sample_rate,
        frame_samples=frame_samples,
    )

    # 一次读取搜索范围内每一帧的能量，后续判断停顿质量和回退切点都会使用。
    frame_energies = _read_frame_energies(
        audio,
        search_start,
        search_end,
        frame_samples=frame_samples,
    )

    # 把搜索范围内连续的 VAD 静音整理成候选停顿，并记录各自的具体切点和能量。
    candidates = _build_cut_candidates(
        speech_frames,
        frame_energies,
        search_start=search_start,
        search_end=search_end,
        min_silence_frames=min_silence_frames,
    )

    # 有可靠停顿时，综合停顿长度、安静程度和片段长度，返回总分最高的切点。
    if candidates:
        return _select_best_candidate(
            candidates,
            segment_start=segment_start,
            max_speech_frames=max_speech_frames,
            full_silence_frames=full_silence_frames,
        ).cut_frame

    # 没有可靠停顿时，在整个搜索范围内选择声音最轻的一帧作为备用切点。
    # argmin 返回能量列表中从 0 开始的下标，加上搜索起点可换回整段音频的位置；
    # 最后的加 1 表示在该帧完整播放后切开。
    return search_start + int(np.argmin(frame_energies)) + 1


def _read_frame_energies(
    audio: NormalizedAudio,
    search_start_frame: int,
    search_end_frame: int,
    *,
    frame_samples: int,
) -> NDArray[np.int64]:
    samples = audio.read_samples(
        search_start_frame * frame_samples,
        search_end_frame * frame_samples,
    )
    frames = samples.reshape(-1, frame_samples).astype(np.int32)
    return np.abs(frames).sum(axis=1, dtype=np.int64)


def _build_cut_candidates(
    speech_frames: NDArray[np.bool_],
    frame_energies: NDArray[np.int64],
    *,
    search_start: int,
    search_end: int,
    min_silence_frames: int,
) -> list[_CutCandidate]:
    intervals = _find_silence_intervals(
        speech_frames,
        search_start=search_start,
        search_end=search_end,
        min_silence_frames=min_silence_frames,
    )
    candidates = []
    for interval in intervals:
        # 静音区间使用整段音频中的位置，能量列表的下标则从 0 开始。
        # 搜索范围的起点在能量列表中对应下标 0，所以要先减去搜索起点。
        relative_start = interval.start_frame - search_start
        relative_end = interval.end_frame - search_start
        interval_energies = frame_energies[relative_start:relative_end]
        candidates.append(
            _CutCandidate(
                interval=interval,
                cut_frame=_find_quietest_cut_frame(interval, interval_energies),
                mean_energy=float(np.mean(interval_energies)),
            )
        )
    return candidates


def _find_quietest_cut_frame(
    interval: _SilenceInterval,
    interval_energies: NDArray[np.int64],
) -> int:
    """在选中的停顿里，找到声音最轻的位置作为实际切点。

    VAD 判断的静音只表示没有检测到人声，其中仍可能有呼吸声和环境噪声。
    选择能量最低的位置，可以减少切到这些声音的概率；如果多个位置同样安静，
    则选择最靠近停顿中间的位置。
    """
    # 找出这个停顿中的最低能量。数值越小，表示这一帧的声音越轻。
    minimum_energy = interval_energies.min()

    # 一个停顿里可能有多帧达到相同的最低能量，因此先保留全部候选位置。
    # 这些位置从 0 开始，只表示它们在当前停顿内部的相对位置。
    quietest_offsets = np.flatnonzero(interval_energies == minimum_energy)

    # 计算最靠近停顿中间的帧下标。4 帧的下标是 0、1、2、3，下标中心是 1.5；
    # 下面会选择下标 1，返回时再加 1，最终切点是正中间的第 2 帧边界。
    middle_offset = (interval.duration_frames - 1) / 2

    # 从所有最低能量位置中，选择距离停顿中点最近的一个。
    quietest_offset = min(
        (int(offset) for offset in quietest_offsets),
        key=lambda offset: abs(offset - middle_offset),
    )

    # 加上静音区间的开始位置，将停顿内的相对位置换回整段音频中的位置。
    # 再加 1 表示在这一帧播放结束后切开，让这一帧完整保留在前一个片段中。
    return interval.start_frame + quietest_offset + 1


def _find_silence_intervals(
    speech_frames: NDArray[np.bool_],
    *,
    search_start: int,
    search_end: int,
    min_silence_frames: int,
) -> list[_SilenceInterval]:
    intervals: list[_SilenceInterval] = []
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
    intervals: list[_SilenceInterval],
    start_frame: int | None,
    end_frame: int,
    *,
    min_silence_frames: int,
) -> None:
    if start_frame is not None and end_frame - start_frame >= min_silence_frames:
        intervals.append(_SilenceInterval(start_frame, end_frame))


def _select_best_candidate(
    candidates: list[_CutCandidate],
    *,
    segment_start: int,
    max_speech_frames: int,
    full_silence_frames: int,
) -> _CutCandidate:
    """综合比较所有候选停顿，返回最适合切分的一个。

    每个候选停顿会得到三项分数：停顿是否足够完整、声音是否足够轻，以及
    切分后的片段长度是否合适。三项分数都换算到 0 到 1 之间，再按权重相加。
    """
    # 找出候选停顿中的最低和最高平均能量，用来把不同录音音量下的能量值
    # 换算成统一的 0 到 1 分数。能量越低，得到的“安静程度”分数越高。
    min_energy = min(candidate.mean_energy for candidate in candidates)
    max_energy = max(candidate.mean_energy for candidate in candidates)

    def score(candidate: _CutCandidate) -> tuple[float, int]:
        # 停顿越长，越像一句话之间的完整停顿。达到 full_silence_frames 后
        # 这一项得到满分 1，继续变长也不会超过 1。
        duration_score = min(
            1.0,
            candidate.interval.duration_frames / full_silence_frames,
        )

        # 比较当前停顿与其他候选停顿的平均能量。最安静的候选得到 1 分，
        # 声音最大的候选得到 0 分，其余候选按能量高低换算到两者之间。
        energy_score = _normalize_quietness(
            candidate.mean_energy,
            min_energy=min_energy,
            max_energy=max_energy,
        )

        # 计算切点位于最长时长的什么位置。例如结果为 0.8，表示切分后的
        # 片段长度达到允许最长时长的 80%。
        length_ratio = (candidate.cut_frame - segment_start) / max_speech_frames

        # 片段长度越接近期望位置，长度分数越高；过早或过晚都会降低分数。
        length_score = _score_segment_length(length_ratio)

        # 按固定权重合并三项分数。停顿完整度占 50%，安静程度和片段长度
        # 各占 25%。
        total_score = (
            duration_score * SILENCE_DURATION_SCORE_WEIGHT
            + energy_score * SILENCE_ENERGY_SCORE_WEIGHT
            + length_score * SEGMENT_LENGTH_SCORE_WEIGHT
        )

        # 返回切点位置作为第二项比较条件。两个候选总分完全相同时，位置更靠后
        # 的候选胜出，使前一个片段尽量保留更多内容。
        return total_score, candidate.cut_frame

    # max 会调用 score 比较每个候选，并返回得分最高的候选停顿。
    return max(candidates, key=score)


def _score_segment_length(length_ratio: float) -> float:
    """根据切分后的片段长度，计算 0 到 1 之间的合理程度分数。

    length_ratio 表示片段实际长度占最长允许时长的比例。例如最长允许 30 秒，
    在 24 秒处切开时，比例就是 24 / 30 = 0.8，此时得到满分 1。
    """
    # 计算实际切分位置与期望位置之间的距离。期望比例是 0.8，因此比例为 0.8
    # 时距离为 0；切点提前或推后都会让距离增加。
    distance_from_preferred = abs(length_ratio - PREFERRED_CUT_POSITION_RATIO)

    # 用期望比例将距离换算成扣分。距离为 0 时得到 1 分；比例为 0 时距离是
    # 0.8，得到 0 分。例如比例为 0.4 时，最终得分是 0.5。
    # max 保证距离较大时分数最低为 0，不会出现负数。
    return max(0.0, 1.0 - distance_from_preferred / PREFERRED_CUT_POSITION_RATIO)


def _normalize_quietness(
    energy: float,
    *,
    min_energy: float,
    max_energy: float,
) -> float:
    """把候选停顿的能量换算成 0 到 1 之间的安静程度分数。

    energy 是当前候选停顿的平均能量，min_energy 和 max_energy 是所有候选
    停顿中的最低与最高平均能量。返回值越接近 1，表示当前停顿越安静。
    """
    # 所有候选的能量完全相同时，能量无法帮助区分优劣。统一返回满分，后续
    # 选择只由停顿完整度和片段长度决定，同时避免除以 0。
    if max_energy == min_energy:
        return 1.0

    # 先计算当前能量在最低值与最高值之间的位置：最低能量得到 0，最高能量
    # 得到 1，中间值按比例落在 0 到 1 之间。
    relative_energy = (energy - min_energy) / (max_energy - min_energy)

    # 能量越低表示越安静，因此用 1 减去相对能量，将方向反转：最低能量得到
    # 1 分，最高能量得到 0 分。
    return 1.0 - relative_energy


def _milliseconds_to_frames(
    duration_ms: int,
    *,
    sample_rate: int,
    frame_samples: int,
) -> int:
    return max(1, ceil(duration_ms * sample_rate / (1_000 * frame_samples)))
