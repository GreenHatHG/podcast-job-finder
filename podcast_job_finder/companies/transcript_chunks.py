from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Sequence

from podcast_job_finder.audio.transcription import TranscribedSpeechSegment


DEFAULT_TRANSCRIPT_CHUNK_MAX_CHARS: Final = 20_000
DEFAULT_TRANSCRIPT_CHUNK_OVERLAP_SEGMENTS: Final = 1
INVALID_CHUNK_SIZE_ERROR: Final = "音频转写文本块的最大字符数必须大于 0。"
INVALID_OVERLAP_ERROR: Final = "音频转写文本块的重叠片段数不能小于 0。"
OVERSIZED_SEGMENT_ERROR: Final = (
    "单个音频转写片段超过文本块字符上限："
    "index={index} chars={chars} max_chars={max_chars}"
)


@dataclass(slots=True, frozen=True)
class TranscriptChunk:
    index: int
    segments: tuple[TranscribedSpeechSegment, ...]

    @property
    def text(self) -> str:
        return "\n".join(segment.text for segment in self.segments)


def build_transcript_chunks(
    segments: Sequence[TranscribedSpeechSegment],
    *,
    max_chars: int = DEFAULT_TRANSCRIPT_CHUNK_MAX_CHARS,
    overlap_segment_count: int = DEFAULT_TRANSCRIPT_CHUNK_OVERLAP_SEGMENTS,
) -> list[TranscriptChunk]:
    """把一整段播客转写文本拆成若干个小块，方便喂给后续处理（比如抽取公司信息）。

    为什么需要拆分：
        一期播客转写出来可能有几万字，一次性塞给下游处理往往超长或效果变差。
        所以这里像切面包一样，一段一段往后累加，累计接近字符上限就切一刀，
        输出一个个独立的文本块，每块的长度都不超过 ``max_chars``。
    """
    # 参数合法性校验：上限不能设成 0 或负数，重叠段数也不能是负数
    if max_chars <= 0:
        raise ValueError(INVALID_CHUNK_SIZE_ERROR)
    if overlap_segment_count < 0:
        raise ValueError(INVALID_OVERLAP_ERROR)
    # 没有内容就不用拆，直接返回空列表
    if not segments:
        return []

    # 这里就是在“切面包”：一段段往 current_segments 里加，
    # 加到快超上限时就切一刀，存进 chunks，然后开始下一块。
    chunks: list[TranscriptChunk] = []
    current_segments: list[TranscribedSpeechSegment] = []  # 当前这一块正在累积的片段
    current_chars = 0  # 当前这一块已经累积了多少字符
    for segment in segments:
        # 先检查这个片段本身是不是就已经超长（连单独放一块都不够装）
        _validate_segment_size(segment, max_chars=max_chars)
        # 新片段加进来时，前面要加一个换行符把它和已有点内容隔开，
        # 这里算一下这个换行符占不占 1 个字符：当前块已经有内容就算 1，没有就算 0
        separator_chars = 1 if current_segments else 0
        if (
            current_segments
            and current_chars + separator_chars + len(segment.text) > max_chars
        ):
            # 如果加上当前的这个片段(segment.text)就会超过字符上限，先把当前累积的内容切成一块输出，
            # 不包括当前这个片段
            chunks.append(_build_chunk(chunks, current_segments))
            # 然后准备下一块：从刚才那块末尾借几段作为新块的开头（实现重叠）
            current_segments = _build_next_chunk_prefix(
                previous_segments=current_segments,
                next_segment=segment,
                overlap_segment_count=overlap_segment_count,
                max_chars=max_chars,
            )
            current_chars = _count_segment_chars(current_segments)
        # 把当前片段真正加进当前块里，并累加它的字符数
        current_segments.append(segment)
        # 这里的 (1 if current_chars else 0) 同样是“换行符占的 1 个字符”：
        # 加进来之前 current_chars 如果不为 0，说明前面已有内容，需要 1 个换行符隔开
        current_chars += (1 if current_chars else 0) + len(segment.text)

    # 循环走完后，最后一块如果还有内容，别忘了也输出出去
    if current_segments:
        chunks.append(_build_chunk(chunks, current_segments))
    return chunks


def _build_chunk(
    existing_chunks: Sequence[TranscriptChunk],
    segments: Sequence[TranscribedSpeechSegment],
) -> TranscriptChunk:
    return TranscriptChunk(
        index=len(existing_chunks) + 1,
        segments=tuple(segments),
    )


def _count_segment_chars(segments: Sequence[TranscribedSpeechSegment]) -> int:
    if not segments:
        return 0
    return sum(len(segment.text) for segment in segments) + len(segments) - 1


def _take_overlap_segments(
    segments: Sequence[TranscribedSpeechSegment],
    overlap_segment_count: int,
) -> list[TranscribedSpeechSegment]:
    if overlap_segment_count == 0:
        return []
    return list(segments[-overlap_segment_count:])


def _validate_segment_size(
    segment: TranscribedSpeechSegment,
    *,
    max_chars: int,
) -> None:
    if len(segment.text) <= max_chars:
        return
    raise ValueError(
        OVERSIZED_SEGMENT_ERROR.format(
            index=segment.index,
            chars=len(segment.text),
            max_chars=max_chars,
        )
    )


def _build_next_chunk_prefix(
    *,
    previous_segments: Sequence[TranscribedSpeechSegment],
    next_segment: TranscribedSpeechSegment,
    overlap_segment_count: int,
    max_chars: int,
) -> list[TranscribedSpeechSegment]:
    overlap_segments = _take_overlap_segments(
        previous_segments,
        overlap_segment_count,
    )
    overlap_chars = _count_segment_chars(overlap_segments)
    separator_chars = 1 if overlap_segments else 0
    if overlap_chars + separator_chars + len(next_segment.text) <= max_chars:
        return overlap_segments
    return []
