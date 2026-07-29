from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Final, Protocol, Sequence

from podcast_job_finder.audio.transcription import TranscribedSpeechSegment
from podcast_job_finder.audio.transcription_format_audit import (
    FormattedChunk,
    FormattedTranscription,
    SourceSegmentRange,
    TranscriptionFormattingValidationError,
    analyze_formatted_text,
)
from podcast_job_finder.audio.transcription_format_report import (
    build_human_audit_report,
)
from podcast_job_finder.llm import (
    EmptyLlmResponseError,
    LlmRetryConfig,
    LlmRetryExhaustedError,
    OpenAiCompatibleLlmError,
    RetryableOpenAiCompatibleLlmError,
    execute_llm_with_retry,
)


FORMATTING_CHUNK_MAX_CHARS: Final = 4_000
FORMATTING_CONTEXT_SEGMENT_COUNT: Final = 2
FORMATTING_CONTEXT_MAX_CHARS: Final = 800
FORMATTING_PROMPT_TEMPLATE: Final = """你是中文音频转写编辑。

请整理输入 JSON 中的 current_text，并遵守以下要求：
1. 修复片段边界造成的错误断句，调整标点、空格和自然段。
2. 删除不影响原意的口头语、语气词和无意义重复。
3. 正文删除量不得超过原文的 5%。
4. 保留所有数字、英文字符、事实和有效表达，保持正文顺序。
5. previous_context 和 following_context 只供理解上下文，禁止输出。
6. 只输出整理后的 current_text 正文，禁止添加标题、解释、代码块或标签。
7. 禁止改写、纠错、概括、补充或重复内容。

输入 JSON：
{input_json}
"""
FORMAT_OPERATION_NAME_TEMPLATE: Final = "整理音频转写文本块 {index}/{total}"
FORMAT_RETRY_EXHAUSTED_TEMPLATE: Final = (
    "音频转写文本块 {index}/{total} 连续 {max_attempts} 次整理失败，"
    "最后一次错误：{error_message}"
)
EMPTY_TRANSCRIPTION_ERROR: Final = "没有可供模型整理的音频转写文本。"
OVERSIZED_SEGMENT_ERROR: Final = (
    "单个音频转写片段超过模型整理字符上限："
    "index={index} chars={chars} max_chars={max_chars}"
)

logger = logging.getLogger(__name__)


class TextGenerationClientProtocol(Protocol):
    def generate(self, prompt: str) -> str: ...


class TranscriptionFormattingError(RuntimeError):
    """音频转写文本无法在受限修改规则内完成整理。"""


EXPECTED_TRANSCRIPTION_FORMATTING_ERRORS: Final = (
    OpenAiCompatibleLlmError,
    TranscriptionFormattingError,
)


@dataclass(slots=True, frozen=True)
class _FormattingChunk:
    index: int
    segment_ranges: tuple[SourceSegmentRange, ...]
    current_text: str
    previous_context: str
    following_context: str


def format_transcription_segments(
    segments: Sequence[TranscribedSpeechSegment],
    *,
    llm_client: TextGenerationClientProtocol,
    retry_config: LlmRetryConfig | None = None,
) -> FormattedTranscription:
    ordered_segments = tuple(
        segment
        for segment in sorted(segments, key=lambda item: (item.start_ms, item.index))
        if segment.text.strip()
    )
    if not ordered_segments:
        raise TranscriptionFormattingError(EMPTY_TRANSCRIPTION_ERROR)
    chunks = _build_formatting_chunks(ordered_segments)
    formatted_chunks = [
        _format_chunk(
            chunk,
            total_chunks=len(chunks),
            llm_client=llm_client,
            retry_config=retry_config,
        )
        for chunk in chunks
    ]

    transcription = _merge_formatted_chunks(formatted_chunks)
    logger.info("%s", build_human_audit_report(transcription))
    return transcription


def _merge_formatted_chunks(
    chunks: Sequence[FormattedChunk],
) -> FormattedTranscription:
    return FormattedTranscription(
        text="\n\n".join(chunk.text for chunk in chunks),
        deletions=tuple(deletion for chunk in chunks for deletion in chunk.deletions),
        source_content_character_count=sum(
            chunk.source_content_character_count for chunk in chunks
        ),
        deleted_content_character_count=sum(
            chunk.deleted_content_character_count for chunk in chunks
        ),
    )


def _build_formatting_chunks(
    segments: Sequence[TranscribedSpeechSegment],
) -> list[_FormattingChunk]:
    ranges = _build_chunk_ranges(segments)
    chunks: list[_FormattingChunk] = []
    for index, (start, end) in enumerate(ranges, start=1):
        context_start = max(0, start - FORMATTING_CONTEXT_SEGMENT_COUNT)
        context_end = min(len(segments), end + FORMATTING_CONTEXT_SEGMENT_COUNT)
        current_text, segment_ranges = _build_chunk_text_and_ranges(segments[start:end])
        chunks.append(
            _FormattingChunk(
                index=index,
                segment_ranges=segment_ranges,
                current_text=current_text,
                previous_context=_build_context(
                    segments[context_start:start],
                    keep_end=True,
                ),
                following_context=_build_context(
                    segments[end:context_end],
                    keep_end=False,
                ),
            )
        )
    return chunks


def _build_chunk_ranges(
    segments: Sequence[TranscribedSpeechSegment],
) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    chunk_start = 0
    chunk_chars = 0
    for position, segment in enumerate(segments):
        segment_chars = len(segment.text.strip())
        if segment_chars > FORMATTING_CHUNK_MAX_CHARS:
            raise TranscriptionFormattingError(
                OVERSIZED_SEGMENT_ERROR.format(
                    index=segment.index,
                    chars=segment_chars,
                    max_chars=FORMATTING_CHUNK_MAX_CHARS,
                )
            )
        separator_chars = 1 if chunk_chars else 0
        if chunk_chars and (
            chunk_chars + separator_chars + segment_chars > FORMATTING_CHUNK_MAX_CHARS
        ):
            ranges.append((chunk_start, position))
            chunk_start = position
            chunk_chars = 0
            separator_chars = 0
        chunk_chars += separator_chars + segment_chars
    if chunk_start < len(segments):
        ranges.append((chunk_start, len(segments)))
    return ranges


def _join_segment_texts(
    segments: Sequence[TranscribedSpeechSegment],
) -> str:
    return "\n".join(segment.text.strip() for segment in segments)


def _build_chunk_text_and_ranges(
    segments: Sequence[TranscribedSpeechSegment],
) -> tuple[str, tuple[SourceSegmentRange, ...]]:
    text_parts: list[str] = []
    segment_ranges: list[SourceSegmentRange] = []
    current_position = 0
    for segment in segments:
        if text_parts:
            current_position += 1
        segment_text = segment.text.strip()
        segment_start = current_position
        text_parts.append(segment_text)
        current_position += len(segment_text)
        segment_ranges.append(
            SourceSegmentRange(
                index=segment.index,
                start=segment_start,
                end=current_position,
            )
        )
    return "\n".join(text_parts), tuple(segment_ranges)


def _build_context(
    segments: Sequence[TranscribedSpeechSegment],
    *,
    keep_end: bool,
) -> str:
    text = _join_segment_texts(segments)
    if len(text) <= FORMATTING_CONTEXT_MAX_CHARS:
        return text
    if keep_end:
        return text[-FORMATTING_CONTEXT_MAX_CHARS:]
    return text[:FORMATTING_CONTEXT_MAX_CHARS]


def _format_chunk(
    chunk: _FormattingChunk,
    *,
    total_chunks: int,
    llm_client: TextGenerationClientProtocol,
    retry_config: LlmRetryConfig | None,
) -> FormattedChunk:
    operation_name = FORMAT_OPERATION_NAME_TEMPLATE.format(
        index=chunk.index,
        total=total_chunks,
    )
    logger.info("%s", operation_name)
    prompt = _build_formatting_prompt(chunk)
    try:
        formatted_text, _ = execute_llm_with_retry(
            lambda: _generate_and_validate(
                llm_client,
                prompt=prompt,
                original_text=chunk.current_text,
                chunk_index=chunk.index,
                source_segment_ranges=chunk.segment_ranges,
            ),
            retry_config=retry_config,
            retryable_errors=(
                RetryableOpenAiCompatibleLlmError,
                EmptyLlmResponseError,
                TranscriptionFormattingValidationError,
            ),
            operation_name=operation_name,
        )
        return formatted_text
    except LlmRetryExhaustedError as error:
        raise TranscriptionFormattingError(
            FORMAT_RETRY_EXHAUSTED_TEMPLATE.format(
                index=chunk.index,
                total=total_chunks,
                max_attempts=error.max_attempts,
                error_message=str(error.last_error),
            )
        ) from error


def _build_formatting_prompt(chunk: _FormattingChunk) -> str:
    input_json = json.dumps(
        {
            "previous_context": chunk.previous_context,
            "current_text": chunk.current_text,
            "following_context": chunk.following_context,
        },
        ensure_ascii=False,
        indent=2,
    )
    return FORMATTING_PROMPT_TEMPLATE.format(input_json=input_json)


def _generate_and_validate(
    llm_client: TextGenerationClientProtocol,
    *,
    prompt: str,
    original_text: str,
    chunk_index: int,
    source_segment_ranges: Sequence[SourceSegmentRange],
) -> FormattedChunk:
    formatted_text = llm_client.generate(prompt).strip()
    return analyze_formatted_text(
        original_text,
        formatted_text,
        chunk_index=chunk_index,
        source_segment_ranges=source_segment_ranges,
    )
