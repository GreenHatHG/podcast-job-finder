from __future__ import annotations

from typing import Final

from podcast_job_finder.audio.transcription_format_audit import (
    FormattedTranscription,
    TextExcerpt,
    TranscriptionDeletion,
)


AUDIT_SUMMARY_TEMPLATE: Final = (
    "格式化审计通过：原文 {source_count} 字，删除 {deleted_count} 字（{ratio:.2%}），"
    "共 {deletion_count} 处；正文顺序、数字英文和删除比例校验均通过"
)
CHANGE_HEADER_TEMPLATE: Final = "[文本块 {chunk_index} | {segment_label}] 删除：{text}"
ORIGINAL_LABEL: Final = "原文："
FORMATTED_LABEL: Final = "改后："
SINGLE_SEGMENT_TEMPLATE: Final = "片段 {index}"
SEGMENT_RANGE_TEMPLATE: Final = "片段 {start}-{end}"
DELETED_TEXT_TEMPLATE: Final = "【{text}】"
ELLIPSIS: Final = "……"
LINE_BREAK_MARKER: Final = "↵"


def build_human_audit_report(transcription: FormattedTranscription) -> str:
    lines = [
        AUDIT_SUMMARY_TEMPLATE.format(
            source_count=transcription.source_content_character_count,
            deleted_count=transcription.deleted_content_character_count,
            ratio=transcription.deletion_ratio,
            deletion_count=len(transcription.deletions),
        )
    ]
    for deletion in transcription.deletions:
        lines.extend(_build_deletion_lines(deletion))
    return "\n".join(lines)


def _build_deletion_lines(deletion: TranscriptionDeletion) -> list[str]:
    return [
        CHANGE_HEADER_TEMPLATE.format(
            chunk_index=deletion.chunk_index,
            segment_label=_build_segment_label(deletion),
            text=_normalize_text(deletion.text),
        ),
        ORIGINAL_LABEL
        + _render_excerpt(
            deletion.original_excerpt,
            middle=DELETED_TEXT_TEMPLATE.format(text=_normalize_text(deletion.text)),
        ),
        FORMATTED_LABEL + _render_excerpt(deletion.formatted_excerpt),
    ]


def _build_segment_label(deletion: TranscriptionDeletion) -> str:
    segment_range = deletion.segment_index_range
    if segment_range.start == segment_range.end:
        return SINGLE_SEGMENT_TEMPLATE.format(index=segment_range.start)
    return SEGMENT_RANGE_TEMPLATE.format(
        start=segment_range.start,
        end=segment_range.end,
    )


def _render_excerpt(excerpt: TextExcerpt, *, middle: str = "") -> str:
    prefix = ELLIPSIS if excerpt.prefix_truncated else ""
    suffix = ELLIPSIS if excerpt.suffix_truncated else ""
    return (
        prefix
        + _normalize_text(excerpt.before)
        + middle
        + _normalize_text(excerpt.after)
        + suffix
    )


def _normalize_text(text: str) -> str:
    return (
        text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", LINE_BREAK_MARKER)
    )
