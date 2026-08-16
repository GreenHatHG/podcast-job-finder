from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass
from typing import Final, Sequence, TypeAlias

from podcast_job_finder.errors import PodcastJobFinderError

MAX_DELETED_CONTENT_RATIO: Final = 0.05
DELETION_CONTEXT_CHARS: Final = 20
INVALID_FORMATTED_TEXT_ERROR: Final = "模型新增、改写或调整了正文字符顺序。"
PROTECTED_CONTENT_DELETED_ERROR: Final = "模型删除了数字或英文字符。"
EXCESSIVE_DELETION_ERROR_TEMPLATE: Final = (
    "模型删除正文比例过高：deleted={deleted} source={source} ratio={ratio:.2%}"
)
MARKDOWN_CODE_FENCE_PATTERN: Final = re.compile(r"^[ \t]*`{3,}[^`\r\n]*$")
MARKDOWN_HEADING_PATTERN: Final = re.compile(r"^[ \t]*#+[ \t]*")

DiffOpcode: TypeAlias = tuple[str, int, int, int, int]


class TranscriptionFormattingValidationError(PodcastJobFinderError, ValueError):
    """模型返回内容超出了允许的修改范围。"""


@dataclass(slots=True, frozen=True)
class TextExcerpt:
    before: str
    after: str
    prefix_truncated: bool
    suffix_truncated: bool


@dataclass(slots=True, frozen=True)
class SourceSegmentRange:
    index: int
    start: int
    end: int


@dataclass(slots=True, frozen=True)
class IndexRange:
    start: int
    end: int


@dataclass(slots=True, frozen=True)
class TranscriptionDeletion:
    chunk_index: int
    segment_index_range: IndexRange
    chunk_character_range: IndexRange
    text: str
    original_excerpt: TextExcerpt
    formatted_excerpt: TextExcerpt

    @property
    def character_count(self) -> int:
        content, _ = _extract_content(self.text)
        return len(content)

    def to_machine_dict(self) -> dict[str, object]:
        return {
            "chunk_index": self.chunk_index,
            "segment_index_start": self.segment_index_range.start,
            "segment_index_end": self.segment_index_range.end,
            "chunk_character_start": self.chunk_character_range.start,
            "chunk_character_end": self.chunk_character_range.end,
            "text": self.text,
            "character_count": self.character_count,
        }


@dataclass(slots=True, frozen=True)
class FormattedTranscription:
    text: str
    deletions: tuple[TranscriptionDeletion, ...]
    source_content_character_count: int = 0
    deleted_content_character_count: int = 0

    @property
    def deletion_ratio(self) -> float:
        if not self.source_content_character_count:
            return 0.0
        return (
            self.deleted_content_character_count / self.source_content_character_count
        )

    def to_machine_audit_dict(self) -> dict[str, object]:
        return {
            "source_content_character_count": self.source_content_character_count,
            "deleted_content_character_count": self.deleted_content_character_count,
            "deletion_ratio": self.deletion_ratio,
            "deletion_count": len(self.deletions),
            "deletions": [deletion.to_machine_dict() for deletion in self.deletions],
        }


FormattedChunk: TypeAlias = FormattedTranscription


@dataclass(slots=True, frozen=True)
class _DeletionSource:
    text: str
    content_positions: tuple[int, ...]
    chunk_index: int
    segment_ranges: tuple[SourceSegmentRange, ...]


def analyze_formatted_text(
    original_text: str,
    formatted_text: str,
    *,
    chunk_index: int,
    source_segment_ranges: Sequence[SourceSegmentRange],
) -> FormattedChunk:
    normalized_text = _normalize_plain_text(formatted_text)
    original_content, original_positions = _extract_content(original_text)
    formatted_content, formatted_positions = _extract_content(normalized_text)
    opcodes = difflib.SequenceMatcher(
        isjunk=None,
        a=original_content,
        b=formatted_content,
        autojunk=False,
    ).get_opcodes()
    deleted_character_count = _validate_diff(opcodes, original_content)
    deletion_source = _DeletionSource(
        text=original_text,
        content_positions=original_positions,
        chunk_index=chunk_index,
        segment_ranges=tuple(source_segment_ranges),
    )
    return FormattedChunk(
        text=normalized_text,
        deletions=_build_deletions(
            opcodes,
            deletion_source,
            formatted_text=normalized_text,
            formatted_positions=formatted_positions,
        ),
        source_content_character_count=len(original_content),
        deleted_content_character_count=deleted_character_count,
    )


def _validate_diff(
    opcodes: Sequence[DiffOpcode],
    original_content: str,
) -> int:
    deleted_character_count = 0
    for tag, original_start, original_end, _, _ in opcodes:
        if tag in {"insert", "replace"}:
            raise TranscriptionFormattingValidationError(INVALID_FORMATTED_TEXT_ERROR)
        if tag != "delete":
            continue
        deleted_content = original_content[original_start:original_end]
        _validate_deleted_content(deleted_content)
        deleted_character_count += len(deleted_content)
    _validate_deletion_ratio(deleted_character_count, len(original_content))
    return deleted_character_count


def _validate_deleted_content(deleted_content: str) -> None:
    if any(_is_protected_character(character) for character in deleted_content):
        raise TranscriptionFormattingValidationError(PROTECTED_CONTENT_DELETED_ERROR)


def _build_deletions(
    opcodes: Sequence[DiffOpcode],
    source: _DeletionSource,
    *,
    formatted_text: str,
    formatted_positions: Sequence[int],
) -> tuple[TranscriptionDeletion, ...]:
    return tuple(
        _build_deletion(
            source,
            content_start,
            content_end,
            formatted_text=formatted_text,
            formatted_position=_content_boundary_position(
                formatted_text,
                formatted_positions,
                formatted_start,
            ),
        )
        for tag, content_start, content_end, formatted_start, _ in opcodes
        if tag == "delete"
    )


def _normalize_plain_text(formatted_text: str) -> str:
    normalized_lines = (
        MARKDOWN_HEADING_PATTERN.sub("", line, count=1)
        for line in formatted_text.splitlines()
        if MARKDOWN_CODE_FENCE_PATTERN.fullmatch(line) is None
    )
    return "\n".join(normalized_lines).strip()


def _extract_content(text: str) -> tuple[str, tuple[int, ...]]:
    characters: list[str] = []
    source_positions: list[int] = []
    for position, character in enumerate(text):
        if character.isspace() or unicodedata.category(character).startswith("P"):
            continue
        characters.append(character)
        source_positions.append(position)
    return "".join(characters), tuple(source_positions)


def _is_protected_character(character: str) -> bool:
    return character.isdigit() or (character.isascii() and character.isalpha())


def _validate_deletion_ratio(deleted_chars: int, source_chars: int) -> None:
    ratio = deleted_chars / source_chars if source_chars else 0.0
    if ratio <= MAX_DELETED_CONTENT_RATIO:
        return
    raise TranscriptionFormattingValidationError(
        EXCESSIVE_DELETION_ERROR_TEMPLATE.format(
            deleted=deleted_chars,
            source=source_chars,
            ratio=ratio,
        )
    )


def _build_deletion(
    source: _DeletionSource,
    content_start: int,
    content_end: int,
    *,
    formatted_text: str,
    formatted_position: int,
) -> TranscriptionDeletion:
    start_position = source.content_positions[content_start]
    end_position = source.content_positions[content_end - 1] + 1
    segment_index_start = _find_segment_index(source.segment_ranges, start_position)
    segment_index_end = _find_segment_index(source.segment_ranges, end_position - 1)
    deleted_text = source.text[start_position:end_position]
    return TranscriptionDeletion(
        chunk_index=source.chunk_index,
        segment_index_range=IndexRange(
            start=segment_index_start,
            end=segment_index_end,
        ),
        chunk_character_range=IndexRange(
            start=start_position,
            end=end_position,
        ),
        text=deleted_text,
        original_excerpt=_build_text_excerpt(
            source.text,
            start=start_position,
            end=end_position,
        ),
        formatted_excerpt=_build_text_excerpt(
            formatted_text,
            start=formatted_position,
            end=formatted_position,
        ),
    )


def _content_boundary_position(
    text: str,
    content_positions: Sequence[int],
    boundary: int,
) -> int:
    if boundary >= len(content_positions):
        return len(text)
    return content_positions[boundary]


def _find_segment_index(
    segment_ranges: Sequence[SourceSegmentRange],
    position: int,
) -> int:
    for segment_range in segment_ranges:
        if segment_range.start <= position < segment_range.end:
            return segment_range.index
    raise ValueError(f"删除位置无法对应到转写片段：position={position}")


def _build_text_excerpt(text: str, *, start: int, end: int) -> TextExcerpt:
    excerpt_start = max(0, start - DELETION_CONTEXT_CHARS)
    excerpt_end = min(len(text), end + DELETION_CONTEXT_CHARS)
    return TextExcerpt(
        before=text[excerpt_start:start],
        after=text[end:excerpt_end],
        prefix_truncated=excerpt_start > 0,
        suffix_truncated=excerpt_end < len(text),
    )
