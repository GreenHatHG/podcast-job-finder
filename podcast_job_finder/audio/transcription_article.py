from __future__ import annotations

from pathlib import Path
from typing import Final

from podcast_job_finder.filesystem import (
    DEFAULT_FILE_CREATION_MODE,
    atomic_write_text,
)


DEFAULT_ARTICLE_TITLE: Final = "音频转写"
TRANSCRIPTION_ARTICLE_FILE_NAME: Final = "transcription.md"
FORMATTED_TRANSCRIPTION_ARTICLE_FILE_NAME: Final = "transcription_formatted.md"


def save_transcription_article(
    path: Path,
    *,
    title: str,
    body: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        path,
        build_transcription_article(title=title, body=body),
        mode=DEFAULT_FILE_CREATION_MODE,
    )


def build_transcription_article(
    *,
    title: str,
    body: str,
) -> str:
    article_title = _normalize_text(title) or DEFAULT_ARTICLE_TITLE
    article_body = body.strip()
    return f"# {article_title}\n\n{article_body}\n"


def _normalize_text(text: str) -> str:
    return " ".join(text.split())
