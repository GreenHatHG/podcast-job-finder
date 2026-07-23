from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Final


CONTENT_SECTION_TITLE: Final = "标题"
AUDIO_URL_SECTION_TITLE: Final = "音频 URL"
BODY_SECTION_TITLE: Final = "正文"
COMMENTS_SECTION_TITLE: Final = "评论"
NO_COMMENTS_TEXT: Final = "无评论"
TOP_LEVEL_COMMENT_TEMPLATE: Final = "评论 {index}｜作者：{author}｜时间：{created_at}"
REPLY_COMMENT_TEMPLATE: Final = "回复 {index}｜作者：{author}｜时间：{created_at}"


@dataclass(slots=True)
class CommentInfo:
    author: str
    created_at: str
    text: str
    replies: list["CommentInfo"] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_text_lines(self, index_label: str, indent_level: int = 0) -> list[str]:
        indent = "  " * indent_level
        header_template = (
            TOP_LEVEL_COMMENT_TEMPLATE if indent_level == 0 else REPLY_COMMENT_TEMPLATE
        )
        header_text = header_template.format(
            index=index_label,
            author=self.author,
            created_at=self.created_at,
        )
        lines = [
            f"{indent}{header_text}",
            f"{indent}{self.text}",
        ]
        for reply_index, reply in enumerate(self.replies, start=1):
            lines.extend(
                reply.to_text_lines(f"{index_label}.{reply_index}", indent_level + 1)
            )
        return lines


@dataclass(slots=True)
class EpisodeInfo:
    title: str
    content: str
    audio_url: str | None = None
    comments: list[CommentInfo] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_text(self) -> str:
        sections = [
            CONTENT_SECTION_TITLE,
            self.title,
            "",
            AUDIO_URL_SECTION_TITLE,
            self.audio_url or "",
            "",
            BODY_SECTION_TITLE,
            self.content,
            "",
            COMMENTS_SECTION_TITLE,
        ]
        if not self.comments:
            sections.append(NO_COMMENTS_TEXT)
            return "\n".join(sections)

        comment_lines: list[str] = []
        for comment_index, comment in enumerate(self.comments, start=1):
            comment_lines.extend(comment.to_text_lines(str(comment_index)))
        sections.append("\n".join(comment_lines))
        return "\n".join(sections)
