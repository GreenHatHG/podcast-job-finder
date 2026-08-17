"""集中定义程序生成文件时使用的目录名称，并提供常用路径的构造函数。

这个模块只负责计算路径，不会创建目录或写入文件。真正需要保存文件的模块会在
写入前创建相应目录。
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Final


# 所有默认输出都放在项目运行目录下的 output 中。下面的常量分别表示
# output 的不同用途，调用方可以直接看出一类文件应该保存在哪里。
OUTPUT_ROOT: Final = Path("output")

# episodes 按“播客名-节目名-节目 ID”保存单集相关内容，包括原始音频、转写和
# 公司提取检查点。
EPISODE_OUTPUT_DIR: Final = OUTPUT_ROOT / "episodes"

# feeds 保存每个 RSS 播客的节目清单 manifest.json。
FEED_OUTPUT_DIR: Final = OUTPUT_ROOT / "feeds"

# reports 按“播客名-feed_id”保存一次批量运行生成的转写报告和公司汇总。
REPORT_OUTPUT_DIR: Final = OUTPUT_ROOT / "reports"

# local 保存用户直接处理本地音频时产生的文件，与 RSS 单集输出分开。
LOCAL_AUDIO_SEGMENTS_OUTPUT_DIR: Final = OUTPUT_ROOT / "local" / "audio_segments"
LOCAL_TRANSCRIPTION_OUTPUT_DIR: Final = OUTPUT_ROOT / "local" / "transcription"

# 这些名称用在 output/episodes/<播客名>-<节目名>-<节目ID>/ 内部，用来区分原始音频、
# 完整转写和公司提取结果。这里只保存目录名称，不包含 output 等上级路径。
EPISODE_AUDIO_DIR_NAME: Final = "audio"
EPISODE_TRANSCRIPTION_DIR_NAME: Final = "transcription"
COMPANY_EXTRACTION_DIR_NAME: Final = "company_extraction"

# 页面正文和音频转写都能用于提取公司，这两个名称把两种来源的检查点分开保存。
PAGE_EXTRACTION_DIR_NAME: Final = "page"
TRANSCRIPTION_EXTRACTION_DIR_NAME: Final = "transcription"

# 同一个播客可以生成三类批量报告，分别放入独立目录，避免文件混在一起。
TRANSCRIPTION_REPORT_DIR_NAME: Final = "transcription"
COMPANY_EXTRACTION_REPORT_DIR_NAME: Final = "company_extraction"
COMPANY_SUMMARY_REPORT_DIR_NAME: Final = "company_summary"

MAX_DIRECTORY_NAME_BYTES: Final = 240
PODCAST_TITLE_MAX_BYTES: Final = 60
INVALID_DIRECTORY_CHARACTERS: Final = re.compile(r'[\x00-\x1f<>:"/\\|?*]+')


def build_named_directory_name(
    *names: str | None,
    identifier: str,
    identifier_label: str | None = None,
) -> str:
    """返回包含可读名称和稳定 ID 的安全目录名。"""

    safe_identifier = _sanitize_directory_part(identifier, fallback="id")
    identifier_part = (
        f"{identifier_label}={safe_identifier}" if identifier_label else safe_identifier
    )
    name_parts = [_sanitize_directory_part(name, fallback="") for name in names if name]
    name_parts = [part for part in name_parts if part]
    if not name_parts:
        return _truncate_utf8(identifier_part, MAX_DIRECTORY_NAME_BYTES)

    identifier_bytes = len(identifier_part.encode("utf-8"))
    names_budget = MAX_DIRECTORY_NAME_BYTES - identifier_bytes - len(name_parts)
    if names_budget <= 0:
        return _truncate_utf8(identifier_part, MAX_DIRECTORY_NAME_BYTES)

    truncated_names: list[str] = []
    remaining_budget = names_budget
    for index, part in enumerate(name_parts):
        remaining_parts = len(name_parts) - index - 1
        if index == 0 and remaining_parts:
            part_budget = min(PODCAST_TITLE_MAX_BYTES, remaining_budget)
        else:
            part_budget = remaining_budget
        truncated_part = _truncate_utf8(part, part_budget).rstrip(" ._")
        if truncated_part:
            truncated_names.append(truncated_part)
            remaining_budget -= len(truncated_part.encode("utf-8"))
    return "-".join((*truncated_names, identifier_part))


def _sanitize_directory_part(value: str, *, fallback: str) -> str:
    normalized_value = unicodedata.normalize("NFC", value)
    safe_value = INVALID_DIRECTORY_CHARACTERS.sub("_", normalized_value).strip(" ._")
    return safe_value or fallback


def _truncate_utf8(value: str, max_bytes: int) -> str:
    encoded_value = value.encode("utf-8")
    if len(encoded_value) <= max_bytes:
        return value
    return encoded_value[:max_bytes].decode("utf-8", errors="ignore")


def build_episode_output_dir(
    output_dir: Path,
    episode_id: str,
    *,
    podcast_title: str | None = None,
    episode_title: str | None = None,
) -> Path:
    """返回一个节目的输出目录。

    ``output_dir`` 先转换成绝对路径，再追加“播客名-节目名-节目 ID”。函数只返回
    计算后的路径，不会检查节目是否存在，也不会创建目录。
    """
    directory_name = build_named_directory_name(
        podcast_title,
        episode_title,
        identifier=episode_id,
        identifier_label="eid",
    )
    return output_dir.resolve() / directory_name


def find_episode_output_dir(
    output_dir: Path,
    episode_id: str,
    *,
    podcast_title: str | None = None,
    episode_title: str | None = None,
) -> Path:
    """按名称构造目录；名称未知或变化时按末尾节目 ID 查找已有目录。"""

    expected_dir = build_episode_output_dir(
        output_dir,
        episode_id,
        podcast_title=podcast_title,
        episode_title=episode_title,
    )
    if expected_dir.exists():
        return expected_dir

    resolved_output_dir = output_dir.resolve()
    if not resolved_output_dir.is_dir():
        return expected_dir
    legacy_directory_name = build_named_directory_name(
        podcast_title,
        episode_title,
        identifier=episode_id,
    )
    for existing_name in (legacy_directory_name, episode_id):
        existing_dir = resolved_output_dir / existing_name
        if existing_dir.is_dir():
            return existing_dir
    id_suffix = f"-eid={episode_id}"
    matches = [
        path
        for path in resolved_output_dir.iterdir()
        if path.is_dir() and path.name.endswith(id_suffix)
    ]
    return matches[0] if len(matches) == 1 else expected_dir


def build_episode_audio_dir(
    output_dir: Path,
    episode_id: str,
    *,
    podcast_title: str | None = None,
    episode_title: str | None = None,
) -> Path:
    """返回一个节目保存原始音频和语音片段的 audio 目录。"""
    return (
        find_episode_output_dir(
            output_dir,
            episode_id,
            podcast_title=podcast_title,
            episode_title=episode_title,
        )
        / EPISODE_AUDIO_DIR_NAME
    )


def build_episode_company_extraction_dir(
    episode_dir: Path,
    source_dir_name: str,
) -> Path:
    """返回一个节目保存公司提取检查点的目录。

    ``source_dir_name`` 表示用于提取公司的内容来源：页面正文使用 ``page``，
    音频转写使用 ``transcription``。两类检查点分开后，恢复任务时不会读错来源。
    """
    return episode_dir / COMPANY_EXTRACTION_DIR_NAME / source_dir_name


def build_feed_report_dir(
    feed_id: str,
    report_dir_name: str,
    *,
    podcast_title: str = "podcast",
    output_dir: Path = REPORT_OUTPUT_DIR,
) -> Path:
    """返回一个播客保存指定类型批量报告的目录。

    播客名便于人工识别，``feed_id`` 用于防止重名，``report_dir_name`` 用于区分
    转写报告、公司提取明细和公司汇总。函数只组合路径，不会创建目录。
    """
    directory_name = build_named_directory_name(
        podcast_title,
        identifier=feed_id,
        identifier_label="feed",
    )
    return output_dir / directory_name / report_dir_name
