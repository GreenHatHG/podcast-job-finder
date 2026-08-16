"""集中定义程序生成文件时使用的目录名称，并提供常用路径的构造函数。

这个模块只负责计算路径，不会创建目录或写入文件。真正需要保存文件的模块会在
写入前创建相应目录。
"""

from __future__ import annotations

from pathlib import Path
from typing import Final


# 所有默认输出都放在项目运行目录下的 output 中。下面的常量分别表示
# output 的不同用途，调用方可以直接看出一类文件应该保存在哪里。
OUTPUT_ROOT: Final = Path("output")

# episodes 按节目 ID 保存单集相关内容，包括原始音频、转写和公司提取检查点。
EPISODE_OUTPUT_DIR: Final = OUTPUT_ROOT / "episodes"

# feeds 保存每个 RSS 播客的节目清单 manifest.json。
FEED_OUTPUT_DIR: Final = OUTPUT_ROOT / "feeds"

# reports 按 feed_id 保存一次批量运行生成的转写报告和公司汇总。
REPORT_OUTPUT_DIR: Final = OUTPUT_ROOT / "reports"

# local 保存用户直接处理本地音频时产生的文件，与 RSS 单集输出分开。
LOCAL_AUDIO_SEGMENTS_OUTPUT_DIR: Final = OUTPUT_ROOT / "local" / "audio_segments"
LOCAL_TRANSCRIPTION_OUTPUT_DIR: Final = OUTPUT_ROOT / "local" / "transcription"

# 这些名称用在 output/episodes/<episode_id>/ 内部，用来区分原始音频、
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


def build_episode_output_dir(output_dir: Path, episode_id: str) -> Path:
    """返回一个节目的输出目录，例如 output/episodes/<episode_id>。

    ``output_dir`` 先转换成绝对路径，再追加 ``episode_id``。函数只返回计算后的
    路径，不会检查节目是否存在，也不会创建目录。
    """
    return output_dir.resolve() / episode_id


def build_episode_audio_dir(output_dir: Path, episode_id: str) -> Path:
    """返回一个节目保存原始音频和语音片段的 audio 目录。"""
    return build_episode_output_dir(output_dir, episode_id) / EPISODE_AUDIO_DIR_NAME


def build_episode_company_extraction_dir(
    episode_dir: Path,
    source_dir_name: str,
) -> Path:
    """返回一个节目保存公司提取检查点的目录。

    ``source_dir_name`` 表示用于提取公司的内容来源：页面正文使用 ``page``，
    音频转写使用 ``transcription``。两类检查点分开后，恢复任务时不会读错来源。
    """
    return episode_dir / COMPANY_EXTRACTION_DIR_NAME / source_dir_name


def build_feed_report_dir(feed_id: str, report_dir_name: str) -> Path:
    """返回一个播客保存指定类型批量报告的目录。

    ``feed_id`` 用于区分不同 RSS 播客，``report_dir_name`` 用于区分转写报告、
    公司提取明细和公司汇总。函数只组合路径，不会创建目录。
    """
    return REPORT_OUTPUT_DIR / feed_id / report_dir_name
