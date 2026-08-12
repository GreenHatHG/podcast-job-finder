from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Mapping

from podcast_job_finder.transcription.formatting.article import (
    build_transcription_article,
)
from podcast_job_finder.transcription.manifest import (
    TranscriptionManifestError,
    load_episode_transcription_manifest,
    parse_transcribed_segment,
)
from podcast_job_finder.transcription.models import TranscribedSpeechSegment


logger = logging.getLogger("podcast_job_finder.transcription.batch")


def can_restore_completed_transcription(  # pylint: disable=too-many-arguments
    *,
    transcription_path: Path,
    article_path: Path,
    quality_report_path: Path,
    segment_dir: Path,
    article_title: str,
    expected_metadata: Mapping[str, object],
) -> bool:
    if not all(
        path.is_file()
        for path in (
            transcription_path,
            article_path,
            quality_report_path,
        )
    ):
        return False
    try:
        manifest = load_episode_transcription_manifest(transcription_path)
        _validate_completed_transcription_artifacts(
            manifest.metadata,
            transcription_path=transcription_path,
            article_path=article_path,
            quality_report_path=quality_report_path,
            segment_dir=segment_dir,
            article_title=article_title,
        )
    except (
        OSError,
        TranscriptionManifestError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        logger.warning("读取完整音频转写清单失败，将继续处理：%s", error)
        return False
    return all(
        manifest.metadata.get(field_name) == expected_value
        for field_name, expected_value in expected_metadata.items()
    )


def _validate_completed_transcription_artifacts(  # pylint: disable=too-many-arguments
    metadata: Mapping[str, object],
    *,
    transcription_path: Path,
    article_path: Path,
    quality_report_path: Path,
    segment_dir: Path,
    article_title: str,
) -> None:
    raw_segments = metadata.get("segments")
    if not isinstance(raw_segments, list):
        raise ValueError("完整音频转写清单缺少 segments 数组。")
    if metadata.get("segment_count") != len(raw_segments):
        raise ValueError("完整音频转写清单中的 segment_count 已变化。")
    text = metadata.get("text")
    if not isinstance(text, str):
        raise ValueError("完整音频转写清单中的 text 必须是字符串。")
    audio_path_value = metadata.get("audio_path")
    if not isinstance(audio_path_value, str) or not _is_non_empty_file(
        Path(audio_path_value)
    ):
        raise ValueError("完整音频转写清单中的源音频文件无效。")
    _validate_quality_report(quality_report_path, len(raw_segments))
    article_text = article_path.read_text(encoding="utf-8")
    expected_article = build_transcription_article(title=article_title, body=text)
    if article_text != expected_article:
        raise ValueError("完整音频转写文章与清单内容不一致。")
    parsed_segments = _validate_manifest_segments(
        raw_segments,
        transcription_path=transcription_path,
        segment_dir=segment_dir,
    )
    if text != "\n".join(segment.text for segment in parsed_segments):
        raise ValueError("完整音频转写清单中的 text 与 segments 不一致。")


def _validate_manifest_segments(
    raw_segments: list[object],
    *,
    transcription_path: Path,
    segment_dir: Path,
) -> list[TranscribedSpeechSegment]:
    parsed_segments: list[TranscribedSpeechSegment] = []
    for index, raw_segment in enumerate(raw_segments):
        if not isinstance(raw_segment, dict):
            raise ValueError(f"完整音频转写清单中的片段无效：index={index}")
        parsed_segment = parse_transcribed_segment(
            raw_segment,
            path=transcription_path,
            index=index,
        )
        if parsed_segment.index != index + 1:
            raise ValueError(
                "完整音频转写片段编号不连续："
                f"expected_index={index + 1} actual_index={parsed_segment.index}"
            )
        parsed_segments.append(parsed_segment)
        _validate_manifest_segment_artifacts(
            raw_segment,
            expected_segment=parsed_segment,
            segment_dir=segment_dir,
        )
    return parsed_segments


def _validate_manifest_segment_artifacts(
    raw_segment: dict[str, object],
    *,
    expected_segment: TranscribedSpeechSegment,
    segment_dir: Path,
) -> None:
    audio_path = _require_artifact_path(raw_segment, "audio_path")
    transcription_path = _require_artifact_path(raw_segment, "transcription_path")
    if audio_path.resolve().parent != segment_dir.resolve():
        raise ValueError(
            f"完整音频转写片段不在预期目录中：index={expected_segment.index}"
        )
    if transcription_path.resolve() != audio_path.with_suffix(".json").resolve():
        raise ValueError(f"完整音频转写片段路径不对应：index={expected_segment.index}")
    if not _is_non_empty_file(audio_path) or not _is_non_empty_file(transcription_path):
        raise ValueError(f"完整音频转写片段文件无效：index={expected_segment.index}")
    segment_payload = _read_json_object(transcription_path)
    parsed_checkpoint = parse_transcribed_segment(
        segment_payload,
        path=transcription_path,
        index=expected_segment.index,
    )
    if parsed_checkpoint != expected_segment:
        raise ValueError(f"完整音频转写片段内容不一致：index={expected_segment.index}")


def _validate_quality_report(path: Path, expected_segment_count: int) -> None:
    payload = _read_json_object(path)
    if payload.get("segment_count") != expected_segment_count:
        raise ValueError("音频转写质量报告中的 segment_count 已变化。")


def _read_json_object(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 文件必须是对象：{path}")
    return payload


def _require_artifact_path(payload: dict[str, object], field_name: str) -> Path:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"完整音频转写片段缺少 {field_name}。")
    return Path(value)


def _is_non_empty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0
