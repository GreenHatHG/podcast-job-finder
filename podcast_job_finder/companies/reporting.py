from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

from podcast_job_finder.episode.models import EpisodeResult
from podcast_job_finder.filesystem import (
    DEFAULT_FILE_CREATION_MODE,
    atomic_write_json,
)
from podcast_job_finder.timestamps import build_utc_timestamp
from podcast_job_finder.output_paths import (
    COMPANY_EXTRACTION_REPORT_DIR_NAME,
    COMPANY_SUMMARY_REPORT_DIR_NAME,
    build_feed_report_dir,
)


REPORT_FILE_TEMPLATE: Final = "{timestamp}.json"
OUTPUT_STATUS_SUCCESS: Final = "success"


@dataclass(slots=True, frozen=True)
class FeedReportData:
    feed_id: str
    model: str
    base_url: str | None
    total: int
    success: int
    failed: int
    episodes: Sequence[EpisodeResult]


def save_feed_reports(report_data: FeedReportData) -> tuple[str, str]:
    episode_payloads: list[dict] = [
        episode.to_dict() for episode in report_data.episodes
    ]
    return _save_result_file(report_data, episode_payloads), _save_summary_file(
        report_data,
        episode_payloads,
    )


def _save_summary_file(
    report_data: FeedReportData,
    episode_payloads: list[dict],
) -> str:
    output_path, created_at = _build_output_file_details(
        COMPANY_SUMMARY_REPORT_DIR_NAME,
        report_data.feed_id,
    )
    companies = _aggregate_companies(episode_payloads)
    report = _build_base_report(
        report_data=report_data,
        created_at=created_at,
        total_key="total_episodes",
        success_key="success_episodes",
        failed_key="failed_episodes",
    )
    report["unique_company_count"] = len(companies)
    report["companies"] = companies
    atomic_write_json(
        Path(output_path),
        report,
        mode=DEFAULT_FILE_CREATION_MODE,
    )
    return output_path


def _aggregate_companies(episodes: list[dict]) -> list[dict]:
    grouped: dict[str, dict] = {}
    for episode in episodes:
        if episode.get("status") != OUTPUT_STATUS_SUCCESS:
            continue
        episode_ref = {
            "eid": episode.get("eid"),
            "title": episode.get("title"),
            "pub_date": episode.get("pub_date"),
            "episode_url": episode.get("episode_url"),
        }
        for company in episode.get("companies", ()):
            raw_name = company.get("name", "")
            normalized_name = raw_name.strip()
            if not normalized_name:
                continue
            entry = grouped.setdefault(
                normalized_name,
                {"name": normalized_name, "occurrence_count": 0, "episodes": []},
            )
            entry["occurrence_count"] += 1
            entry["episodes"].append(
                {**episode_ref, "evidence": company.get("evidence", "")}
            )

    return sorted(
        grouped.values(),
        key=lambda item: (-item["occurrence_count"], item["name"]),
    )


def _save_result_file(
    report_data: FeedReportData,
    episode_payloads: list[dict],
) -> str:
    output_path, created_at = _build_output_file_details(
        COMPANY_EXTRACTION_REPORT_DIR_NAME,
        report_data.feed_id,
    )
    report = _build_base_report(
        report_data=report_data,
        created_at=created_at,
        total_key="total",
        success_key="success",
        failed_key="failed",
    )
    report["episodes"] = episode_payloads
    atomic_write_json(
        Path(output_path),
        report,
        mode=DEFAULT_FILE_CREATION_MODE,
    )
    return output_path


def _build_output_file_details(report_dir_name: str, feed_id: str) -> tuple[str, str]:
    timestamp = build_utc_timestamp()
    output_dir = build_feed_report_dir(feed_id, report_dir_name)
    output_path = str(
        output_dir / REPORT_FILE_TEMPLATE.format(timestamp=timestamp.file_label)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_path, timestamp.text


def _build_base_report(
    *,
    report_data: FeedReportData,
    created_at: str,
    total_key: str,
    success_key: str,
    failed_key: str,
) -> dict[str, object]:
    return {
        "feed_id": report_data.feed_id,
        "model": report_data.model,
        "base_url": report_data.base_url,
        "created_at": created_at,
        total_key: report_data.total,
        success_key: report_data.success,
        failed_key: report_data.failed,
    }
