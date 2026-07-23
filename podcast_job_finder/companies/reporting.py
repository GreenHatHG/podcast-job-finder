from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final


OUTPUT_DIR: Final = "output"
OUTPUT_FILE_TEMPLATE: Final = "result_{pid}_{timestamp}.json"
SUMMARY_FILE_TEMPLATE: Final = "summary_{pid}_{timestamp}.json"
OUTPUT_STATUS_SUCCESS: Final = "success"


@dataclass(slots=True, frozen=True)
class PidReportData:
    pid: str
    model: str
    base_url: str | None
    total: int
    success: int
    failed: int
    episodes: list[dict]


def save_pid_reports(report_data: PidReportData) -> tuple[str, str]:
    return _save_result_file(report_data), _save_summary_file(report_data)


def _save_summary_file(report_data: PidReportData) -> str:
    output_path, created_at = _build_output_file_details(
        SUMMARY_FILE_TEMPLATE,
        report_data.pid,
    )
    companies = _aggregate_companies(report_data.episodes)
    report = _build_base_report(
        report_data=report_data,
        created_at=created_at,
        total_key="total_episodes",
        success_key="success_episodes",
        failed_key="failed_episodes",
    )
    report["unique_company_count"] = len(companies)
    report["companies"] = companies
    _write_report_json(output_path, report)
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


def _save_result_file(report_data: PidReportData) -> str:
    output_path, created_at = _build_output_file_details(
        OUTPUT_FILE_TEMPLATE,
        report_data.pid,
    )
    report = _build_base_report(
        report_data=report_data,
        created_at=created_at,
        total_key="total",
        success_key="success",
        failed_key="failed",
    )
    report["episodes"] = report_data.episodes
    _write_report_json(output_path, report)
    return output_path


def _build_output_file_details(template: str, pid: str) -> tuple[str, str]:
    now = datetime.now(tz=timezone.utc)
    timestamp_label = now.strftime("%Y%m%d_%H%M%S")
    created_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    return _build_output_path(template, pid, timestamp_label), created_at


def _build_output_path(template: str, pid: str, timestamp_label: str) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, template.format(pid=pid, timestamp=timestamp_label))


def _build_base_report(
    *,
    report_data: PidReportData,
    created_at: str,
    total_key: str,
    success_key: str,
    failed_key: str,
) -> dict[str, object]:
    return {
        "pid": report_data.pid,
        "model": report_data.model,
        "base_url": report_data.base_url,
        "created_at": created_at,
        total_key: report_data.total,
        success_key: report_data.success,
        failed_key: report_data.failed,
    }


def _write_report_json(path: str, payload: object) -> None:
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")
