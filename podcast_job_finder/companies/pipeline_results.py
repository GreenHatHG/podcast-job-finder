from __future__ import annotations

from typing import Final

from podcast_job_finder.companies.episode_runner import CompletedEpisodeExtraction
from podcast_job_finder.episode.models import EpisodeWorkItem


RESULT_STATUS_SUCCESS: Final = "success"
RESULT_STATUS_ERROR: Final = "error"
EPISODE_RESULT_INCOMPLETE_ERROR: Final = "节目流水线未生成完整结果。"


def build_error_result_record(
    *,
    work_item: EpisodeWorkItem,
    error_message: str,
) -> dict:
    record = work_item.to_result_metadata()
    record.update(
        {
            "status": RESULT_STATUS_ERROR,
            "error": error_message,
        }
    )
    return record


def build_success_result_record(
    completed_extraction: CompletedEpisodeExtraction,
) -> dict:
    record = completed_extraction.episode.to_result_metadata()
    record.update(
        {
            "status": RESULT_STATUS_SUCCESS,
            "companies": [
                company.to_dict()
                for company in completed_extraction.extraction_result.companies
            ],
            "filtered_count": completed_extraction.extraction_result.filtered_count,
        }
    )
    return record
