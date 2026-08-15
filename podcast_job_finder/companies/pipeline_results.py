from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal, TypeAlias

from podcast_job_finder.companies.models import CompanyExtractionResult
from podcast_job_finder.episode.models import EpisodeResult

if TYPE_CHECKING:
    from podcast_job_finder.transcription.pipeline_results import (
        SuccessfulEpisodeTranscriptionResult,
    )


RESULT_STATUS_SUCCESS: Final = "success"
RESULT_STATUS_ERROR: Final = "error"
EPISODE_RESULT_INCOMPLETE_ERROR: Final = "节目流水线未生成完整结果。"


@dataclass(slots=True, frozen=True)
class BatchEpisodePipelineResult:
    episode_results: list[EpisodeResult]
    success_count: int
    fail_count: int


@dataclass(slots=True, frozen=True)
class SuccessfulCompanyEpisodeResult(EpisodeResult):
    extraction_result: CompanyExtractionResult
    transcription_result: SuccessfulEpisodeTranscriptionResult | None = None
    extraction_chunk_count: int | None = None
    candidate_company_count: int | None = None
    extraction_cached: bool | None = None
    status: Literal["success"] = field(
        init=False,
        default=RESULT_STATUS_SUCCESS,
    )

    def to_dict(self) -> dict[str, object]:
        if self.transcription_result is None:
            payload = EpisodeResult.to_dict(self)
        else:
            payload = self.transcription_result.to_dict()
            payload["status"] = self.status
        payload.update(
            {
                "companies": [
                    company.to_dict() for company in self.extraction_result.companies
                ],
                "filtered_count": self.extraction_result.filtered_count,
            }
        )
        if self.extraction_chunk_count is not None:
            payload["extraction_chunk_count"] = self.extraction_chunk_count
        if self.candidate_company_count is not None:
            payload["candidate_company_count"] = self.candidate_company_count
        if self.extraction_cached is not None:
            payload["extraction_cached"] = self.extraction_cached
        return payload


@dataclass(slots=True, frozen=True)
class FailedCompanyEpisodeResult(EpisodeResult):
    error: str
    transcription_result: SuccessfulEpisodeTranscriptionResult | None = None
    status: Literal["error"] = field(init=False, default=RESULT_STATUS_ERROR)

    def to_dict(self) -> dict[str, object]:
        if self.transcription_result is None:
            payload = EpisodeResult.to_dict(self)
        else:
            payload = self.transcription_result.to_dict()
            payload["status"] = self.status
        payload["error"] = self.error
        return payload


CompanyEpisodeResult: TypeAlias = (
    SuccessfulCompanyEpisodeResult | FailedCompanyEpisodeResult
)
