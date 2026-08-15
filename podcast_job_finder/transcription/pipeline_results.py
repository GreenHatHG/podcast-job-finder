"""音频转写流水线的共享结果类型和状态。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal, TypeAlias

from podcast_job_finder.episode.models import EpisodeResult


RESULT_STATUS_SUCCESS: Final = "success"
RESULT_STATUS_ERROR: Final = "error"


@dataclass(slots=True, frozen=True)
class SuccessfulEpisodeTranscriptionResult(EpisodeResult):
    cached: bool
    transcription_path: str
    article_path: str | None = None
    transcription_quality_report_path: str | None = None
    segment_directory: str | None = None
    status: Literal["success"] = field(
        init=False,
        default=RESULT_STATUS_SUCCESS,
    )

    def to_dict(self) -> dict[str, object]:
        payload = EpisodeResult.to_dict(self)
        payload.update(
            {
                "cached": self.cached,
                "transcription_path": self.transcription_path,
            }
        )
        if self.article_path is not None:
            payload["article_path"] = self.article_path
        if self.transcription_quality_report_path is not None:
            payload["transcription_quality_report_path"] = (
                self.transcription_quality_report_path
            )
        if self.segment_directory is not None:
            payload["segment_directory"] = self.segment_directory
        return payload


@dataclass(slots=True, frozen=True)
class FailedEpisodeTranscriptionResult(EpisodeResult):
    error: str
    status: Literal["error"] = field(init=False, default=RESULT_STATUS_ERROR)

    def to_dict(self) -> dict[str, object]:
        payload = EpisodeResult.to_dict(self)
        payload.update({"cached": False, "error": self.error})
        return payload


EpisodeTranscriptionResult: TypeAlias = (
    SuccessfulEpisodeTranscriptionResult | FailedEpisodeTranscriptionResult
)


@dataclass(slots=True, frozen=True)
class BatchAudioTranscriptionResult:
    episode_results: list[EpisodeTranscriptionResult]
    success_count: int
    fail_count: int
