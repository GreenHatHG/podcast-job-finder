from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from podcast_job_finder.audio.segmentation.segment_export import ExportedSpeechSegment
from podcast_job_finder.timestamps import format_duration_ms
from podcast_job_finder.transcription.models import AudioTranscriptionError

from .response import AsrResponseProtocol


if TYPE_CHECKING:
    from doubaoime_asr import ASRConfig  # type: ignore[import-untyped]


logger = logging.getLogger(__name__)
DOUBAO_REQUEST_ERROR = "豆包 ASR 请求失败：path={path} error={error}"


class DoubaoRequestError(AudioTranscriptionError):
    """豆包请求在底层客户端完成重试后仍然失败。"""


SessionResponses = list[AsrResponseProtocol]
TranscribeStream = Callable[..., AsyncIterator[AsrResponseProtocol]]


@dataclass(slots=True, frozen=True)
class DoubaoJob:
    path: Path
    segment: ExportedSpeechSegment | None = None
    total_segment_count: int | None = None


def run_doubao_job(
    job: DoubaoJob,
    *,
    transcribe_stream: TranscribeStream,
    asr_config: ASRConfig,
) -> SessionResponses:
    if job.segment is not None and job.total_segment_count is not None:
        segment = job.segment.segment
        logger.info(
            "识别音频片段：progress=%d/%d start_ms=%d end_ms=%d "
            "start=%s end=%s duration=%s",
            job.segment.index,
            job.total_segment_count,
            segment.start_ms,
            segment.end_ms,
            format_duration_ms(segment.start_ms),
            format_duration_ms(segment.end_ms),
            format_duration_ms(segment.duration_ms),
        )
    return asyncio.run(
        _read_transcribe_stream(
            job.path,
            asr_config,
            transcribe_stream,
        )
    )


async def _read_transcribe_stream(
    path: Path,
    config: ASRConfig,
    transcribe_stream: TranscribeStream,
) -> SessionResponses:
    responses = []
    try:
        async for response in transcribe_stream(
            path,
            config=config,
            realtime=True,
        ):
            responses.append(response)
    except Exception as error:  # pylint: disable=broad-exception-caught
        raise DoubaoRequestError(
            DOUBAO_REQUEST_ERROR.format(path=path, error=error)
        ) from error
    return responses
