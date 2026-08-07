from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path

from podcast_job_finder.audio.segment_export import ExportedSpeechSegment

from .request_scheduler import DoubaoRequestScheduler
from .response import AsrResponseProtocol


logger = logging.getLogger(__name__)


SessionResponses = list[AsrResponseProtocol]
TranscribeStream = Callable[..., AsyncIterator[AsrResponseProtocol]]


@dataclass(slots=True, frozen=True)
class _DoubaoRequest:
    path: Path
    config: object
    segment: ExportedSpeechSegment | None = None


class DoubaoRequestClient:
    def __init__(
        self,
        *,
        asr_config: object,
        transcribe_stream: TranscribeStream,
        max_in_flight_requests: int,
        request_interval_seconds: float,
    ) -> None:
        self._asr_config = asr_config
        self._transcribe_stream = transcribe_stream
        self._request_scheduler = DoubaoRequestScheduler[
            _DoubaoRequest,
            SessionResponses,
        ](
            max_in_flight_requests=max_in_flight_requests,
            interval_seconds=request_interval_seconds,
            worker=self._run_scheduled_request,
        )
        logger.info(
            "豆包请求调度配置：request_interval_seconds=%.3f max_in_flight_requests=%d",
            request_interval_seconds,
            max_in_flight_requests,
        )

    def submit_segment(
        self,
        segment: ExportedSpeechSegment,
    ) -> Future[SessionResponses]:
        return self._request_scheduler.submit(
            _DoubaoRequest(
                path=segment.file_path,
                config=self._asr_config,
                segment=segment,
            )
        )

    def collect_responses(self, path: Path) -> SessionResponses:
        return asyncio.run(self._collect_responses(path))

    async def collect_probe_responses(self, path: Path) -> SessionResponses:
        return await self._collect_responses_with_config(
            path,
            self._asr_config,
        )

    def close(self) -> None:
        self._request_scheduler.close()

    def _run_scheduled_request(
        self,
        request: _DoubaoRequest,
    ) -> SessionResponses:
        if request.segment is not None:
            logger.info(
                "识别音频片段：index=%d start_ms=%d end_ms=%d",
                request.segment.index,
                request.segment.segment.start_ms,
                request.segment.segment.end_ms,
            )
        return asyncio.run(
            self._collect_stream_responses(
                request.path,
                request.config,
            )
        )

    async def _collect_responses(
        self,
        path: Path,
    ) -> SessionResponses:
        return await self._collect_responses_with_config(
            path,
            self._asr_config,
        )

    async def _collect_responses_with_config(
        self,
        path: Path,
        config: object,
    ) -> SessionResponses:
        future = await asyncio.to_thread(
            self._request_scheduler.submit,
            _DoubaoRequest(
                path=path,
                config=config,
            ),
        )
        return await asyncio.wrap_future(future)

    async def _collect_stream_responses(
        self,
        path: Path,
        config: object,
    ) -> SessionResponses:
        responses = []
        async for response in self._transcribe_stream(
            path,
            config=config,
            realtime=True,
        ):
            responses.append(response)
        return responses
