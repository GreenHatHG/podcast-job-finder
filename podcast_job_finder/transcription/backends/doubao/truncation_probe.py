from __future__ import annotations

import asyncio
import logging
import tempfile
import wave
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Coroutine, Final, Iterator, Protocol

import numpy as np
from numpy.typing import NDArray

from podcast_job_finder.audio.segmentation._pcm import (
    PCM_CHANNELS,
    PCM_SAMPLE_WIDTH_BYTES,
)
from podcast_job_finder.audio.segmentation.normalized_audio import normalize_audio_file
from podcast_job_finder.transcription.diagnostics import (
    TruncationAssessment,
    TruncationProbeDiagnostics,
)
from podcast_job_finder.audio.segmentation.vad import VAD_SAMPLE_RATE
from podcast_job_finder.filesystem import OWNER_READ_WRITE_MODE, temporary_sibling_path

from .request_client import DoubaoRequestError
from .response import (
    AsrResponseProtocol,
    DOUBAO_RESPONSE_ERROR,
    build_doubao_response_summary,
)


TRUNCATION_PROBE_CONTEXT_MS: Final = 500
TRUNCATION_PROBE_FILE_NAME: Final = "doubao-truncation-probe.wav"
logger = logging.getLogger(__name__)


class ProbeResponseTypes(Protocol):
    FINAL_RESULT: object
    SESSION_FINISHED: object
    ERROR: object


ProbeResponse = list[AsrResponseProtocol] | DoubaoRequestError
CollectResponses = Callable[
    [Path],
    Coroutine[Any, Any, list[AsrResponseProtocol]],
]


@dataclass(slots=True, frozen=True)
class TruncationProbeAudio:
    path: Path
    start_ms: int
    end_ms: int


@dataclass(slots=True, frozen=True)
class TruncationProbeRequest:
    audio_path: Path
    assessment: TruncationAssessment


def run_doubao_truncation_probe(
    request: TruncationProbeRequest,
    *,
    collect_responses: CollectResponses,
    response_types: ProbeResponseTypes,
) -> TruncationProbeDiagnostics:
    with build_truncation_probe_audio(
        request.audio_path,
        request.assessment,
    ) as probe:
        try:
            responses: ProbeResponse = asyncio.run(collect_responses(probe.path))
        except DoubaoRequestError as error:
            responses = error
        return _build_probe_diagnostics(probe, responses, response_types=response_types)


def _build_probe_diagnostics(
    probe: TruncationProbeAudio,
    responses: ProbeResponse,
    *,
    response_types: ProbeResponseTypes,
) -> TruncationProbeDiagnostics:
    if isinstance(responses, DoubaoRequestError):
        return TruncationProbeDiagnostics(
            start_ms=probe.start_ms,
            end_ms=probe.end_ms,
            text="",
            confirmed_speech=False,
            error=str(responses),
        )
    response_summary = build_doubao_response_summary(
        responses,
        final_response_type=response_types.FINAL_RESULT,
        error_response_type=response_types.ERROR,
        terminal_response_type=response_types.SESSION_FINISHED,
    )
    error = None
    if not response_summary.is_complete:
        error = DOUBAO_RESPONSE_ERROR.format(path=probe.path)
    return TruncationProbeDiagnostics(
        start_ms=probe.start_ms,
        end_ms=probe.end_ms,
        text=response_summary.text,
        confirmed_speech=contains_transcribable_text(response_summary.text),
        error=error,
    )


def log_truncation_probe_result(
    *,
    segment_index: int,
    assessment: TruncationAssessment,
    probe: TruncationProbeDiagnostics,
) -> None:
    arguments = (
        segment_index,
        assessment.speech_coverage_ratio,
        assessment.longest_uncovered_speech_ms,
        probe.start_ms,
        probe.end_ms,
    )
    if probe.error:
        logger.warning(
            "豆包检查“识别结果里是否有一段人声没有对应文字”时失败，执行完整重试："
            "index=%d coverage=%.3f gap_ms=%d probe_start_ms=%d "
            "probe_end_ms=%d error=%s",
            *arguments,
            probe.error,
        )
        return
    if probe.confirmed_speech:
        logger.warning(
            "豆包确认“识别结果里有一段人声没有对应文字”，执行完整重试："
            "index=%d coverage=%.3f gap_ms=%d probe_start_ms=%d probe_end_ms=%d",
            *arguments,
        )
        return
    logger.info(
        "豆包检查时未识别出有效文字，跳过完整重试："
        "index=%d coverage=%.3f gap_ms=%d probe_start_ms=%d probe_end_ms=%d",
        *arguments,
    )


def probe_requires_full_retry(probe: TruncationProbeDiagnostics) -> bool:
    return probe.confirmed_speech or probe.error is not None


@contextmanager
def build_truncation_probe_audio(
    audio_path: Path,
    assessment: TruncationAssessment,
) -> Iterator[TruncationProbeAudio]:
    with normalize_audio_file(audio_path, sample_rate=VAD_SAMPLE_RATE) as audio:
        requested_start_ms = max(
            0,
            assessment.longest_uncovered_speech_start_ms - TRUNCATION_PROBE_CONTEXT_MS,
        )
        requested_end_ms = (
            assessment.longest_uncovered_speech_end_ms + TRUNCATION_PROBE_CONTEXT_MS
        )
        start_sample = _milliseconds_to_samples(requested_start_ms)
        end_sample = min(
            audio.sample_count,
            _milliseconds_to_samples(requested_end_ms),
        )
        samples = audio.read_samples(start_sample, end_sample).copy()

    temporary_target = Path(tempfile.gettempdir()) / TRUNCATION_PROBE_FILE_NAME
    with temporary_sibling_path(
        temporary_target,
        mode=OWNER_READ_WRITE_MODE,
    ) as probe_path:
        _write_probe_audio(probe_path, samples)
        yield TruncationProbeAudio(
            path=probe_path,
            start_ms=_samples_to_milliseconds(start_sample),
            end_ms=_samples_to_milliseconds(end_sample),
        )


def contains_transcribable_text(text: str) -> bool:
    return any(character.isalnum() for character in text)


def _write_probe_audio(path: Path, samples: NDArray[np.int16]) -> None:
    with wave.Wave_write(str(path)) as wav_file:
        wav_file.setnchannels(PCM_CHANNELS)
        wav_file.setsampwidth(PCM_SAMPLE_WIDTH_BYTES)
        wav_file.setframerate(VAD_SAMPLE_RATE)
        wav_file.writeframes(samples.tobytes())


def _milliseconds_to_samples(value: int) -> int:
    return round(value * VAD_SAMPLE_RATE / 1_000)


def _samples_to_milliseconds(value: int) -> int:
    return round(value * 1_000 / VAD_SAMPLE_RATE)
