from __future__ import annotations

import os
from typing import Final

from podcast_job_finder.transcription.backends.doubao.config import (
    DEFAULT_DOUBAO_MAX_IN_FLIGHT_REQUESTS,
    DEFAULT_DOUBAO_REQUEST_INTERVAL_SECONDS,
    DoubaoTranscriberConfig,
)
from podcast_job_finder.transcription.backends.doubao.transcriber import (
    DoubaoAudioTranscriber,
)
from podcast_job_finder.transcription.backends.firered.alignment import (
    FireRedAlignmentConfig,
    FireRedTextAlignmentClient,
)
from podcast_job_finder.transcription.backends.firered.config import (
    DEFAULT_ORT_INTRA_OP_THREADS,
    DEFAULT_ORT_PROVIDER,
)
from podcast_job_finder.transcription.backends.firered.runtime import (
    load_firered_transcription_runtime,
)
from podcast_job_finder.transcription.backends.openai_compatible import (
    LlmAudioTranscriber,
)
from podcast_job_finder.audio.segmentation.segment_export import (
    MP3_SEGMENT_AUDIO_FORMAT,
    WAV_SEGMENT_AUDIO_FORMAT,
)
from podcast_job_finder.audio.segmentation.speech_pipeline import (
    DEFAULT_SILENCE_PADDING_MS,
)
from podcast_job_finder.transcription.models import AudioTranscriptionRuntime
from podcast_job_finder.audio.segmentation.vad import VadConfig
from podcast_job_finder.llm import (
    load_audio_transcription_llm_runtime_from_env,
)
from podcast_job_finder.transcription.runtime_environment import (
    AudioTranscriptionConfigError,
    FIRERED_ASR_MODEL_DIR_ENV,
    FIRERED_ASR_MODEL_RELATIVE_PATH,
    FIRERED_ORT_INTRA_OP_THREADS_ENV,
    FIRERED_ORT_PROVIDER_ENV,
    load_firered_python,
    load_float_env,
    load_integer_env,
    load_optional_path_env,
)


AUDIO_TRANSCRIPTION_BACKEND_ENV: Final = "AUDIO_TRANSCRIPTION_BACKEND"
LLM_BACKEND: Final = "llm"
FIRERED_BACKEND: Final = "firered"
DOUBAO_BACKEND: Final = "doubao"
DOUBAO_MAX_IN_FLIGHT_REQUESTS_ENV: Final = "DOUBAO_ASR_MAX_IN_FLIGHT_REQUESTS"
DOUBAO_REQUEST_INTERVAL_ENV: Final = "DOUBAO_ASR_REQUEST_INTERVAL_SECONDS"
DOUBAO_MAX_INPUT_AUDIO_DURATION_MS: Final = 20_000
INVALID_BACKEND_ERROR: Final = (
    "AUDIO_TRANSCRIPTION_BACKEND 必须是 llm、firered 或 doubao：{backend}"
)


def load_audio_transcription_runtime_from_env(
    *,
    silence_padding_ms: int = DEFAULT_SILENCE_PADDING_MS,
) -> AudioTranscriptionRuntime:
    backend = (
        os.environ.get(AUDIO_TRANSCRIPTION_BACKEND_ENV, LLM_BACKEND).strip().lower()
    )
    if backend == LLM_BACKEND:
        return _load_llm_runtime(silence_padding_ms=silence_padding_ms)
    if backend == FIRERED_BACKEND:
        return load_firered_transcription_runtime(silence_padding_ms=silence_padding_ms)
    if backend == DOUBAO_BACKEND:
        return _load_doubao_runtime(silence_padding_ms=silence_padding_ms)
    raise AudioTranscriptionConfigError(INVALID_BACKEND_ERROR.format(backend=backend))


def _load_llm_runtime(*, silence_padding_ms: int) -> AudioTranscriptionRuntime:
    runtime = load_audio_transcription_llm_runtime_from_env()
    metadata = {
        "transcription_backend": LLM_BACKEND,
        "model": runtime.model,
        "base_url": runtime.base_url,
        "api_style": runtime.api_style,
    }
    return AudioTranscriptionRuntime(
        transcriber=LlmAudioTranscriber(
            runtime.client,
            retry_config=runtime.retry_config,
        ),
        metadata=metadata,
        segment_audio_format=MP3_SEGMENT_AUDIO_FORMAT,
        silence_padding_ms=silence_padding_ms,
    )


def _load_doubao_runtime(*, silence_padding_ms: int) -> AudioTranscriptionRuntime:
    vad_config = _build_doubao_vad_config(silence_padding_ms)
    alignment_config = FireRedAlignmentConfig(
        python_executable=load_firered_python(),
        asr_model_dir=load_optional_path_env(
            FIRERED_ASR_MODEL_DIR_ENV,
            FIRERED_ASR_MODEL_RELATIVE_PATH,
        ),
        ort_provider=os.environ.get(
            FIRERED_ORT_PROVIDER_ENV,
            DEFAULT_ORT_PROVIDER,
        ),
        ort_intra_op_threads=load_integer_env(
            FIRERED_ORT_INTRA_OP_THREADS_ENV,
            DEFAULT_ORT_INTRA_OP_THREADS,
        ),
    )
    transcriber_config = DoubaoTranscriberConfig(
        max_in_flight_requests=load_integer_env(
            DOUBAO_MAX_IN_FLIGHT_REQUESTS_ENV,
            DEFAULT_DOUBAO_MAX_IN_FLIGHT_REQUESTS,
        ),
        request_interval_seconds=load_float_env(
            DOUBAO_REQUEST_INTERVAL_ENV,
            DEFAULT_DOUBAO_REQUEST_INTERVAL_SECONDS,
        ),
        silence_padding_ms=silence_padding_ms,
        vad_threshold=vad_config.threshold,
    )
    metadata = {
        **transcriber_config.metadata(),
        **alignment_config.metadata(),
        "max_input_audio_duration_ms": DOUBAO_MAX_INPUT_AUDIO_DURATION_MS,
        "min_speech_duration_ms": vad_config.min_speech_duration_ms,
        "max_speech_duration_ms": vad_config.max_speech_duration_ms,
        "forced_split_overlap_ms": vad_config.forced_split_overlap_ms,
        "min_silence_duration_ms": vad_config.min_silence_duration_ms,
    }
    return AudioTranscriptionRuntime(
        transcriber=DoubaoAudioTranscriber(
            transcriber_config,
            aligner=FireRedTextAlignmentClient(alignment_config),
        ),
        metadata=metadata,
        segment_audio_format=WAV_SEGMENT_AUDIO_FORMAT,
        vad_config=vad_config,
        silence_padding_ms=silence_padding_ms,
    )


def _build_doubao_vad_config(silence_padding_ms: int) -> VadConfig:
    default_config = VadConfig()
    max_speech_duration_ms = (
        DOUBAO_MAX_INPUT_AUDIO_DURATION_MS
        - 2 * silence_padding_ms
        - default_config.forced_split_overlap_ms
    )
    return VadConfig(
        min_speech_duration_ms=min(
            default_config.min_speech_duration_ms,
            max_speech_duration_ms,
        ),
        max_speech_duration_ms=max_speech_duration_ms,
    )
