from __future__ import annotations

import os
from typing import Final

from podcast_job_finder.transcription.backends.doubao.runtime import (
    DOUBAO_BACKEND,
    DOUBAO_MAX_IN_FLIGHT_REQUESTS_ENV as DOUBAO_MAX_IN_FLIGHT_REQUESTS_ENV,
    DOUBAO_MAX_INPUT_AUDIO_DURATION_MS as DOUBAO_MAX_INPUT_AUDIO_DURATION_MS,
    DOUBAO_REQUEST_INTERVAL_ENV as DOUBAO_REQUEST_INTERVAL_ENV,
    load_doubao_transcription_runtime,
)
from podcast_job_finder.transcription.backends.firered.runtime import (
    FIRERED_BACKEND,
    load_firered_transcription_runtime,
)
from podcast_job_finder.transcription.backends.openai_compatible import (
    LLM_BACKEND,
    load_llm_transcription_runtime,
)
from podcast_job_finder.audio.segmentation.speech_pipeline import (
    DEFAULT_SILENCE_PADDING_MS,
)
from podcast_job_finder.transcription.models import AudioTranscriptionRuntime
from podcast_job_finder.transcription.runtime_environment import (
    AudioTranscriptionConfigError,
    FIRERED_ASR_MODEL_DIR_ENV as FIRERED_ASR_MODEL_DIR_ENV,
    FIRERED_ASR_MODEL_RELATIVE_PATH as FIRERED_ASR_MODEL_RELATIVE_PATH,
    FIRERED_ORT_INTRA_OP_THREADS_ENV as FIRERED_ORT_INTRA_OP_THREADS_ENV,
    FIRERED_ORT_PROVIDER_ENV as FIRERED_ORT_PROVIDER_ENV,
    FIRERED_PUNC_MODEL_DIR_ENV as FIRERED_PUNC_MODEL_DIR_ENV,
    FIRERED_PUNC_MODEL_RELATIVE_PATH as FIRERED_PUNC_MODEL_RELATIVE_PATH,
    FIRERED_PYTHON_ENV as FIRERED_PYTHON_ENV,
    FIRERED_PYTHON_NOT_FOUND_ERROR as FIRERED_PYTHON_NOT_FOUND_ERROR,
    FIRERED_PYTHON_RELATIVE_PATH as FIRERED_PYTHON_RELATIVE_PATH,
    INVALID_FLOAT_ENV_ERROR as INVALID_FLOAT_ENV_ERROR,
    INVALID_INTEGER_ENV_ERROR as INVALID_INTEGER_ENV_ERROR,
    SOURCE_PROJECT_ROOT as SOURCE_PROJECT_ROOT,
)


__all__ = (
    "AUDIO_TRANSCRIPTION_BACKEND_ENV",
    "AudioTranscriptionConfigError",
    "AudioTranscriptionRuntime",
    "DOUBAO_BACKEND",
    "DOUBAO_MAX_IN_FLIGHT_REQUESTS_ENV",
    "DOUBAO_MAX_INPUT_AUDIO_DURATION_MS",
    "DOUBAO_REQUEST_INTERVAL_ENV",
    "FIRERED_ASR_MODEL_DIR_ENV",
    "FIRERED_ASR_MODEL_RELATIVE_PATH",
    "FIRERED_BACKEND",
    "FIRERED_ORT_INTRA_OP_THREADS_ENV",
    "FIRERED_ORT_PROVIDER_ENV",
    "FIRERED_PUNC_MODEL_DIR_ENV",
    "FIRERED_PUNC_MODEL_RELATIVE_PATH",
    "FIRERED_PYTHON_ENV",
    "FIRERED_PYTHON_NOT_FOUND_ERROR",
    "FIRERED_PYTHON_RELATIVE_PATH",
    "INVALID_BACKEND_ERROR",
    "INVALID_FLOAT_ENV_ERROR",
    "INVALID_INTEGER_ENV_ERROR",
    "LLM_BACKEND",
    "SOURCE_PROJECT_ROOT",
    "load_audio_transcription_runtime_from_env",
)


AUDIO_TRANSCRIPTION_BACKEND_ENV: Final = "AUDIO_TRANSCRIPTION_BACKEND"
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
        return load_llm_transcription_runtime(silence_padding_ms=silence_padding_ms)
    if backend == FIRERED_BACKEND:
        return load_firered_transcription_runtime(silence_padding_ms=silence_padding_ms)
    if backend == DOUBAO_BACKEND:
        return load_doubao_transcription_runtime(silence_padding_ms=silence_padding_ms)
    raise AudioTranscriptionConfigError(INVALID_BACKEND_ERROR.format(backend=backend))
