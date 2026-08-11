from __future__ import annotations

import os
from typing import Final

from podcast_job_finder.audio.segmentation.segment_export import (
    WAV_SEGMENT_AUDIO_FORMAT,
)
from podcast_job_finder.audio.segmentation.speech_pipeline import (
    DEFAULT_SILENCE_PADDING_MS,
)
from podcast_job_finder.transcription.backends.firered.config import (
    DEFAULT_ORT_INTRA_OP_THREADS,
    DEFAULT_ORT_PROVIDER,
)
from podcast_job_finder.transcription.backends.firered.transcriber import (
    FireRedAudioTranscriber,
)
from podcast_job_finder.transcription.models import AudioTranscriptionRuntime
from podcast_job_finder.transcription.runtime_environment import (
    FIRERED_ASR_MODEL_DIR_ENV,
    FIRERED_ASR_MODEL_RELATIVE_PATH,
    FIRERED_ORT_INTRA_OP_THREADS_ENV,
    FIRERED_ORT_PROVIDER_ENV,
    FIRERED_PUNC_MODEL_DIR_ENV,
    FIRERED_PUNC_MODEL_RELATIVE_PATH,
    load_firered_python,
    load_integer_env,
    load_optional_path_env,
)


FIRERED_BACKEND: Final = "firered"


def load_firered_transcription_runtime(
    *,
    silence_padding_ms: int = DEFAULT_SILENCE_PADDING_MS,
) -> AudioTranscriptionRuntime:
    transcriber = FireRedAudioTranscriber(
        python_executable=load_firered_python(),
        asr_model_dir=load_optional_path_env(
            FIRERED_ASR_MODEL_DIR_ENV,
            FIRERED_ASR_MODEL_RELATIVE_PATH,
        ),
        punc_model_dir=load_optional_path_env(
            FIRERED_PUNC_MODEL_DIR_ENV,
            FIRERED_PUNC_MODEL_RELATIVE_PATH,
        ),
        ort_provider=os.environ.get(FIRERED_ORT_PROVIDER_ENV, DEFAULT_ORT_PROVIDER),
        ort_intra_op_threads=load_integer_env(
            FIRERED_ORT_INTRA_OP_THREADS_ENV,
            DEFAULT_ORT_INTRA_OP_THREADS,
        ),
        silence_padding_ms=silence_padding_ms,
    )
    return AudioTranscriptionRuntime(
        transcriber=transcriber,
        metadata=transcriber.metadata(),
        segment_audio_format=WAV_SEGMENT_AUDIO_FORMAT,
        silence_padding_ms=silence_padding_ms,
    )
