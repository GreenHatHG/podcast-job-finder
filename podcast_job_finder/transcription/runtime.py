from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping

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
)
from podcast_job_finder.transcription.backends.firered.config import (
    DEFAULT_ORT_INTRA_OP_THREADS,
    DEFAULT_ORT_PROVIDER,
)
from podcast_job_finder.transcription.backends.firered.transcriber import (
    FireRedAudioTranscriber,
    FireRedTranscriberConfig,
)
from podcast_job_finder.transcription.backends.openai_compatible import (
    LlmAudioTranscriber,
)
from podcast_job_finder.audio.segmentation.segment_export import (
    MP3_SEGMENT_AUDIO_FORMAT,
    WAV_SEGMENT_AUDIO_FORMAT,
    SegmentAudioFormat,
)
from podcast_job_finder.audio.segmentation.speech_pipeline import (
    DEFAULT_SILENCE_PADDING_MS,
)
from podcast_job_finder.transcription.models import AudioTranscriberProtocol
from podcast_job_finder.audio.segmentation.vad import VadConfig
from podcast_job_finder.llm import (
    load_audio_transcription_llm_runtime_config_from_env,
)


AUDIO_TRANSCRIPTION_BACKEND_ENV: Final = "AUDIO_TRANSCRIPTION_BACKEND"
FIRERED_PYTHON_ENV: Final = "FIRERED_PYTHON"
FIRERED_ASR_MODEL_DIR_ENV: Final = "FIRERED_ASR_MODEL_DIR"
FIRERED_PUNC_MODEL_DIR_ENV: Final = "FIRERED_PUNC_MODEL_DIR"
FIRERED_ORT_PROVIDER_ENV: Final = "FIRERED_ORT_PROVIDER"
FIRERED_ORT_INTRA_OP_THREADS_ENV: Final = "FIRERED_ORT_INTRA_OP_THREADS"
LLM_BACKEND: Final = "llm"
FIRERED_BACKEND: Final = "firered"
DOUBAO_BACKEND: Final = "doubao"
DOUBAO_MAX_IN_FLIGHT_REQUESTS_ENV: Final = "DOUBAO_ASR_MAX_IN_FLIGHT_REQUESTS"
DOUBAO_REQUEST_INTERVAL_ENV: Final = "DOUBAO_ASR_REQUEST_INTERVAL_SECONDS"
DOUBAO_MAX_INPUT_AUDIO_DURATION_MS: Final = 20_000
INVALID_BACKEND_ERROR: Final = (
    "AUDIO_TRANSCRIPTION_BACKEND 必须是 llm、firered 或 doubao：{backend}"
)
INVALID_INTEGER_ENV_ERROR: Final = "{name} 必须是整数：{value}"
INVALID_FLOAT_ENV_ERROR: Final = "{name} 必须是有限数值：{value}"
FIRERED_PYTHON_NOT_FOUND_ERROR: Final = (
    "未找到 FireRed Python 解释器，已检查：{candidates}。"
    "请创建对应的虚拟环境或设置 {environment_variable}。"
)
SOURCE_PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
FIRERED_PYTHON_RELATIVE_PATH: Final = Path("firered_worker/.venv/bin/python")
FIRERED_ASR_MODEL_RELATIVE_PATH: Final = Path(
    "models/firered/FireRedASR2-AED-ONNX-modelscope"
)
FIRERED_PUNC_MODEL_RELATIVE_PATH: Final = Path("models/firered/FireRedPunc")


class AudioTranscriptionConfigError(ValueError):
    """音频转写后端配置无效。"""


@dataclass(slots=True)
class AudioTranscriptionRuntime:
    transcriber: AudioTranscriberProtocol
    metadata: Mapping[str, object]
    segment_audio_format: SegmentAudioFormat
    vad_config: VadConfig = VadConfig()

    def close(self) -> None:
        self.transcriber.close()


def load_audio_transcription_runtime_from_env(
    *,
    silence_padding_ms: int = DEFAULT_SILENCE_PADDING_MS,
) -> AudioTranscriptionRuntime:
    backend = (
        os.environ.get(AUDIO_TRANSCRIPTION_BACKEND_ENV, LLM_BACKEND).strip().lower()
    )
    if backend == LLM_BACKEND:
        return _load_llm_runtime()
    if backend == FIRERED_BACKEND:
        return _load_firered_runtime(silence_padding_ms=silence_padding_ms)
    if backend == DOUBAO_BACKEND:
        return _load_doubao_runtime(silence_padding_ms=silence_padding_ms)
    raise AudioTranscriptionConfigError(INVALID_BACKEND_ERROR.format(backend=backend))


def _load_llm_runtime() -> AudioTranscriptionRuntime:
    runtime = load_audio_transcription_llm_runtime_config_from_env()
    config = runtime.client_config
    metadata = {
        "transcription_backend": LLM_BACKEND,
        "model": config.model,
        "base_url": config.base_url,
        "api_style": config.api_style,
    }
    return AudioTranscriptionRuntime(
        transcriber=LlmAudioTranscriber(
            runtime.build_client(),
            retry_config=runtime.retry_config,
        ),
        metadata=metadata,
        segment_audio_format=MP3_SEGMENT_AUDIO_FORMAT,
    )


def _load_firered_runtime(*, silence_padding_ms: int) -> AudioTranscriptionRuntime:
    config = FireRedTranscriberConfig(
        python_executable=_load_firered_python(),
        asr_model_dir=_load_optional_path_env(
            FIRERED_ASR_MODEL_DIR_ENV,
            FIRERED_ASR_MODEL_RELATIVE_PATH,
        ),
        punc_model_dir=_load_optional_path_env(
            FIRERED_PUNC_MODEL_DIR_ENV,
            FIRERED_PUNC_MODEL_RELATIVE_PATH,
        ),
        ort_provider=os.environ.get(FIRERED_ORT_PROVIDER_ENV, DEFAULT_ORT_PROVIDER),
        ort_intra_op_threads=_load_integer_env(
            FIRERED_ORT_INTRA_OP_THREADS_ENV,
            DEFAULT_ORT_INTRA_OP_THREADS,
        ),
        silence_padding_ms=silence_padding_ms,
    )
    return AudioTranscriptionRuntime(
        transcriber=FireRedAudioTranscriber(config),
        metadata=config.metadata(),
        segment_audio_format=WAV_SEGMENT_AUDIO_FORMAT,
    )


def _load_doubao_runtime(*, silence_padding_ms: int) -> AudioTranscriptionRuntime:
    vad_config = _build_doubao_vad_config(silence_padding_ms)
    config = DoubaoTranscriberConfig(
        alignment_config=FireRedAlignmentConfig(
            python_executable=_load_firered_python(),
            asr_model_dir=_load_optional_path_env(
                FIRERED_ASR_MODEL_DIR_ENV,
                FIRERED_ASR_MODEL_RELATIVE_PATH,
            ),
            ort_provider=os.environ.get(
                FIRERED_ORT_PROVIDER_ENV,
                DEFAULT_ORT_PROVIDER,
            ),
            ort_intra_op_threads=_load_integer_env(
                FIRERED_ORT_INTRA_OP_THREADS_ENV,
                DEFAULT_ORT_INTRA_OP_THREADS,
            ),
        ),
        max_in_flight_requests=_load_integer_env(
            DOUBAO_MAX_IN_FLIGHT_REQUESTS_ENV,
            DEFAULT_DOUBAO_MAX_IN_FLIGHT_REQUESTS,
        ),
        request_interval_seconds=_load_float_env(
            DOUBAO_REQUEST_INTERVAL_ENV,
            DEFAULT_DOUBAO_REQUEST_INTERVAL_SECONDS,
        ),
        silence_padding_ms=silence_padding_ms,
        vad_threshold=vad_config.threshold,
    )
    metadata = {
        **config.metadata(),
        "max_input_audio_duration_ms": DOUBAO_MAX_INPUT_AUDIO_DURATION_MS,
        "min_speech_duration_ms": vad_config.min_speech_duration_ms,
        "max_speech_duration_ms": vad_config.max_speech_duration_ms,
        "forced_split_overlap_ms": vad_config.forced_split_overlap_ms,
        "min_silence_duration_ms": vad_config.min_silence_duration_ms,
    }
    return AudioTranscriptionRuntime(
        transcriber=DoubaoAudioTranscriber(config),
        metadata=metadata,
        segment_audio_format=WAV_SEGMENT_AUDIO_FORMAT,
        vad_config=vad_config,
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


def _load_firered_python() -> Path:
    configured_path = os.environ.get(FIRERED_PYTHON_ENV, "").strip()
    if configured_path:
        return Path(configured_path)
    return _default_firered_python()


def _default_firered_python() -> Path:
    candidates = _candidate_project_paths(FIRERED_PYTHON_RELATIVE_PATH)
    for python_executable in candidates:
        if python_executable.is_file():
            return python_executable
    raise AudioTranscriptionConfigError(
        FIRERED_PYTHON_NOT_FOUND_ERROR.format(
            candidates=", ".join(str(path) for path in candidates),
            environment_variable=FIRERED_PYTHON_ENV,
        )
    )


def _load_optional_path_env(name: str, default_relative_path: Path) -> Path:
    value = os.environ.get(name, "").strip()
    if value:
        return Path(value)
    candidates = _candidate_project_paths(default_relative_path)
    return next((path for path in candidates if path.exists()), candidates[-1])


def _candidate_project_paths(relative_path: Path) -> tuple[Path, ...]:
    working_directory_path = Path.cwd() / relative_path
    source_path = SOURCE_PROJECT_ROOT / relative_path
    if source_path == working_directory_path:
        return (source_path,)
    return working_directory_path, source_path


def _load_integer_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as error:
        raise AudioTranscriptionConfigError(
            INVALID_INTEGER_ENV_ERROR.format(name=name, value=value)
        ) from error


def _load_float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed_value = float(value)
    except ValueError as error:
        raise AudioTranscriptionConfigError(
            INVALID_FLOAT_ENV_ERROR.format(name=name, value=value)
        ) from error
    if not math.isfinite(parsed_value):
        raise AudioTranscriptionConfigError(
            INVALID_FLOAT_ENV_ERROR.format(name=name, value=value)
        )
    return parsed_value
