from __future__ import annotations

from pathlib import Path
from typing import Final

from podcast_job_finder.audio.segmentation.segment_export import (
    ExportedSpeechSegment,
    to_source_audio_timestamp,
)
from podcast_job_finder.errors import ConfigurationError
from podcast_job_finder.audio.segmentation.speech_pipeline import (
    DEFAULT_SILENCE_PADDING_MS,
)
from podcast_job_finder.transcription.models import (
    AudioTranscriptionError,
    TimedTranscriptionText,
    TranscribedSpeechSegment,
    TranscriptionOutput,
)

from ._worker_process import (
    WORKER_RESULT_STATUS,
    JsonLineWorkerProcess,
    WorkerExitedError,
    WorkerResponseError,
    WorkerStartError,
)


FIRERED_MODEL_NAME: Final = "FireRedASR2-AED-ONNX+FireRedPunc"
REQUIRED_ASR_FILES: Final = (
    "encoder.int8.onnx",
    "decoder.int8.onnx",
    "ctc.int8.onnx",
    "cmvn.ark",
    "tokens.txt",
)
REQUIRED_PUNC_FILES: Final = (
    "model.pth.tar",
    "chinese-bert-wwm-ext_vocab.txt",
    "out_dict",
    "chinese-lert-base/config.json",
    "chinese-lert-base/pytorch_model.bin",
    "chinese-lert-base/vocab.txt",
)
WORKER_START_ERROR: Final = "FireRed 工作进程启动失败：{message}"
WORKER_EXIT_ERROR: Final = "FireRed 工作进程意外退出：returncode={returncode}"
WORKER_RESPONSE_ERROR: Final = "FireRed 工作进程返回无效结果：{message}"


class FireRedConfigError(ConfigurationError, ValueError):
    """FireRed 本地转写配置无效。"""


class FireRedAudioTranscriber:
    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        python_executable: Path,
        asr_model_dir: Path,
        punc_model_dir: Path,
        ort_provider: str,
        ort_intra_op_threads: int,
        silence_padding_ms: int = DEFAULT_SILENCE_PADDING_MS,
    ) -> None:
        _require_file(python_executable, "FIRERED_PYTHON")
        _require_model_files(asr_model_dir, REQUIRED_ASR_FILES)
        _require_model_files(punc_model_dir, REQUIRED_PUNC_FILES)
        if ort_intra_op_threads <= 0:
            raise FireRedConfigError("FIRERED_ORT_INTRA_OP_THREADS 必须大于 0。")
        if silence_padding_ms < 0:
            raise FireRedConfigError("silence_padding_ms 必须大于等于 0。")
        self._asr_model_dir = asr_model_dir
        self._punc_model_dir = punc_model_dir
        self._ort_provider = ort_provider
        self._silence_padding_ms = silence_padding_ms
        worker_path = Path(__file__).with_name("worker") / "worker.py"
        self._worker_process = JsonLineWorkerProcess(
            command=(
                str(python_executable),
                str(worker_path),
                "--asr-model-dir",
                str(asr_model_dir),
                "--punc-model-dir",
                str(punc_model_dir),
                "--ort-provider",
                ort_provider,
                "--ort-intra-op-threads",
                str(ort_intra_op_threads),
            ),
            stdin_unavailable_error="FireRed 工作进程标准输入不可用。",
        )

    def metadata(self) -> dict[str, object]:
        return {
            "transcription_backend": "firered",
            "model": FIRERED_MODEL_NAME,
            "provider": self._ort_provider,
            "asr_model_dir": str(self._asr_model_dir.resolve()),
            "punc_model_dir": str(self._punc_model_dir.resolve()),
        }

    def transcribe(
        self,
        segment: ExportedSpeechSegment,
        *,
        previous_segment: TranscribedSpeechSegment | None,
    ) -> TranscriptionOutput:
        self._ensure_process()
        response = self._request(
            {
                "audio_path": str(segment.file_path.resolve()),
                "discard_before_ms": self._build_discard_before_ms(
                    segment,
                    previous_segment=previous_segment,
                ),
            },
        )
        if response.get("status") != WORKER_RESULT_STATUS:
            raise AudioTranscriptionError(
                WORKER_RESPONSE_ERROR.format(message=response.get("error", response))
            )
        text = response.get("text")
        if not isinstance(text, str) or not text.strip():
            raise AudioTranscriptionError(
                WORKER_RESPONSE_ERROR.format(message="text 为空")
            )
        return TranscriptionOutput(
            text=text.strip(),
            character_timestamps=self._parse_timestamps(
                response.get("character_timestamps"),
                segment=segment,
            ),
            sentences=self._parse_timestamps(
                response.get("sentences"),
                segment=segment,
            ),
        )

    def _build_discard_before_ms(
        self,
        segment: ExportedSpeechSegment,
        *,
        previous_segment: TranscribedSpeechSegment | None,
    ) -> int:
        # 相邻音频片段有重叠；从上一段结束位置继续，避免重复识别。
        if previous_segment is None or not previous_segment.character_timestamps:
            return 0
        previous_end_ms = previous_segment.character_timestamps[-1].end_ms
        if previous_end_ms <= segment.segment.start_ms:
            return 0
        return previous_end_ms - segment.segment.start_ms + self._silence_padding_ms

    def close(self) -> None:
        self._worker_process.close()

    def _ensure_process(self) -> None:
        try:
            self._worker_process.ensure_ready()
        except OSError as error:
            raise AudioTranscriptionError(
                WORKER_START_ERROR.format(message=str(error))
            ) from error
        except WorkerStartError as error:
            raise AudioTranscriptionError(
                WORKER_START_ERROR.format(message=error.detail)
            ) from error
        except WorkerExitedError as error:
            raise AudioTranscriptionError(
                WORKER_EXIT_ERROR.format(returncode=error.returncode)
            ) from error
        except WorkerResponseError as error:
            raise AudioTranscriptionError(
                WORKER_RESPONSE_ERROR.format(message=error.response)
            ) from error

    def _request(
        self,
        payload: dict[str, object],
    ) -> dict[str, object]:
        try:
            return self._worker_process.request(payload)
        except (BrokenPipeError, OSError) as error:
            raise AudioTranscriptionError(
                WORKER_EXIT_ERROR.format(returncode=self._worker_process.returncode)
            ) from error
        except WorkerExitedError as error:
            raise AudioTranscriptionError(
                WORKER_EXIT_ERROR.format(returncode=error.returncode)
            ) from error
        except WorkerResponseError as error:
            raise AudioTranscriptionError(
                WORKER_RESPONSE_ERROR.format(message=error.response)
            ) from error

    def _parse_timestamps(
        self,
        value: object,
        *,
        segment: ExportedSpeechSegment,
    ) -> tuple[TimedTranscriptionText, ...]:
        if not isinstance(value, list):
            return ()
        timestamps = []
        for item in value:
            timestamp = self._parse_timestamp(item, segment=segment)
            if timestamp is not None:
                timestamps.append(timestamp)
        return tuple(timestamps)

    def _parse_timestamp(
        self,
        value: object,
        *,
        segment: ExportedSpeechSegment,
    ) -> TimedTranscriptionText | None:
        if not isinstance(value, dict):
            return None
        text = value.get("text")
        start_ms = value.get("start_ms")
        end_ms = value.get("end_ms")
        if (
            not isinstance(text, str)
            or not isinstance(start_ms, int)
            or not isinstance(end_ms, int)
        ):
            return None
        absolute_start = to_source_audio_timestamp(
            start_ms,
            segment=segment,
            silence_padding_ms=self._silence_padding_ms,
        )
        absolute_end = to_source_audio_timestamp(
            end_ms,
            segment=segment,
            silence_padding_ms=self._silence_padding_ms,
        )
        if absolute_end <= absolute_start:
            return None
        return TimedTranscriptionText(
            text=text,
            start_ms=absolute_start,
            end_ms=absolute_end,
        )


def _require_file(path: Path, field_name: str) -> None:
    if not path.is_file():
        raise FireRedConfigError(f"{field_name} 指向的文件不存在：{path}")


def _require_model_files(model_dir: Path, relative_paths: tuple[str, ...]) -> None:
    missing_paths = [
        relative_path
        for relative_path in relative_paths
        if not (model_dir / relative_path).is_file()
    ]
    if missing_paths:
        raise FireRedConfigError(
            f"FireRed 模型目录缺少文件：{model_dir}，{', '.join(missing_paths)}"
        )
