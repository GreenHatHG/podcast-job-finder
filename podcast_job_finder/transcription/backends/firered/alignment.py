from __future__ import annotations

from pathlib import Path
from typing import Final

from podcast_job_finder.transcription.models import (
    AudioTranscriptionError,
    CharacterAlignment,
)

from ._worker_process import (
    WORKER_READY_STATUS,
    WORKER_RESULT_STATUS,
    JsonLineWorkerProcess,
    WorkerExitedError,
    WorkerResponseError,
)


REQUIRED_ALIGNMENT_MODEL_FILES: Final = (
    "encoder.int8.onnx",
    "ctc.int8.onnx",
    "cmvn.ark",
    "tokens.txt",
)
WORKER_ERROR: Final = "FireRed CTC 工作进程返回无效结果：{message}"


class FireRedTextAlignmentClient:
    def __init__(
        self,
        *,
        python_executable: Path,
        asr_model_dir: Path,
        ort_provider: str,
        ort_intra_op_threads: int,
    ) -> None:
        _require_file(python_executable, "FIRERED_PYTHON")
        _require_model_files(asr_model_dir)
        if ort_intra_op_threads <= 0:
            raise ValueError("FIRERED_ORT_INTRA_OP_THREADS 必须大于 0。")
        self._asr_model_dir = asr_model_dir
        self._ort_provider = ort_provider
        worker_path = Path(__file__).with_name("worker") / "alignment_worker.py"
        self._worker_process = JsonLineWorkerProcess(
            command=(
                str(python_executable),
                str(worker_path),
                "--asr-model-dir",
                str(asr_model_dir),
                "--ort-provider",
                ort_provider,
                "--ort-intra-op-threads",
                str(ort_intra_op_threads),
            ),
            stdin_unavailable_error="FireRed CTC 工作进程标准输入不可用。",
        )

    def metadata(self) -> dict[str, object]:
        return {
            "timestamp_model": "FireRedASR2-CTC-ONNX",
            "timestamp_provider": self._ort_provider,
            "timestamp_model_dir": str(self._asr_model_dir.resolve()),
        }

    def align(self, audio_path: Path, text: str) -> tuple[CharacterAlignment, ...]:
        self._ensure_process()
        response = self._request(
            {"audio_path": str(audio_path.resolve()), "text": text}
        )
        if response.get("status") != WORKER_RESULT_STATUS:
            raise AudioTranscriptionError(
                WORKER_ERROR.format(message=response.get("error", response))
            )
        value = response.get("alignments")
        if not isinstance(value, list):
            raise AudioTranscriptionError(
                WORKER_ERROR.format(message="alignments 不是数组")
            )
        return tuple(_parse_alignment(item) for item in value)

    def close(self) -> None:
        self._worker_process.close()

    def _ensure_process(self) -> None:
        if self._worker_process.is_running:
            return
        self._worker_process.start()
        try:
            response = self._worker_process.read_response()
        except (WorkerExitedError, WorkerResponseError) as error:
            raise AudioTranscriptionError(
                WORKER_ERROR.format(message=_worker_error_message(error))
            ) from error
        if response.get("status") != WORKER_READY_STATUS:
            self.close()
            raise AudioTranscriptionError(
                WORKER_ERROR.format(message=response.get("error", response))
            )

    def _request(
        self,
        payload: dict[str, object],
    ) -> dict[str, object]:
        self._worker_process.write_request(payload)
        try:
            return self._worker_process.read_response()
        except (WorkerExitedError, WorkerResponseError) as error:
            raise AudioTranscriptionError(
                WORKER_ERROR.format(message=_worker_error_message(error))
            ) from error


def _parse_alignment(value: object) -> CharacterAlignment:
    if not isinstance(value, dict):
        raise AudioTranscriptionError(WORKER_ERROR.format(message=value))
    text = value.get("text")
    confidence = value.get("confidence")
    if (
        not isinstance(text, str)
        or not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not 0 <= float(confidence) <= 1
    ):
        raise AudioTranscriptionError(WORKER_ERROR.format(message=value))
    return CharacterAlignment(
        text=text,
        source_start=_parse_integer(value, "source_start"),
        source_end=_parse_integer(value, "source_end"),
        start_ms=_parse_integer(value, "start_ms"),
        end_ms=_parse_integer(value, "end_ms"),
        confidence=float(confidence),
    )


def _parse_integer(payload: dict[str, object], field_name: str) -> int:
    value = payload.get(field_name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise AudioTranscriptionError(WORKER_ERROR.format(message=payload))
    return value


def _worker_error_message(error: WorkerExitedError | WorkerResponseError) -> object:
    if isinstance(error, WorkerExitedError):
        return f"进程退出：{error.returncode}"
    return error.response


def _require_file(path: Path, field_name: str) -> None:
    if not path.is_file():
        raise ValueError(f"{field_name} 指向的文件不存在：{path}")


def _require_model_files(model_dir: Path) -> None:
    missing = [
        name
        for name in REQUIRED_ALIGNMENT_MODEL_FILES
        if not (model_dir / name).is_file()
    ]
    if missing:
        raise ValueError(f"FireRed CTC 模型目录缺少文件：{', '.join(missing)}")
