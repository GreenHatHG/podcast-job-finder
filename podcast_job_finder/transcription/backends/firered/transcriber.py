from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Final

from podcast_job_finder.transcription.backends.firered.config import (
    DEFAULT_ORT_INTRA_OP_THREADS,
    DEFAULT_ORT_PROVIDER,
)
from podcast_job_finder.audio.segmentation.segment_export import ExportedSpeechSegment
from podcast_job_finder.audio.segmentation.speech_pipeline import (
    DEFAULT_SILENCE_PADDING_MS,
)
from podcast_job_finder.transcription.models import (
    AudioTranscriptionError,
    TimedTranscriptionText,
    TranscribedSpeechSegment,
    TranscriptionOutput,
)


FIRERED_MODEL_NAME: Final = "FireRedASR2-AED-ONNX+FireRedPunc"
WORKER_READY_STATUS: Final = "ready"
WORKER_RESULT_STATUS: Final = "result"
WORKER_SHUTDOWN_COMMAND: Final = "shutdown"
WORKER_CLOSE_TIMEOUT_SECONDS: Final = 5
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


class FireRedConfigError(ValueError):
    """FireRed 本地转写配置无效。"""


@dataclass(slots=True, frozen=True)
class FireRedTranscriberConfig:
    python_executable: Path
    asr_model_dir: Path
    punc_model_dir: Path
    ort_provider: str = DEFAULT_ORT_PROVIDER
    ort_intra_op_threads: int = DEFAULT_ORT_INTRA_OP_THREADS
    silence_padding_ms: int = DEFAULT_SILENCE_PADDING_MS

    def __post_init__(self) -> None:
        _require_file(self.python_executable, "FIRERED_PYTHON")
        _require_model_files(self.asr_model_dir, REQUIRED_ASR_FILES)
        _require_model_files(self.punc_model_dir, REQUIRED_PUNC_FILES)
        if self.ort_intra_op_threads <= 0:
            raise FireRedConfigError("FIRERED_ORT_INTRA_OP_THREADS 必须大于 0。")
        if self.silence_padding_ms < 0:
            raise FireRedConfigError("silence_padding_ms 必须大于等于 0。")

    def metadata(self) -> dict[str, object]:
        return {
            "transcription_backend": "firered",
            "model": FIRERED_MODEL_NAME,
            "provider": self.ort_provider,
            "asr_model_dir": str(self.asr_model_dir.resolve()),
            "punc_model_dir": str(self.punc_model_dir.resolve()),
        }


class FireRedAudioTranscriber:
    def __init__(self, config: FireRedTranscriberConfig) -> None:
        self._config = config
        self._process: subprocess.Popen[str] | None = None

    def transcribe(
        self,
        segment: ExportedSpeechSegment,
        *,
        previous_segment: TranscribedSpeechSegment | None,
    ) -> TranscriptionOutput:
        process = self._ensure_process()
        response = self._request(
            process,
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
        return (
            previous_end_ms - segment.segment.start_ms + self._config.silence_padding_ms
        )

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        try:
            self._write_request(process.stdin, {"command": WORKER_SHUTDOWN_COMMAND})
            process.wait(timeout=WORKER_CLOSE_TIMEOUT_SECONDS)
        except BrokenPipeError, OSError, subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=WORKER_CLOSE_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def _ensure_process(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process
        worker_path = Path(__file__).with_name("worker") / "worker.py"
        environment = os.environ.copy()
        environment.setdefault("TOKENIZERS_PARALLELISM", "false")
        try:
            process = subprocess.Popen(  # pylint: disable=consider-using-with
                [
                    str(self._config.python_executable),
                    str(worker_path),
                    "--asr-model-dir",
                    str(self._config.asr_model_dir),
                    "--punc-model-dir",
                    str(self._config.punc_model_dir),
                    "--ort-provider",
                    self._config.ort_provider,
                    "--ort-intra-op-threads",
                    str(self._config.ort_intra_op_threads),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
                env=environment,
            )
        except OSError as error:
            raise AudioTranscriptionError(
                WORKER_START_ERROR.format(message=str(error))
            ) from error
        self._process = process
        ready_response = self._read_response(process)
        if ready_response.get("status") != WORKER_READY_STATUS:
            self.close()
            raise AudioTranscriptionError(
                WORKER_START_ERROR.format(
                    message=ready_response.get("error", ready_response)
                )
            )
        return process

    def _request(
        self,
        process: subprocess.Popen[str],
        payload: dict[str, object],
    ) -> dict[str, object]:
        try:
            self._write_request(process.stdin, payload)
        except (BrokenPipeError, OSError) as error:
            raise AudioTranscriptionError(
                WORKER_EXIT_ERROR.format(returncode=process.poll())
            ) from error
        return self._read_response(process)

    @staticmethod
    def _write_request(stream: IO[str] | None, payload: dict[str, object]) -> None:
        if stream is None:
            raise BrokenPipeError("FireRed 工作进程标准输入不可用。")
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        stream.flush()

    @staticmethod
    def _read_response(process: subprocess.Popen[str]) -> dict[str, object]:
        if process.stdout is None:
            raise AudioTranscriptionError(
                WORKER_RESPONSE_ERROR.format(message="标准输出不可用")
            )
        response_line = process.stdout.readline()
        if not response_line:
            raise AudioTranscriptionError(
                WORKER_EXIT_ERROR.format(returncode=process.poll())
            )
        try:
            response = json.loads(response_line)
        except json.JSONDecodeError as error:
            raise AudioTranscriptionError(
                WORKER_RESPONSE_ERROR.format(message=response_line.strip())
            ) from error
        if not isinstance(response, dict):
            raise AudioTranscriptionError(
                WORKER_RESPONSE_ERROR.format(message=response)
            )
        return response

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
        absolute_start = _to_source_timestamp(
            start_ms,
            segment=segment,
            silence_padding_ms=self._config.silence_padding_ms,
        )
        absolute_end = _to_source_timestamp(
            end_ms,
            segment=segment,
            silence_padding_ms=self._config.silence_padding_ms,
        )
        if absolute_end <= absolute_start:
            return None
        return TimedTranscriptionText(
            text=text,
            start_ms=absolute_start,
            end_ms=absolute_end,
        )


def _to_source_timestamp(
    timestamp_ms: int,
    *,
    segment: ExportedSpeechSegment,
    silence_padding_ms: int,
) -> int:
    relative_timestamp = max(0, timestamp_ms - silence_padding_ms)
    return min(
        segment.segment.end_ms,
        segment.segment.start_ms + relative_timestamp,
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
