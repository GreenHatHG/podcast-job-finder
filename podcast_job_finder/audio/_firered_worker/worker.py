from __future__ import annotations

# This script runs in the dedicated FireRed environment and imports sibling modules.
# pylint: disable=import-error

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from worker_asr import FireRedOnnxAsr  # type: ignore[import-not-found]
from worker_punctuation import FireRedPunctuation  # type: ignore[import-not-found]


READY_STATUS = "ready"
RESULT_STATUS = "result"
ERROR_STATUS = "error"
SHUTDOWN_COMMAND = "shutdown"


logger = logging.getLogger(__name__)


class FireRedWorker:
    def __init__(
        self,
        *,
        asr_model_dir: Path,
        punc_model_dir: Path,
        ort_provider: str,
        ort_intra_op_threads: int,
    ) -> None:
        load_started = time.perf_counter()
        self._asr = FireRedOnnxAsr(
            asr_model_dir,
            provider=ort_provider,
            intra_op_threads=ort_intra_op_threads,
        )
        self._punctuation = FireRedPunctuation(punc_model_dir)
        self.load_seconds = time.perf_counter() - load_started

    def transcribe(
        self,
        audio_path: Path,
        *,
        discard_before_ms: int,
    ) -> dict[str, object]:
        asr_started = time.perf_counter()
        tokens = self._asr.transcribe(
            audio_path,
            discard_before_ms=discard_before_ms,
        )
        asr_seconds = time.perf_counter() - asr_started
        punc_started = time.perf_counter()
        text, sentences = self._punctuation.punctuate(tokens)
        punc_seconds = time.perf_counter() - punc_started
        return {
            "status": RESULT_STATUS,
            "text": text,
            "raw_text": "".join(token.text for token in tokens),
            "character_timestamps": [
                {
                    "text": token.text,
                    "start_ms": token.start_ms,
                    "end_ms": token.end_ms,
                }
                for token in tokens
            ],
            "sentences": [
                {
                    "text": sentence.text,
                    "start_ms": sentence.start_ms,
                    "end_ms": sentence.end_ms,
                }
                for sentence in sentences
            ],
            "timings": {
                "asr_seconds": round(asr_seconds, 4),
                "punc_seconds": round(punc_seconds, 4),
                "total_seconds": round(asr_seconds + punc_seconds, 4),
            },
        }


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    args = _build_argument_parser().parse_args()
    try:
        worker = FireRedWorker(
            asr_model_dir=args.asr_model_dir,
            punc_model_dir=args.punc_model_dir,
            ort_provider=args.ort_provider,
            ort_intra_op_threads=args.ort_intra_op_threads,
        )
    except Exception as error:  # pylint: disable=broad-exception-caught
        _write_response({"status": ERROR_STATUS, "error": str(error)})
        return 1

    _write_response(
        {
            "status": READY_STATUS,
            "model_load_seconds": round(worker.load_seconds, 4),
        }
    )
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("FireRed 请求必须是 JSON 对象。")
            if request.get("command") == SHUTDOWN_COMMAND:
                return 0
            audio_path = _require_audio_path(request)
            _write_response(
                worker.transcribe(
                    audio_path,
                    discard_before_ms=_load_discard_before_ms(request),
                )
            )
        except Exception as error:  # pylint: disable=broad-exception-caught
            logger.exception("FireRed 转写失败")
            _write_response({"status": ERROR_STATUS, "error": str(error)})
    return 0


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr-model-dir", type=Path, required=True)
    parser.add_argument("--punc-model-dir", type=Path, required=True)
    parser.add_argument("--ort-provider", required=True)
    parser.add_argument("--ort-intra-op-threads", type=int, required=True)
    return parser


def _require_audio_path(request: dict[str, Any]) -> Path:
    value = request.get("audio_path")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("FireRed 请求缺少 audio_path。")
    path = Path(value)
    if not path.is_file():
        raise ValueError(f"FireRed 输入音频不存在：{path}")
    return path


def _load_discard_before_ms(request: dict[str, Any]) -> int:
    value = request.get("discard_before_ms", 0)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("FireRed 请求中的 discard_before_ms 必须是非负整数。")
    return value


def _write_response(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
