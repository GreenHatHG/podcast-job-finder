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

from worker_text_alignment import FireRedTextAligner  # type: ignore[import-not-found]


READY_STATUS = "ready"
RESULT_STATUS = "result"
ERROR_STATUS = "error"
SHUTDOWN_COMMAND = "shutdown"
logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    args = _build_argument_parser().parse_args()
    try:
        load_started = time.perf_counter()
        aligner = FireRedTextAligner(
            args.asr_model_dir,
            provider=args.ort_provider,
            intra_op_threads=args.ort_intra_op_threads,
        )
        _write_response(
            {
                "status": READY_STATUS,
                "model_load_seconds": round(time.perf_counter() - load_started, 4),
            }
        )
    except Exception as error:  # pylint: disable=broad-exception-caught
        _write_response({"status": ERROR_STATUS, "error": str(error)})
        return 1

    for line in sys.stdin:
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("FireRed CTC 请求必须是 JSON 对象。")
            if request.get("command") == SHUTDOWN_COMMAND:
                return 0
            audio_path = _require_audio_path(request)
            text = _require_text(request)
            started = time.perf_counter()
            alignments = aligner.align(audio_path, text)
            _write_response(
                {
                    "status": RESULT_STATUS,
                    "alignments": [
                        {
                            "text": item.text,
                            "source_start": item.source_start,
                            "source_end": item.source_end,
                            "start_ms": item.start_ms,
                            "end_ms": item.end_ms,
                            "confidence": round(item.confidence, 8),
                        }
                        for item in alignments
                    ],
                    "alignment_seconds": round(time.perf_counter() - started, 4),
                }
            )
        except Exception as error:  # pylint: disable=broad-exception-caught
            logger.exception("FireRed CTC 时间匹配失败")
            _write_response({"status": ERROR_STATUS, "error": str(error)})
    return 0


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr-model-dir", type=Path, required=True)
    parser.add_argument("--ort-provider", required=True)
    parser.add_argument("--ort-intra-op-threads", type=int, required=True)
    return parser


def _require_audio_path(request: dict[str, Any]) -> Path:
    value = request.get("audio_path")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("FireRed CTC 请求缺少 audio_path。")
    path = Path(value)
    if not path.is_file():
        raise ValueError(f"FireRed CTC 输入音频不存在：{path}")
    return path


def _require_text(request: dict[str, Any]) -> str:
    value = request.get("text")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("FireRed CTC 请求缺少 text。")
    return value


def _write_response(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
