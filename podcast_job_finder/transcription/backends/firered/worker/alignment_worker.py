from __future__ import annotations

# This script runs in the dedicated FireRed environment and imports sibling modules.
# pylint: disable=import-error

import argparse
import logging
import sys
import time
from pathlib import Path

from worker_text_alignment import FireRedTextAligner  # type: ignore[import-not-found]
from worker_protocol import (  # type: ignore[import-not-found]
    ERROR_STATUS,
    READY_STATUS,
    RESULT_STATUS,
    configure_logging,
    is_shutdown,
    parse_request,
    require_audio_path,
    require_text,
    write_response,
)

logger = logging.getLogger(__name__)


def main() -> int:
    configure_logging()
    args = _build_argument_parser().parse_args()
    try:
        load_started = time.perf_counter()
        aligner = FireRedTextAligner(
            args.asr_model_dir,
            provider=args.ort_provider,
            intra_op_threads=args.ort_intra_op_threads,
        )
        write_response(
            {
                "status": READY_STATUS,
                "model_load_seconds": round(time.perf_counter() - load_started, 4),
            }
        )
    except Exception as error:  # pylint: disable=broad-exception-caught
        logger.exception("FireRed CTC 初始化失败")
        write_response({"status": ERROR_STATUS, "error": str(error)})
        return 1

    for line in sys.stdin:
        try:
            request = parse_request(line, request_name="FireRed CTC")
            if is_shutdown(request):
                return 0
            audio_path = require_audio_path(request, request_name="FireRed CTC")
            text = require_text(request, request_name="FireRed CTC")
            started = time.perf_counter()
            alignments = aligner.align(audio_path, text)
            write_response(
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
            write_response({"status": ERROR_STATUS, "error": str(error)})
    return 0


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr-model-dir", type=Path, required=True)
    parser.add_argument("--ort-provider", required=True)
    parser.add_argument("--ort-intra-op-threads", type=int, required=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
