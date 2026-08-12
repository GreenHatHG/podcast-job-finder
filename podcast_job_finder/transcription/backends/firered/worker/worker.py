from __future__ import annotations

# This script runs in the dedicated FireRed environment and imports sibling modules.
# pylint: disable=import-error

import argparse
import logging
import sys
import time
from pathlib import Path

from worker_asr import FireRedOnnxAsr  # type: ignore[import-not-found]
from worker_punctuation import FireRedPunctuation  # type: ignore[import-not-found]
from worker_protocol import (  # type: ignore[import-not-found]
    ERROR_STATUS,
    READY_STATUS,
    RESULT_STATUS,
    configure_logging,
    is_shutdown,
    parse_request,
    require_audio_path,
    require_nonnegative_int,
    write_response,
)


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
    configure_logging()
    args = _build_argument_parser().parse_args()
    try:
        worker = FireRedWorker(
            asr_model_dir=args.asr_model_dir,
            punc_model_dir=args.punc_model_dir,
            ort_provider=args.ort_provider,
            ort_intra_op_threads=args.ort_intra_op_threads,
        )
    except Exception as error:  # pylint: disable=broad-exception-caught
        write_response({"status": ERROR_STATUS, "error": str(error)})
        return 1

    write_response(
        {
            "status": READY_STATUS,
            "model_load_seconds": round(worker.load_seconds, 4),
        }
    )
    for line in sys.stdin:
        try:
            request = parse_request(line, request_name="FireRed")
            if is_shutdown(request):
                return 0
            audio_path = require_audio_path(request, request_name="FireRed")
            write_response(
                worker.transcribe(
                    audio_path,
                    discard_before_ms=require_nonnegative_int(
                        request,
                        "discard_before_ms",
                        request_name="FireRed",
                    ),
                )
            )
        except Exception as error:  # pylint: disable=broad-exception-caught
            logger.exception("FireRed 转写失败")
            write_response({"status": ERROR_STATUS, "error": str(error)})
    return 0


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr-model-dir", type=Path, required=True)
    parser.add_argument("--punc-model-dir", type=Path, required=True)
    parser.add_argument("--ort-provider", required=True)
    parser.add_argument("--ort-intra-op-threads", type=int, required=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
