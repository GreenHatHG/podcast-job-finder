from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Final, Sequence

from podcast_job_finder.transcription.formatting.article import (
    FORMATTED_TRANSCRIPTION_ARTICLE_FILE_NAME,
    save_transcription_article,
)
from podcast_job_finder.transcription.formatting.formatter import (
    format_transcription_segments,
)
from podcast_job_finder.transcription.formatting.input import (
    load_transcription_inputs,
)
from podcast_job_finder.errors import PodcastJobFinderError
from podcast_job_finder.llm import load_transcription_formatting_llm_runtime_from_env
from podcast_job_finder.logging import configure_logging


PROGRAM_NAME: Final = "podcast-merge-transcriptions"


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    args = _build_argument_parser().parse_args(argv)
    try:
        formatting_runtime = load_transcription_formatting_llm_runtime_from_env()
        loaded_input = load_transcription_inputs(args.inputs)
        output_path = args.output or _build_default_output_path(loaded_input.json_paths)
        title = args.title or loaded_input.title
        formatted_article = format_transcription_segments(
            loaded_input.segments,
            llm_client=formatting_runtime.client,
            retry_config=formatting_runtime.retry_config,
            max_workers=formatting_runtime.max_in_flight_requests,
        )
        save_transcription_article(
            output_path,
            title=title,
            body=formatted_article.text,
        )
    except PodcastJobFinderError as error:
        print(str(error), file=sys.stderr)
        return 1

    result = {
        "article_path": str(output_path),
        "model": formatting_runtime.model,
        "input_file_count": len(loaded_input.json_paths),
        "segment_count": len(loaded_input.segments),
        "formatting": formatted_article.to_machine_audit_dict(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROGRAM_NAME)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--title")
    return parser


def _build_default_output_path(json_paths: Sequence[Path]) -> Path:
    first_parent = json_paths[0].parent
    if all(path.parent == first_parent for path in json_paths):
        return first_parent / FORMATTED_TRANSCRIPTION_ARTICLE_FILE_NAME
    return Path(FORMATTED_TRANSCRIPTION_ARTICLE_FILE_NAME)


if __name__ == "__main__":
    raise SystemExit(main())
