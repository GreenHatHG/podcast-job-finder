from __future__ import annotations

import argparse
import base64
import os
import sys
from pathlib import Path
from typing import Final, Sequence, cast

from openai import OpenAI, OpenAIError
from openai.types.responses import ResponseInputParam

from podcast_job_finder.http.user_agents import DEFAULT_BROWSER_USER_AGENT


PROGRAM_NAME: Final = "test-ark-audio-transcription"
ARK_BASE_URL: Final = "https://ark.cn-beijing.volces.com/api/v3"
ARK_API_KEY_ENV: Final = "ARK_API_KEY"
ARK_AUDIO_MODEL_ENV: Final = "ARK_AUDIO_MODEL"
DEFAULT_MODEL: Final = "doubao-seed-2-0-lite-260428"
MP3_MIME_TYPE: Final = "audio/mpeg"
MAX_AUDIO_SIZE_BYTES: Final = 25 * 1024 * 1024
MAX_RETRIES: Final = 0
MISSING_API_KEY_ERROR: Final = f"缺少环境变量：{ARK_API_KEY_ENV}"
MISSING_AUDIO_FILE_ERROR_TEMPLATE: Final = "音频文件不存在：{audio_path}"
AUDIO_FILE_TOO_LARGE_ERROR_TEMPLATE: Final = (
    "Base64 音频文件不能超过 25 MB：{audio_path}"
)
EMPTY_TRANSCRIPT_ERROR: Final = "方舟模型返回了空文本。"
TRANSCRIPTION_INSTRUCTIONS: Final = """你是专业的中文播客音频转写助手。

严格遵守以下要求：
1. 只输出音频对应的转写正文，不添加解释、标题或 Markdown。
2. 准确识别人名、公司名、产品名、专业术语和英文表达。
3. 根据语义和停顿添加自然、完整的中文标点。
4. 保留说话人的原意和口语表达，不总结、不改写、不补充内容。
5. 无法确认的内容按实际发音转写，不编造。
"""
TRANSCRIPTION_REQUEST: Final = "请完整转写这段音频。"


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_argument_parser().parse_args(argv)
    api_key = os.getenv(ARK_API_KEY_ENV, "").strip()
    if not api_key:
        print(MISSING_API_KEY_ERROR, file=sys.stderr)
        return 1
    if not args.audio_path.is_file():
        print(
            MISSING_AUDIO_FILE_ERROR_TEMPLATE.format(audio_path=args.audio_path),
            file=sys.stderr,
        )
        return 1
    if args.audio_path.stat().st_size > MAX_AUDIO_SIZE_BYTES:
        print(
            AUDIO_FILE_TOO_LARGE_ERROR_TEMPLATE.format(audio_path=args.audio_path),
            file=sys.stderr,
        )
        return 1

    try:
        client = OpenAI(
            base_url=ARK_BASE_URL,
            api_key=api_key,
            max_retries=MAX_RETRIES,
            default_headers={"User-Agent": DEFAULT_BROWSER_USER_AGENT},
        )
        response = client.responses.create(
            model=_load_model(),
            instructions=TRANSCRIPTION_INSTRUCTIONS,
            input=_build_response_input(args.audio_path),
        )
    except (OSError, OpenAIError) as error:
        print(str(error), file=sys.stderr)
        return 1

    transcript = response.output_text.strip()
    if not transcript:
        print(EMPTY_TRANSCRIPT_ERROR, file=sys.stderr)
        return 1
    print(transcript)
    return 0


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROGRAM_NAME)
    parser.add_argument("audio_path", type=Path, help="本地 MP3 文件路径")
    return parser


def _load_model() -> str:
    return os.getenv(ARK_AUDIO_MODEL_ENV, DEFAULT_MODEL).strip() or DEFAULT_MODEL


def _build_audio_data_url(audio_path: Path) -> str:
    encoded_audio = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    return f"data:{MP3_MIME_TYPE};base64,{encoded_audio}"


def _build_response_input(audio_path: Path) -> ResponseInputParam:
    # Ark's audio_url extension is not represented by the OpenAI SDK type yet.
    return cast(
        ResponseInputParam,
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "audio_url": _build_audio_data_url(audio_path),
                    },
                    {
                        "type": "input_text",
                        "text": TRANSCRIPTION_REQUEST,
                    },
                ],
            }
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main())
