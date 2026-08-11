from __future__ import annotations

import os
from typing import AsyncIterator, Callable, Protocol, cast

import certifi

from podcast_job_finder.transcription.models import AudioTranscriptionError

from .response import AsrResponseProtocol


DOUBAO_IMPORT_ERROR = (
    "豆包 ASR 依赖加载失败，请确认已安装 cryptography，"
    "并在系统中安装 libopus。原始错误：{error}"
)


class DoubaoResponseTypes(Protocol):
    FINAL_RESULT: object
    ERROR: object


def build_doubao_client() -> tuple[
    object,
    Callable[..., AsyncIterator[AsrResponseProtocol]],
    DoubaoResponseTypes,
]:
    _ensure_default_ca_file()
    try:
        # pylint: disable-next=import-outside-toplevel
        from doubaoime_asr import (  # type: ignore[import-untyped]
            ASRConfig,
            ResponseType,
            transcribe_stream,
        )
    except Exception as error:  # pragma: no cover - depends on system libopus
        raise AudioTranscriptionError(
            DOUBAO_IMPORT_ERROR.format(error=error)
        ) from error
    asr_config = ASRConfig(enable_speech_rejection=False)
    return cast(
        tuple[
            object,
            Callable[..., AsyncIterator[AsrResponseProtocol]],
            DoubaoResponseTypes,
        ],
        (
            asr_config,
            transcribe_stream,
            ResponseType,
        ),
    )


def _ensure_default_ca_file() -> None:
    if os.environ.get("SSL_CERT_FILE") or os.environ.get("SSL_CERT_DIR"):
        return
    os.environ["SSL_CERT_FILE"] = certifi.where()
