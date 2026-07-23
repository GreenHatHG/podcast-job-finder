from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Final, Protocol, Sequence

from openai_compatible_llm import AudioFormat
from podcast_job_finder.audio.segment_export import ExportedSpeechSegment


PREVIOUS_CONTEXT_MAX_CHARS: Final = 200
WAV_AUDIO_FORMAT: Final[AudioFormat] = "wav"
NO_PREVIOUS_CONTEXT_TEXT: Final = "无"
TRANSCRIPTION_PROMPT_TEMPLATE: Final = """你是专业的中文音频转写助手。

请准确转写这段播客音频，保留人名、公司名、产品名和英文词的原始表达。

上一片段末尾文本（仅供理解上下文，不要重复输出）：
{previous_context}

要求：
1. 只输出当前音频对应的转写正文。
2. 使用自然的中文标点，不添加解释、标题或 Markdown。
3. 音频开头与上一片段重复的内容只保留一次。
4. 无法确认的内容保留原始发音，不要编造。
"""
READ_AUDIO_ERROR_TEMPLATE: Final = "无法读取待识别音频：{path}，{error_message}"

logger = logging.getLogger(__name__)


class AudioTranscriptionError(RuntimeError):
    """音频片段无法完成转写时抛出的错误。"""


class AudioTranscriptionClientProtocol(Protocol):
    def transcribe_audio(
        self,
        audio_data: bytes,
        *,
        audio_format: AudioFormat,
        prompt: str,
    ) -> str:
        """返回音频内容的文字转写。"""
        ...


@dataclass(slots=True, frozen=True)
class TranscribedSpeechSegment:
    index: int
    start_ms: int
    end_ms: int
    text: str

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class AudioTranscriptionResult:
    segments: list[TranscribedSpeechSegment] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(segment.text for segment in self.segments)

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "segments": [segment.to_dict() for segment in self.segments],
        }


def transcribe_speech_segments(
    segments: Sequence[ExportedSpeechSegment],
    *,
    llm_client: AudioTranscriptionClientProtocol,
) -> AudioTranscriptionResult:
    transcribed_segments: list[TranscribedSpeechSegment] = []
    previous_text = ""
    for segment in segments:
        logger.info(
            "识别音频片段：index=%d start_ms=%d end_ms=%d",
            segment.index,
            segment.segment.start_ms,
            segment.segment.end_ms,
        )
        text = llm_client.transcribe_audio(
            _read_audio(segment.file_path),
            audio_format=WAV_AUDIO_FORMAT,
            prompt=_build_transcription_prompt(previous_text),
        )
        transcribed_segments.append(
            TranscribedSpeechSegment(
                index=segment.index,
                start_ms=segment.segment.start_ms,
                end_ms=segment.segment.end_ms,
                text=text,
            )
        )
        previous_text = text
    return AudioTranscriptionResult(segments=transcribed_segments)


def _build_transcription_prompt(previous_text: str) -> str:
    normalized_previous_text = " ".join(previous_text.split())
    previous_context = (
        normalized_previous_text[-PREVIOUS_CONTEXT_MAX_CHARS:]
        if normalized_previous_text
        else NO_PREVIOUS_CONTEXT_TEXT
    )
    return TRANSCRIPTION_PROMPT_TEMPLATE.format(previous_context=previous_context)


def _read_audio(audio_path: Path) -> bytes:
    try:
        return audio_path.read_bytes()
    except OSError as error:
        raise AudioTranscriptionError(
            READ_AUDIO_ERROR_TEMPLATE.format(
                path=audio_path,
                error_message=str(error),
            )
        ) from error
