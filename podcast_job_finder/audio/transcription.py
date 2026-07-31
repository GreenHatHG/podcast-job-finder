from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from podcast_job_finder.audio.segment_export import ExportedSpeechSegment


logger = logging.getLogger(__name__)


class AudioTranscriptionError(RuntimeError):
    """音频片段无法完成转写时抛出的错误。"""


@dataclass(slots=True, frozen=True)
class TimedTranscriptionText:
    text: str
    start_ms: int
    end_ms: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "text": self.text,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
        }


def parse_timed_transcription_texts(
    value: object,
    *,
    field_name: str,
) -> tuple[TimedTranscriptionText, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{field_name} 必须是数组。")
    return tuple(
        _parse_timed_transcription_text(item, field_name=field_name, index=index)
        for index, item in enumerate(value)
    )


@dataclass(slots=True, frozen=True)
class TranscriptionOutput:
    text: str
    # 记录每个字或词在原音频里出现的时间，主要用于处理相邻片段的重复内容。
    character_timestamps: tuple[TimedTranscriptionText, ...] = ()
    # 按完整句子记录起止时间，主要用于字幕切句和阅读展示。
    sentences: tuple[TimedTranscriptionText, ...] = ()


class AudioTranscriberProtocol(Protocol):
    def transcribe(
        self,
        segment: ExportedSpeechSegment,
        *,
        previous_segment: TranscribedSpeechSegment | None,
    ) -> TranscriptionOutput: ...

    def close(self) -> None: ...


@dataclass(slots=True, frozen=True)
class TranscribedSpeechSegment:
    index: int
    start_ms: int
    end_ms: int
    text: str
    character_timestamps: tuple[TimedTranscriptionText, ...] = ()
    sentences: tuple[TimedTranscriptionText, ...] = ()

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "index": self.index,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "text": self.text,
        }
        if self.character_timestamps:
            payload["character_timestamps"] = [
                timestamp.to_dict() for timestamp in self.character_timestamps
            ]
        if self.sentences:
            payload["sentences"] = [sentence.to_dict() for sentence in self.sentences]
        return payload


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
    transcriber: AudioTranscriberProtocol,
) -> AudioTranscriptionResult:
    transcribed_segments: list[TranscribedSpeechSegment] = []
    previous_segment = None
    for segment in segments:
        transcribed_segment = transcribe_speech_segment(
            segment,
            transcriber=transcriber,
            previous_segment=previous_segment,
        )
        transcribed_segments.append(transcribed_segment)
        previous_segment = transcribed_segment
    return AudioTranscriptionResult(segments=transcribed_segments)


def transcribe_speech_segment(
    segment: ExportedSpeechSegment,
    *,
    transcriber: AudioTranscriberProtocol,
    previous_segment: TranscribedSpeechSegment | None = None,
) -> TranscribedSpeechSegment:
    logger.info(
        "识别音频片段：index=%d start_ms=%d end_ms=%d",
        segment.index,
        segment.segment.start_ms,
        segment.segment.end_ms,
    )
    output = transcriber.transcribe(
        segment,
        previous_segment=previous_segment,
    )
    return TranscribedSpeechSegment(
        index=segment.index,
        start_ms=segment.segment.start_ms,
        end_ms=segment.segment.end_ms,
        text=output.text,
        character_timestamps=output.character_timestamps,
        sentences=output.sentences,
    )


def _parse_timed_transcription_text(
    value: object,
    *,
    field_name: str,
    index: int,
) -> TimedTranscriptionText:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name}[{index}] 必须是对象。")
    text = value.get("text")
    start_ms = value.get("start_ms")
    end_ms = value.get("end_ms")
    if not isinstance(text, str) or not text:
        raise ValueError(f"{field_name}[{index}] 内容无效。")
    parsed_start_ms, parsed_end_ms = _parse_timestamp_range(
        start_ms,
        end_ms,
        field_name=field_name,
        index=index,
    )
    return TimedTranscriptionText(
        text=text,
        start_ms=parsed_start_ms,
        end_ms=parsed_end_ms,
    )


def _parse_timestamp_range(
    start_ms: object,
    end_ms: object,
    *,
    field_name: str,
    index: int,
) -> tuple[int, int]:
    if not isinstance(start_ms, int) or isinstance(start_ms, bool):
        raise ValueError(f"{field_name}[{index}] 内容无效。")
    if not isinstance(end_ms, int) or isinstance(end_ms, bool):
        raise ValueError(f"{field_name}[{index}] 内容无效。")
    if not 0 <= start_ms < end_ms:
        raise ValueError(f"{field_name}[{index}] 内容无效。")
    return start_ms, end_ms
