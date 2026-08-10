from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterator, Protocol, Sequence, runtime_checkable

from podcast_job_finder.audio.segment_export import ExportedSpeechSegment
from podcast_job_finder.audio.transcription_diagnostics import (
    TranscriptionDiagnostics,
)


logger = logging.getLogger(__name__)


class AudioTranscriptionError(RuntimeError):
    """音频片段无法完成转写时抛出的错误。"""


PREVIOUS_SEGMENT_ORDER_ERROR = (
    "上一音频片段编号不连续：expected_index={expected_index} actual_index={actual_index} "
    "current_index={current_index}"
)


@dataclass(slots=True, frozen=True)
class TimedTranscriptionText:
    text: str
    start_ms: int
    end_ms: int
    confidence: float | None = None

    def to_dict(self) -> dict[str, int | float | str]:
        payload: dict[str, int | float | str] = {
            "text": self.text,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
        }
        if self.confidence is not None:
            payload["confidence"] = round(self.confidence, 8)
        return payload


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
    diagnostics: TranscriptionDiagnostics | None = None


class AudioTranscriberProtocol(Protocol):
    def transcribe(
        self,
        segment: ExportedSpeechSegment,
        *,
        previous_segment: TranscribedSpeechSegment | None,
    ) -> TranscriptionOutput: ...

    def close(self) -> None: ...


@runtime_checkable
class BatchAudioTranscriberProtocol(AudioTranscriberProtocol, Protocol):
    batch_size: int

    def transcribe_batches(
        self,
        segments: Sequence[ExportedSpeechSegment],
        *,
        previous_segment: TranscribedSpeechSegment | None,
    ) -> Iterator[Sequence[TranscriptionSegmentResult]]: ...


@dataclass(slots=True, frozen=True)
class TranscribedSpeechSegment:
    index: int
    start_ms: int
    end_ms: int
    text: str
    character_timestamps: tuple[TimedTranscriptionText, ...] = ()
    sentences: tuple[TimedTranscriptionText, ...] = ()
    diagnostics: TranscriptionDiagnostics | None = None

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
        if self.diagnostics is not None:
            payload["diagnostics"] = self.diagnostics.to_dict()
        return payload


@dataclass(slots=True, frozen=True)
class TranscriptionSegmentResult:
    """一次处理多个片段时，其中一个音频片段的转写结果。"""

    segment: ExportedSpeechSegment
    output: TranscriptionOutput
    previous_segment: TranscribedSpeechSegment | None


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


def transcribe_speech_segment(
    segment: ExportedSpeechSegment,
    *,
    transcriber: AudioTranscriberProtocol,
    previous_segment: TranscribedSpeechSegment | None = None,
) -> TranscribedSpeechSegment:
    validate_previous_segment_order(segment, previous_segment)
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
    return build_transcribed_speech_segment(segment, output)


def validate_previous_segment_order(
    segment: ExportedSpeechSegment,
    previous_segment: TranscribedSpeechSegment | None,
) -> None:
    if previous_segment is None:
        return
    expected_index = segment.index - 1
    if previous_segment.index != expected_index:
        raise ValueError(
            PREVIOUS_SEGMENT_ORDER_ERROR.format(
                expected_index=expected_index,
                actual_index=previous_segment.index,
                current_index=segment.index,
            )
        )


def build_transcribed_speech_segment(
    segment: ExportedSpeechSegment,
    output: TranscriptionOutput,
) -> TranscribedSpeechSegment:
    return TranscribedSpeechSegment(
        index=segment.index,
        start_ms=segment.segment.start_ms,
        end_ms=segment.segment.end_ms,
        text=output.text,
        character_timestamps=output.character_timestamps,
        sentences=output.sentences,
        diagnostics=output.diagnostics,
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
        confidence=_parse_optional_confidence(
            value, field_name=field_name, index=index
        ),
    )


def _parse_optional_confidence(
    value: dict[str, object],
    *,
    field_name: str,
    index: int,
) -> float | None:
    confidence = value.get("confidence")
    if confidence is None:
        return None
    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not 0 <= float(confidence) <= 1
    ):
        raise ValueError(f"{field_name}[{index}] confidence 无效。")
    return float(confidence)


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
    # 旧版转写结果偶尔会把最后一个字保存为零时长（start_ms == end_ms）。
    # 保留这条时间记录，才能复用已有清单；负数或倒序仍然视为无效。
    if not 0 <= start_ms <= end_ms:
        raise ValueError(f"{field_name}[{index}] 内容无效。")
    return start_ms, end_ms
