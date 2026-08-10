from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final


MAX_UNCOVERED_SPEECH_MS: Final = 2_000
MIN_SPEECH_COVERAGE_RATIO: Final = 0.60
LOW_CHARACTER_CONFIDENCE_THRESHOLD: Final = 0.05


class SpeechCoverageStatus(StrEnum):
    NO_SPEECH = "no_speech"
    NO_TRANSCRIPT = "no_transcript"
    ALIGNMENT_FAILED = "alignment_failed"
    LOW_COVERAGE = "low_coverage"
    SUFFICIENT_COVERAGE = "sufficient_coverage"


@dataclass(slots=True, frozen=True)
class TruncationAssessment:
    """记录识别结果里是否有一段人声没有对应文字。

    程序会把原音频中有人说话的部分，与已经得到文字的部分进行比较。
    这个类保存比较结果，供后续流程决定是否需要提醒、检查或重新转写。
    """

    # 最长一段“有人说话但没有对应文字”的持续时间，单位是毫秒。
    # 例如值为 3_000，表示最长有 3 秒人声没有对应文字。
    longest_uncovered_speech_ms: int

    # 这段没有对应文字的人声在音频中的开始时间，单位是毫秒。
    longest_uncovered_speech_start_ms: int

    # 这段没有对应文字的人声在音频中的结束时间，单位是毫秒。
    longest_uncovered_speech_end_ms: int

    # 原音频中有人说话的内容，有多少比例已经有对应文字。
    # 例如 0.8 表示大约八成说话内容已经被识别出来。
    speech_coverage_ratio: float

    # 说话内容和文字的对应情况，以及对应不足的原因。
    speech_coverage_status: SpeechCoverageStatus

    # 是否有一段没有对应文字的人声超过允许的时长。
    has_long_uncovered_speech: bool

    @property
    def is_anomalous(self) -> bool:
        """说明这次转写是否存在值得检查的异常。"""
        return self.has_long_uncovered_speech or self.speech_coverage_status not in {
            SpeechCoverageStatus.NO_SPEECH,
            SpeechCoverageStatus.SUFFICIENT_COVERAGE,
        }

    @property
    def requires_truncation_probe(self) -> bool:
        """说明是否需要单独识别没有对应文字的那段音频。"""
        if self.speech_coverage_status is SpeechCoverageStatus.NO_TRANSCRIPT:
            return True
        return (
            self.speech_coverage_status is SpeechCoverageStatus.LOW_COVERAGE
            and self.has_long_uncovered_speech
        )

    def to_dict(self) -> dict[str, object]:
        """把检查结果整理成便于保存和输出的普通字典。"""
        return {
            # 保存没有对应文字的人声的长度和位置，方便之后定位原音频检查。
            "longest_uncovered_speech_ms": self.longest_uncovered_speech_ms,
            "longest_uncovered_speech_start_ms": (
                self.longest_uncovered_speech_start_ms
            ),
            "longest_uncovered_speech_end_ms": self.longest_uncovered_speech_end_ms,
            # 最多保留四位小数，让保存的结果简洁，同时足够判断文字是否完整。
            "speech_coverage_ratio": round(self.speech_coverage_ratio, 4),
            "speech_coverage_status": self.speech_coverage_status.value,
            # 同时保存具体问题和派生判断，查看结果时无需再次计算。
            "has_long_uncovered_speech": self.has_long_uncovered_speech,
            "is_anomalous": self.is_anomalous,
            "requires_truncation_probe": self.requires_truncation_probe,
        }


@dataclass(slots=True, frozen=True)
class TranscriptionAttemptDiagnostics:
    attempt: int
    assessment: TruncationAssessment
    raw_responses: tuple[dict[str, object], ...] = ()
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "attempt": self.attempt,
            "assessment": self.assessment.to_dict(),
        }
        if self.raw_responses:
            payload["raw_responses"] = list(self.raw_responses)
        if self.error:
            payload["error"] = self.error
        return payload


@dataclass(slots=True, frozen=True)
class TruncationProbeDiagnostics:
    start_ms: int
    end_ms: int
    text: str
    confirmed_speech: bool
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "text": self.text,
            "confirmed_speech": self.confirmed_speech,
        }
        if self.error:
            payload["error"] = self.error
        return payload


@dataclass(slots=True, frozen=True)
class TranscriptionDiagnostics:
    selected_attempt: int
    attempts: tuple[TranscriptionAttemptDiagnostics, ...]
    truncation_probe: TruncationProbeDiagnostics | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "selected_attempt": self.selected_attempt,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }
        if self.truncation_probe is not None:
            payload["truncation_probe"] = self.truncation_probe.to_dict()
        return payload


def parse_transcription_diagnostics(value: object) -> TranscriptionDiagnostics | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("diagnostics 必须是对象。")
    selected_attempt = _require_integer(value, "selected_attempt")
    raw_attempts = value.get("attempts")
    if not isinstance(raw_attempts, list) or not raw_attempts:
        raise ValueError("diagnostics.attempts 必须是非空数组。")
    return TranscriptionDiagnostics(
        selected_attempt=selected_attempt,
        attempts=tuple(_parse_attempt(item) for item in raw_attempts),
        truncation_probe=_parse_probe(value.get("truncation_probe")),
    )


def _parse_attempt(value: object) -> TranscriptionAttemptDiagnostics:
    if not isinstance(value, dict):
        raise ValueError("diagnostics.attempts 项必须是对象。")
    raw_responses = value.get("raw_responses", [])
    if not isinstance(raw_responses, list) or not all(
        isinstance(item, dict) for item in raw_responses
    ):
        raise ValueError("diagnostics.attempts.raw_responses 必须是对象数组。")
    error = value.get("error")
    if error is not None and not isinstance(error, str):
        raise ValueError("diagnostics.attempts.error 必须是字符串。")
    return TranscriptionAttemptDiagnostics(
        attempt=_require_integer(value, "attempt"),
        assessment=_parse_assessment(value.get("assessment")),
        raw_responses=tuple(raw_responses),
        error=error,
    )


def _parse_assessment(value: object) -> TruncationAssessment:
    if not isinstance(value, dict):
        raise ValueError("diagnostics.attempts.assessment 必须是对象。")
    coverage = value.get("speech_coverage_ratio")
    coverage_status = value.get("speech_coverage_status")
    long_gap = value.get("has_long_uncovered_speech")
    if not isinstance(coverage, (int, float)) or isinstance(coverage, bool):
        raise ValueError("speech_coverage_ratio 必须是数字。")
    if coverage_status is None:
        parsed_coverage_status = _infer_legacy_coverage_status(
            coverage=float(coverage),
            has_low_coverage=value.get("has_low_speech_coverage"),
        )
    else:
        if not isinstance(coverage_status, str):
            raise ValueError("speech_coverage_status 必须是字符串。")
        try:
            parsed_coverage_status = SpeechCoverageStatus(coverage_status)
        except ValueError as error:
            raise ValueError("speech_coverage_status 无效。") from error
    if not isinstance(long_gap, bool):
        raise ValueError("has_long_uncovered_speech 必须是布尔值。")
    return TruncationAssessment(
        longest_uncovered_speech_ms=_require_integer(
            value, "longest_uncovered_speech_ms"
        ),
        longest_uncovered_speech_start_ms=_require_integer(
            value,
            "longest_uncovered_speech_start_ms",
            default=0,
        ),
        longest_uncovered_speech_end_ms=_require_integer(
            value,
            "longest_uncovered_speech_end_ms",
            default=_require_integer(value, "longest_uncovered_speech_ms"),
        ),
        speech_coverage_ratio=float(coverage),
        speech_coverage_status=parsed_coverage_status,
        has_long_uncovered_speech=long_gap,
    )


def _infer_legacy_coverage_status(
    *,
    coverage: float,
    has_low_coverage: object,
) -> SpeechCoverageStatus:
    if coverage <= 0:
        return SpeechCoverageStatus.NO_TRANSCRIPT
    if has_low_coverage is True:
        return SpeechCoverageStatus.LOW_COVERAGE
    return SpeechCoverageStatus.SUFFICIENT_COVERAGE


def _parse_probe(value: object) -> TruncationProbeDiagnostics | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("diagnostics.truncation_probe 必须是对象。")
    text = value.get("text")
    confirmed_speech = value.get("confirmed_speech")
    error = value.get("error")
    if not isinstance(text, str):
        raise ValueError("diagnostics.truncation_probe.text 必须是字符串。")
    if not isinstance(confirmed_speech, bool):
        raise ValueError("diagnostics.truncation_probe.confirmed_speech 必须是布尔值。")
    if error is not None and not isinstance(error, str):
        raise ValueError("diagnostics.truncation_probe.error 必须是字符串。")
    return TruncationProbeDiagnostics(
        start_ms=_require_integer(value, "start_ms"),
        end_ms=_require_integer(value, "end_ms"),
        text=text,
        confirmed_speech=confirmed_speech,
        error=error,
    )


def _require_integer(
    payload: dict[str, object],
    field_name: str,
    *,
    default: int | None = None,
) -> int:
    value = payload.get(field_name, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field_name} 必须是整数。")
    return value
