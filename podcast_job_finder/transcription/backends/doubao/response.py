from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

from podcast_job_finder.transcription.models import AudioTranscriptionError


DOUBAO_RESPONSE_ERROR = "豆包 ASR 未返回完整终止响应：{path}"


class DoubaoMissingFinalResponseError(AudioTranscriptionError):
    """豆包没有返回完整的最终结果和会话终止消息。"""


class AsrResponseProtocol(Protocol):
    type: object
    text: str
    raw_json: dict[str, object] | None


@dataclass(slots=True, frozen=True)
class DoubaoResponseSummary:
    text: str
    raw_responses: tuple[dict[str, object], ...]
    has_final_response: bool
    has_terminal_response: bool
    has_non_final_text: bool
    has_error_response: bool

    @property
    def is_complete(self) -> bool:
        """判断响应是否已经完整结束，可以生成最终输出。"""

        if self.has_error_response or not self.has_terminal_response:
            return False
        if self.has_final_response:
            return True
        return not self.has_non_final_text


def build_doubao_response_summary(
    responses: Iterable[AsrResponseProtocol],
    *,
    final_response_type: object,
    error_response_type: object,
    terminal_response_type: object | None = None,
) -> DoubaoResponseSummary:
    response_list = list(responses)
    final_responses = [
        response for response in response_list if response.type == final_response_type
    ]
    text = final_responses[-1].text.strip() if final_responses else ""
    return DoubaoResponseSummary(
        text=text,
        raw_responses=tuple(
            _serialize_raw_response(response)
            for response in response_list
            if response.raw_json is not None
        ),
        has_final_response=bool(final_responses),
        has_terminal_response=(
            terminal_response_type is not None
            and any(
                response.type == terminal_response_type for response in response_list
            )
        ),
        # 是否曾经返回过文字，但这些文字不是正式的 FINAL_RESULT
        # 没有返回正式最终结果，可能存在文字未完成，不能当成正常空转写
        has_non_final_text=any(
            response.type != final_response_type and response.text.strip()
            for response in response_list
        ),
        has_error_response=any(
            response.type == error_response_type for response in response_list
        ),
    )


def _serialize_raw_response(
    response: AsrResponseProtocol,
) -> dict[str, object]:
    response_type = getattr(response.type, "name", str(response.type))
    return {
        "response_type": response_type,
        "payload": response.raw_json,
    }
