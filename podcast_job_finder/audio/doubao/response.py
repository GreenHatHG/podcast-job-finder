from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol


DOUBAO_RESPONSE_ERROR = "豆包 ASR 未返回最终识别结果：{path}"


class AsrResponseProtocol(Protocol):
    type: object
    text: str
    raw_json: dict[str, object] | None


@dataclass(slots=True, frozen=True)
class DoubaoResponseSummary:
    text: str
    raw_responses: tuple[dict[str, object], ...]
    has_final_response: bool
    has_error_response: bool


def build_doubao_response_summary(
    responses: Iterable[AsrResponseProtocol],
    *,
    final_response_type: object,
    error_response_type: object,
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
