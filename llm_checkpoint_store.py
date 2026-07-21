from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from company_extraction import CompanyExtractionResult
from podcast_job_finder.filesystem import (
    DEFAULT_FILE_CREATION_MODE,
    atomic_write_text,
)


CHECKPOINT_ROOT_DIR: Final = os.path.join("output", "checkpoints", "episodes")
STATE_FILE_NAME: Final = "llm_state.json"
PROMPT_FILE_NAME: Final = "llm_prompt.txt"
RESPONSE_FILE_NAME: Final = "llm_response.txt"
STATUS_PREPARED: Final = "prepared"
STATUS_SUCCESS: Final = "success"
STATUS_FAILED: Final = "failed"
VALID_STATUSES: Final = frozenset({STATUS_PREPARED, STATUS_SUCCESS, STATUS_FAILED})
URL_HASH_PREFIX: Final = "url_"
URL_HASH_LENGTH: Final = 16

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class LlmCheckpointState:  # pylint: disable=too-many-instance-attributes
    status: str
    episode_url: str
    title: str | None
    pub_date: str | None
    runtime_signature: str
    companies: list[dict]
    filtered_count: int
    error: str | None
    updated_at: str

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "episode_url": self.episode_url,
            "title": self.title,
            "pub_date": self.pub_date,
            "runtime_signature": self.runtime_signature,
            "companies": self.companies,
            "filtered_count": self.filtered_count,
            "error": self.error,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "LlmCheckpointState":
        if not isinstance(payload, dict):
            raise ValueError("检查点状态必须是对象。")

        status = _parse_checkpoint_status(payload.get("status"))
        episode_url = _require_non_empty_checkpoint_text(
            payload.get("episode_url"),
            "检查点缺少有效的 episode_url。",
        )
        runtime_signature = _require_non_empty_checkpoint_text(
            payload.get("runtime_signature"),
            "检查点缺少有效的 runtime_signature。",
        )
        filtered_count = _require_checkpoint_integer(
            payload.get("filtered_count"),
            "检查点中的 filtered_count 必须是整数。",
        )
        companies = _require_checkpoint_companies(payload.get("companies", []))
        title = _require_optional_checkpoint_text(
            payload.get("title"),
            "检查点中的 title 必须是字符串或 null。",
        )
        pub_date = _require_optional_checkpoint_text(
            payload.get("pub_date"),
            "检查点中的 pub_date 必须是字符串或 null。",
        )
        error = _require_optional_checkpoint_text(
            payload.get("error"),
            "检查点中的 error 必须是字符串或 null。",
        )
        updated_at = _require_non_empty_checkpoint_text(
            payload.get("updated_at"),
            "检查点缺少有效的 updated_at。",
        )

        return cls(
            status=status,
            episode_url=episode_url,
            title=title,
            pub_date=pub_date,
            runtime_signature=runtime_signature,
            companies=companies,
            filtered_count=filtered_count,
            error=error,
            updated_at=updated_at,
        )


def _parse_checkpoint_status(value: object) -> str:
    if value not in VALID_STATUSES:
        raise ValueError("检查点状态无效。")
    return str(value)


def _require_non_empty_checkpoint_text(value: object, error_message: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(error_message)
    return value


def _require_optional_checkpoint_text(
    value: object,
    error_message: str,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(error_message)
    return value


def _require_checkpoint_integer(value: object, error_message: str) -> int:
    if not isinstance(value, int):
        raise ValueError(error_message)
    return value


def _require_checkpoint_companies(value: object) -> list[dict]:
    if not isinstance(value, list):
        raise ValueError("检查点中的 companies 必须是数组。")

    CompanyExtractionResult.from_dict(
        {
            "companies": value,
            "filtered_count": 0,
        }
    )
    return value


@dataclass(slots=True, frozen=True)
class LlmCheckpoint:
    directory_path: str
    state: LlmCheckpointState
    prompt_text: str | None
    response_text: str | None


@dataclass(slots=True, frozen=True)
class LlmCheckpointSavePayload:
    episode_key: str
    episode_url: str
    title: str | None
    pub_date: str | None
    runtime_signature: str
    prompt_text: str


class LlmCheckpointStore:
    def __init__(self, root_dir: str = CHECKPOINT_ROOT_DIR) -> None:
        self._root_dir = root_dir

    def build_episode_key(self, *, eid: str | None, episode_url: str) -> str:
        normalized_eid = (eid or "").strip()
        if normalized_eid:
            return normalized_eid

        normalized_url = episode_url.strip()
        digest = hashlib.sha256(normalized_url.encode("utf-8")).hexdigest()
        return f"{URL_HASH_PREFIX}{digest[:URL_HASH_LENGTH]}"

    def load(self, episode_key: str) -> LlmCheckpoint | None:
        directory_path = self._build_episode_directory(episode_key)
        state_path = os.path.join(directory_path, STATE_FILE_NAME)
        if not os.path.exists(state_path):
            return None

        try:
            state_payload = self._read_json_file(state_path)
            state = LlmCheckpointState.from_dict(state_payload)
            prompt_text = self._read_optional_text_file(
                os.path.join(directory_path, PROMPT_FILE_NAME)
            )
            response_text = self._read_optional_text_file(
                os.path.join(directory_path, RESPONSE_FILE_NAME)
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            logger.warning(
                "读取检查点失败，已忽略：key=%s error=%s", episode_key, error
            )
            return None

        return LlmCheckpoint(
            directory_path=directory_path,
            state=state,
            prompt_text=prompt_text,
            response_text=response_text,
        )

    def save_prepared(self, payload: LlmCheckpointSavePayload) -> None:
        self._save_checkpoint_state(
            payload=payload,
            status=STATUS_PREPARED,
            response_text=None,
        )

    def save_success(
        self,
        payload: LlmCheckpointSavePayload,
        *,
        response_text: str,
        extraction_result: CompanyExtractionResult,
    ) -> None:
        self._save_checkpoint_state(
            payload=payload,
            status=STATUS_SUCCESS,
            companies=[company.to_dict() for company in extraction_result.companies],
            filtered_count=extraction_result.filtered_count,
            response_text=response_text,
        )

    def save_failed(
        self,
        payload: LlmCheckpointSavePayload,
        *,
        error_message: str,
        response_text: str | None,
    ) -> None:
        self._save_checkpoint_state(
            payload=payload,
            status=STATUS_FAILED,
            error=error_message,
            response_text=response_text,
        )

    def _save_checkpoint_state(
        self,
        *,
        payload: LlmCheckpointSavePayload,
        status: str,
        response_text: str | None,
        companies: list[dict] | None = None,
        filtered_count: int = 0,
        error: str | None = None,
    ) -> None:
        self._save_checkpoint(
            episode_key=payload.episode_key,
            state=self._build_checkpoint_state(
                payload=payload,
                status=status,
                companies=companies,
                filtered_count=filtered_count,
                error=error,
            ),
            prompt_text=payload.prompt_text,
            response_text=response_text,
        )

    def _build_checkpoint_state(
        self,
        *,
        payload: LlmCheckpointSavePayload,
        status: str,
        companies: list[dict] | None = None,
        filtered_count: int = 0,
        error: str | None = None,
    ) -> LlmCheckpointState:
        return LlmCheckpointState(
            status=status,
            episode_url=payload.episode_url,
            title=payload.title,
            pub_date=payload.pub_date,
            runtime_signature=payload.runtime_signature,
            companies=[] if companies is None else companies,
            filtered_count=filtered_count,
            error=error,
            updated_at=_build_updated_at(),
        )

    def _save_checkpoint(
        self,
        *,
        episode_key: str,
        state: LlmCheckpointState,
        prompt_text: str,
        response_text: str | None,
    ) -> None:
        directory_path = self._build_episode_directory(episode_key)
        os.makedirs(directory_path, exist_ok=True)
        self._write_json_file(
            os.path.join(directory_path, STATE_FILE_NAME),
            state.to_dict(),
        )
        self._write_text_file(
            os.path.join(directory_path, PROMPT_FILE_NAME),
            prompt_text,
        )
        response_path = os.path.join(directory_path, RESPONSE_FILE_NAME)
        if response_text is None:
            self._remove_file_if_exists(response_path)
        else:
            self._write_text_file(response_path, response_text)

    def _build_episode_directory(self, episode_key: str) -> str:
        return os.path.join(self._root_dir, episode_key)

    def _read_json_file(self, path: str) -> object:
        with open(path, encoding="utf-8") as file_obj:
            return json.load(file_obj)

    def _read_optional_text_file(self, path: str) -> str | None:
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as file_obj:
            return file_obj.read()

    def _write_json_file(self, path: str, payload: object) -> None:
        content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        self._write_text_file(path, content)

    def _write_text_file(self, path: str, content: str) -> None:
        atomic_write_text(
            Path(path),
            content,
            mode=DEFAULT_FILE_CREATION_MODE,
        )

    def _remove_file_if_exists(self, path: str) -> None:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def _build_updated_at() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
