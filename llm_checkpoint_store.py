from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final

from company_extraction import CompanyExtractionResult


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
class LlmCheckpointState:
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

        status = payload.get("status")
        if status not in VALID_STATUSES:
            raise ValueError("检查点状态无效。")

        episode_url = payload.get("episode_url")
        runtime_signature = payload.get("runtime_signature")
        filtered_count = payload.get("filtered_count")
        companies = payload.get("companies", [])
        if not isinstance(episode_url, str) or not episode_url.strip():
            raise ValueError("检查点缺少有效的 episode_url。")
        if not isinstance(runtime_signature, str) or not runtime_signature.strip():
            raise ValueError("检查点缺少有效的 runtime_signature。")
        if not isinstance(filtered_count, int):
            raise ValueError("检查点中的 filtered_count 必须是整数。")
        if not isinstance(companies, list):
            raise ValueError("检查点中的 companies 必须是数组。")

        CompanyExtractionResult.from_dict(
            {
                "companies": companies,
                "filtered_count": filtered_count,
            }
        )

        title = payload.get("title")
        pub_date = payload.get("pub_date")
        error = payload.get("error")
        updated_at = payload.get("updated_at")
        if title is not None and not isinstance(title, str):
            raise ValueError("检查点中的 title 必须是字符串或 null。")
        if pub_date is not None and not isinstance(pub_date, str):
            raise ValueError("检查点中的 pub_date 必须是字符串或 null。")
        if error is not None and not isinstance(error, str):
            raise ValueError("检查点中的 error 必须是字符串或 null。")
        if not isinstance(updated_at, str) or not updated_at.strip():
            raise ValueError("检查点缺少有效的 updated_at。")

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


@dataclass(slots=True, frozen=True)
class LlmCheckpoint:
    directory_path: str
    state: LlmCheckpointState
    prompt_text: str | None
    response_text: str | None


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

    def save_prepared(
        self,
        *,
        episode_key: str,
        episode_url: str,
        title: str | None,
        pub_date: str | None,
        runtime_signature: str,
        prompt_text: str,
    ) -> None:
        self._save_checkpoint(
            episode_key=episode_key,
            state=LlmCheckpointState(
                status=STATUS_PREPARED,
                episode_url=episode_url,
                title=title,
                pub_date=pub_date,
                runtime_signature=runtime_signature,
                companies=[],
                filtered_count=0,
                error=None,
                updated_at=_build_updated_at(),
            ),
            prompt_text=prompt_text,
            response_text=None,
        )

    def save_success(
        self,
        *,
        episode_key: str,
        episode_url: str,
        title: str | None,
        pub_date: str | None,
        runtime_signature: str,
        prompt_text: str,
        response_text: str,
        extraction_result: CompanyExtractionResult,
    ) -> None:
        self._save_checkpoint(
            episode_key=episode_key,
            state=LlmCheckpointState(
                status=STATUS_SUCCESS,
                episode_url=episode_url,
                title=title,
                pub_date=pub_date,
                runtime_signature=runtime_signature,
                companies=[
                    company.to_dict() for company in extraction_result.companies
                ],
                filtered_count=extraction_result.filtered_count,
                error=None,
                updated_at=_build_updated_at(),
            ),
            prompt_text=prompt_text,
            response_text=response_text,
        )

    def save_failed(
        self,
        *,
        episode_key: str,
        episode_url: str,
        title: str | None,
        pub_date: str | None,
        runtime_signature: str,
        prompt_text: str,
        error_message: str,
        response_text: str | None,
    ) -> None:
        self._save_checkpoint(
            episode_key=episode_key,
            state=LlmCheckpointState(
                status=STATUS_FAILED,
                episode_url=episode_url,
                title=title,
                pub_date=pub_date,
                runtime_signature=runtime_signature,
                companies=[],
                filtered_count=0,
                error=error_message,
                updated_at=_build_updated_at(),
            ),
            prompt_text=prompt_text,
            response_text=response_text,
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
        self._write_atomic_text(path, content)

    def _write_text_file(self, path: str, content: str) -> None:
        self._write_atomic_text(path, content)

    def _write_atomic_text(self, path: str, content: str) -> None:
        temp_path = f"{path}.tmp.{uuid.uuid4().hex}"
        try:
            with open(temp_path, "w", encoding="utf-8") as file_obj:
                file_obj.write(content)
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _remove_file_if_exists(self, path: str) -> None:
        try:
            os.remove(path)
        except FileNotFoundError:
            return


def _build_updated_at() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
