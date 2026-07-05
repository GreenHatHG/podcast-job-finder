from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Final


AUTH_SESSION_FILE_NAME: Final = ".xiaoyuzhou_auth.json"
AUTH_SESSION_PATH: Final = Path(__file__).resolve().parent / AUTH_SESSION_FILE_NAME
AUTH_SESSION_MISSING_ERROR_TEMPLATE: Final = (
    "未找到小宇宙登录态文件：{file_name}。请先执行 send-code 和 login。"
)
AUTH_SESSION_INVALID_ERROR_TEMPLATE: Final = (
    "小宇宙登录态文件格式无效：{file_name}。请重新执行 login。"
)
AUTH_SESSION_SAVE_ERROR_TEMPLATE: Final = "写入小宇宙登录态文件失败：{error_message}"
UPDATED_AT_FIELD_NAME: Final = "updated_at"


class XiaoyuzhouAuthStoreError(ValueError):
    """Raised when the local Xiaoyuzhou auth session cannot be used."""


@dataclass(slots=True, frozen=True)
class XiaoyuzhouAuthSession:
    mobile_phone_number: str
    area_code: str
    uid: str
    access_token: str
    refresh_token: str
    updated_at: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def build_auth_session(
    *,
    mobile_phone_number: str,
    area_code: str,
    uid: str,
    access_token: str,
    refresh_token: str,
) -> XiaoyuzhouAuthSession:
    return XiaoyuzhouAuthSession(
        mobile_phone_number=mobile_phone_number,
        area_code=area_code,
        uid=uid,
        access_token=access_token,
        refresh_token=refresh_token,
        updated_at=_build_updated_at(),
    )


def load_auth_session() -> XiaoyuzhouAuthSession:
    if not AUTH_SESSION_PATH.exists():
        raise XiaoyuzhouAuthStoreError(
            AUTH_SESSION_MISSING_ERROR_TEMPLATE.format(file_name=AUTH_SESSION_FILE_NAME)
        )

    try:
        payload = json.loads(AUTH_SESSION_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise XiaoyuzhouAuthStoreError(
            AUTH_SESSION_INVALID_ERROR_TEMPLATE.format(file_name=AUTH_SESSION_FILE_NAME)
        ) from error

    if not isinstance(payload, dict):
        raise XiaoyuzhouAuthStoreError(
            AUTH_SESSION_INVALID_ERROR_TEMPLATE.format(file_name=AUTH_SESSION_FILE_NAME)
        )

    return XiaoyuzhouAuthSession(
        mobile_phone_number=_get_required_string(payload, "mobile_phone_number"),
        area_code=_get_required_string(payload, "area_code"),
        uid=_get_required_string(payload, "uid"),
        access_token=_get_required_string(payload, "access_token"),
        refresh_token=_get_required_string(payload, "refresh_token"),
        updated_at=_get_required_string(payload, UPDATED_AT_FIELD_NAME),
    )


def save_auth_session(session: XiaoyuzhouAuthSession) -> None:
    try:
        AUTH_SESSION_PATH.write_text(
            json.dumps(session.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise XiaoyuzhouAuthStoreError(
            AUTH_SESSION_SAVE_ERROR_TEMPLATE.format(error_message=str(error))
        ) from error


def update_auth_session_tokens(
    session: XiaoyuzhouAuthSession,
    *,
    access_token: str,
    refresh_token: str,
) -> XiaoyuzhouAuthSession:
    return replace(
        session,
        access_token=access_token,
        refresh_token=refresh_token,
        updated_at=_build_updated_at(),
    )


def _get_required_string(payload: dict[str, object], field_name: str) -> str:
    raw_value = payload.get(field_name)
    if not isinstance(raw_value, str):
        raise XiaoyuzhouAuthStoreError(
            AUTH_SESSION_INVALID_ERROR_TEMPLATE.format(file_name=AUTH_SESSION_FILE_NAME)
        )

    normalized_value = raw_value.strip()
    if not normalized_value:
        raise XiaoyuzhouAuthStoreError(
            AUTH_SESSION_INVALID_ERROR_TEMPLATE.format(file_name=AUTH_SESSION_FILE_NAME)
        )
    return normalized_value


def _build_updated_at() -> str:
    return datetime.now(timezone.utc).isoformat()
