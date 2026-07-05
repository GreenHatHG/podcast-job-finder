from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Any, Final, NoReturn

import requests

from http_user_agents import DEFAULT_BROWSER_USER_AGENT


DEFAULT_XYZ_BASE_URL: Final = "http://localhost:23020"
DEFAULT_AREA_CODE: Final = "+86"
REQUEST_TIMEOUT_SECONDS: Final = 30
JSON_CONTENT_TYPE: Final = "application/json"
USER_AGENT_HEADER_NAME: Final = "User-Agent"
CONTENT_TYPE_HEADER_NAME: Final = "Content-Type"
ACCESS_TOKEN_HEADER_NAME: Final = "x-jike-access-token"
SEND_CODE_PATH: Final = "/sendCode"
LOGIN_PATH: Final = "/login"
REFRESH_TOKEN_PATH: Final = "/refresh_token"
EPISODE_LIST_PATH: Final = "/episode_list"
DEFAULT_EPISODE_LIST_ORDER: Final = "desc"
UNEXPECTED_RESPONSE_ERROR_TEMPLATE: Final = "xyz 接口返回格式异常：{path}，{detail}"
REQUEST_FAILED_ERROR_TEMPLATE: Final = "请求 xyz 服务失败：{error_message}"
HTTP_REQUEST_FAILED_TEMPLATE: Final = (
    "调用 xyz 接口失败：{path}，HTTP {status_code}，{detail}"
)
INVALID_JSON_DETAIL_TEMPLATE: Final = (
    "响应体不是合法 JSON，HTTP {status_code}，body={body}"
)
UNEXPECTED_FIELD_DETAIL_TEMPLATE: Final = (
    "字段 {field_name} 需要 {expected_description}，实际值 {actual_value}"
)
DEBUG_REQUEST_TEMPLATE: Final = "xyz 请求 path=%s payload=%s"
DEBUG_RESPONSE_TEMPLATE: Final = "xyz 响应 path=%s status=%s body=%s"
DEBUG_PARSE_FAILURE_TEMPLATE: Final = (
    "xyz 响应无法解析为 JSON path=%s status=%s body=%s"
)
DEBUG_UNEXPECTED_RESPONSE_TEMPLATE: Final = (
    "xyz 响应字段异常 path=%s field=%s payload=%s"
)
DEBUG_TEXT_TRUNCATION_SUFFIX: Final = "...<truncated>"
MAX_DEBUG_TEXT_LENGTH: Final = 4000
MASKED_VALUE: Final = "***"
SENSITIVE_FIELD_NAMES: Final = frozenset(
    {
        "mobilePhoneNumber",
        "verifyCode",
        "x-jike-access-token",
        "x-jike-refresh-token",
    }
)


logger = logging.getLogger(__name__)


class XyzClientError(RuntimeError):
    """Raised when the local xyz service cannot satisfy the request."""


class XyzUnauthorizedError(XyzClientError):
    """Raised when the local xyz service reports an expired auth session."""


@dataclass(slots=True, frozen=True)
class LoginResult:
    uid: str
    access_token: str
    refresh_token: str


@dataclass(slots=True, frozen=True)
class RefreshedTokens:
    access_token: str
    refresh_token: str


@dataclass(slots=True, frozen=True)
class EpisodeLoadMoreKey:
    pub_date: str
    id: str
    direction: str


@dataclass(slots=True, frozen=True)
class PodcastEpisodeSummary:
    pid: str
    eid: str
    title: str
    pub_date: str


@dataclass(slots=True, frozen=True)
class EpisodeListPage:
    episodes: tuple[PodcastEpisodeSummary, ...]
    load_more_key: EpisodeLoadMoreKey | None


class XyzClient:
    def __init__(self, base_url: str = DEFAULT_XYZ_BASE_URL) -> None:
        self._base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update(
            {USER_AGENT_HEADER_NAME: DEFAULT_BROWSER_USER_AGENT}
        )

    def send_code(
        self,
        *,
        mobile_phone_number: str,
        area_code: str = DEFAULT_AREA_CODE,
    ) -> None:
        response_data = self._post_json(
            SEND_CODE_PATH,
            {
                "mobilePhoneNumber": mobile_phone_number,
                "areaCode": area_code,
            },
        )
        self._extract_local_payload(response_data, SEND_CODE_PATH)

    def login(
        self,
        *,
        mobile_phone_number: str,
        verify_code: str,
        area_code: str = DEFAULT_AREA_CODE,
    ) -> LoginResult:
        response_data = self._post_json(
            LOGIN_PATH,
            {
                "mobilePhoneNumber": mobile_phone_number,
                "verifyCode": verify_code,
                "areaCode": area_code,
            },
        )
        payload = self._extract_local_payload(response_data, LOGIN_PATH)
        user_payload = self._require_dict(
            payload.get("data"),
            LOGIN_PATH,
            field_name="data.data",
        )
        return LoginResult(
            uid=self._require_non_empty_string(
                user_payload.get("uid"),
                LOGIN_PATH,
                field_name="data.data.uid",
            ),
            access_token=self._require_non_empty_string(
                payload.get("x-jike-access-token"),
                LOGIN_PATH,
                field_name="data.x-jike-access-token",
            ),
            refresh_token=self._require_non_empty_string(
                payload.get("x-jike-refresh-token"),
                LOGIN_PATH,
                field_name="data.x-jike-refresh-token",
            ),
        )

    def refresh_token(
        self,
        *,
        access_token: str,
        refresh_token: str,
    ) -> RefreshedTokens:
        response_data = self._post_json(
            REFRESH_TOKEN_PATH,
            {
                "x-jike-access-token": access_token,
                "x-jike-refresh-token": refresh_token,
            },
        )
        payload = self._extract_local_payload(response_data, REFRESH_TOKEN_PATH)
        return RefreshedTokens(
            access_token=self._require_non_empty_string(
                payload.get("x-jike-access-token"),
                REFRESH_TOKEN_PATH,
                field_name="data.x-jike-access-token",
            ),
            refresh_token=self._require_non_empty_string(
                payload.get("x-jike-refresh-token"),
                REFRESH_TOKEN_PATH,
                field_name="data.x-jike-refresh-token",
            ),
        )

    def list_podcast_episodes(
        self,
        *,
        pid: str,
        access_token: str,
        load_more_key: EpisodeLoadMoreKey | None = None,
        order: str = DEFAULT_EPISODE_LIST_ORDER,
    ) -> EpisodeListPage:
        payload: dict[str, Any] = {
            "pid": pid,
            "order": order,
        }
        if load_more_key is not None:
            payload["loadMoreKey"] = {
                "pubDate": load_more_key.pub_date,
                "id": load_more_key.id,
                "direction": load_more_key.direction,
            }

        response_data = self._post_json(
            EPISODE_LIST_PATH,
            payload,
            access_token=access_token,
        )
        page_payload = self._extract_local_payload(response_data, EPISODE_LIST_PATH)
        episodes_data = page_payload.get("data")
        if not isinstance(episodes_data, list):
            self._raise_unexpected_response(
                path=EPISODE_LIST_PATH,
                field_name="data",
                expected_description="列表",
                payload=episodes_data,
            )

        episodes = tuple(
            self._parse_episode_summary(
                episode_data=episode_data,
                requested_pid=pid,
            )
            for episode_data in episodes_data
        )
        return EpisodeListPage(
            episodes=episodes,
            load_more_key=self._parse_load_more_key(page_payload.get("loadMoreKey")),
        )

    def _post_json(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        access_token: str | None = None,
    ) -> dict[str, Any]:
        request_headers = {CONTENT_TYPE_HEADER_NAME: JSON_CONTENT_TYPE}
        if access_token is not None:
            request_headers[ACCESS_TOKEN_HEADER_NAME] = access_token

        self._debug_log_request(path=path, payload=payload)
        try:
            response = self._session.post(
                f"{self._base_url}{path}",
                json=payload,
                headers=request_headers,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except requests.RequestException as error:
            raise XyzClientError(
                REQUEST_FAILED_ERROR_TEMPLATE.format(error_message=str(error))
            ) from error

        response_data = self._parse_response_json(response, path)
        if response.status_code == requests.codes.unauthorized:
            raise XyzUnauthorizedError(
                HTTP_REQUEST_FAILED_TEMPLATE.format(
                    path=path,
                    status_code=response.status_code,
                    detail=self._extract_error_detail(response_data),
                )
            )
        if response.status_code >= requests.codes.bad_request:
            raise XyzClientError(
                HTTP_REQUEST_FAILED_TEMPLATE.format(
                    path=path,
                    status_code=response.status_code,
                    detail=self._extract_error_detail(response_data),
                )
            )

        return response_data

    def _parse_response_json(
        self,
        response: requests.Response,
        path: str,
    ) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as error:
            response_body = self._truncate_debug_text(response.text)
            self._debug_log_parse_failure(
                path=path,
                status_code=response.status_code,
                response_text=response_body,
            )
            self._raise_response_format_error(
                path=path,
                detail=INVALID_JSON_DETAIL_TEMPLATE.format(
                    status_code=response.status_code,
                    body=response_body,
                ),
                cause=error,
            )

        self._debug_log_response(
            path=path,
            status_code=response.status_code,
            payload=payload,
        )
        if not isinstance(payload, dict):
            self._raise_unexpected_response(
                path=path,
                field_name="response",
                expected_description="对象",
                payload=payload,
            )
        return payload

    def _extract_local_payload(
        self,
        response_data: dict[str, Any],
        path: str,
    ) -> dict[str, Any]:
        payload = response_data.get("data")
        return self._require_dict(payload, path, field_name="data")

    def _parse_episode_summary(
        self,
        *,
        episode_data: object,
        requested_pid: str,
    ) -> PodcastEpisodeSummary:
        payload = self._require_dict(
            episode_data,
            EPISODE_LIST_PATH,
            field_name="episode",
        )
        eid = self._require_non_empty_string(
            payload.get("eid"),
            EPISODE_LIST_PATH,
            field_name="episode.eid",
        )
        pid = str(payload.get("pid") or requested_pid).strip()
        if not pid:
            self._raise_unexpected_response(
                path=EPISODE_LIST_PATH,
                field_name="episode.pid",
                expected_description="非空字符串",
                payload=payload.get("pid"),
            )
        return PodcastEpisodeSummary(
            pid=pid,
            eid=eid,
            title=str(payload.get("title") or "").strip(),
            pub_date=str(payload.get("pubDate") or "").strip(),
        )

    def _parse_load_more_key(
        self,
        load_more_key_data: object,
    ) -> EpisodeLoadMoreKey | None:
        if load_more_key_data is None:
            return None

        payload = self._require_dict(
            load_more_key_data,
            EPISODE_LIST_PATH,
            field_name="loadMoreKey",
        )
        return EpisodeLoadMoreKey(
            pub_date=self._require_non_empty_string(
                payload.get("pubDate"),
                EPISODE_LIST_PATH,
                field_name="loadMoreKey.pubDate",
            ),
            id=self._require_non_empty_string(
                payload.get("id"),
                EPISODE_LIST_PATH,
                field_name="loadMoreKey.id",
            ),
            direction=self._require_non_empty_string(
                payload.get("direction"),
                EPISODE_LIST_PATH,
                field_name="loadMoreKey.direction",
            ),
        )

    def _extract_error_detail(self, response_data: dict[str, Any]) -> str:
        detail = response_data.get("data")
        if isinstance(detail, str):
            normalized_detail = detail.strip()
            if normalized_detail:
                return normalized_detail

        message = response_data.get("msg")
        if isinstance(message, str):
            normalized_message = message.strip()
            if normalized_message:
                return normalized_message
        return "未知错误"

    def _require_dict(
        self,
        payload: object,
        path: str,
        *,
        field_name: str = "payload",
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            self._raise_unexpected_response(
                path=path,
                field_name=field_name,
                expected_description="对象",
                payload=payload,
            )
        return payload

    def _require_non_empty_string(
        self,
        value: object,
        path: str,
        *,
        field_name: str = "value",
    ) -> str:
        if not isinstance(value, str):
            self._raise_unexpected_response(
                path=path,
                field_name=field_name,
                expected_description="非空字符串",
                payload=value,
            )

        normalized_value = value.strip()
        if not normalized_value:
            self._raise_unexpected_response(
                path=path,
                field_name=field_name,
                expected_description="非空字符串",
                payload=value,
            )
        return normalized_value

    def _raise_unexpected_response(
        self,
        *,
        path: str,
        field_name: str,
        expected_description: str,
        payload: object,
    ) -> NoReturn:
        detail = UNEXPECTED_FIELD_DETAIL_TEMPLATE.format(
            field_name=field_name,
            expected_description=expected_description,
            actual_value=self._format_debug_payload(payload),
        )
        logger.debug(
            DEBUG_UNEXPECTED_RESPONSE_TEMPLATE,
            path,
            field_name,
            self._format_debug_payload(payload),
        )
        self._raise_response_format_error(path=path, detail=detail)

    def _raise_response_format_error(
        self,
        *,
        path: str,
        detail: str,
        cause: Exception | None = None,
    ) -> NoReturn:
        error = XyzClientError(
            UNEXPECTED_RESPONSE_ERROR_TEMPLATE.format(path=path, detail=detail)
        )
        if cause is not None:
            raise error from cause
        raise error

    def _debug_log_request(self, *, path: str, payload: dict[str, Any]) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return
        logger.debug(
            DEBUG_REQUEST_TEMPLATE,
            path,
            self._format_debug_payload(payload),
        )

    def _debug_log_response(
        self,
        *,
        path: str,
        status_code: int,
        payload: object,
    ) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return
        logger.debug(
            DEBUG_RESPONSE_TEMPLATE,
            path,
            status_code,
            self._format_debug_payload(payload),
        )

    def _debug_log_parse_failure(
        self,
        *,
        path: str,
        status_code: int,
        response_text: str,
    ) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return
        logger.debug(
            DEBUG_PARSE_FAILURE_TEMPLATE,
            path,
            status_code,
            self._truncate_debug_text(response_text),
        )

    def _format_debug_payload(self, payload: object) -> str:
        sanitized_payload = self._sanitize_debug_value(payload)
        try:
            serialized_payload = json.dumps(
                sanitized_payload,
                ensure_ascii=False,
                sort_keys=True,
            )
        except TypeError:
            serialized_payload = repr(sanitized_payload)
        return self._truncate_debug_text(serialized_payload)

    def _sanitize_debug_value(self, value: object) -> object:
        if isinstance(value, dict):
            sanitized_payload: dict[object, object] = {}
            for key, item in value.items():
                if isinstance(key, str) and key in SENSITIVE_FIELD_NAMES:
                    sanitized_payload[key] = MASKED_VALUE
                    continue
                sanitized_payload[key] = self._sanitize_debug_value(item)
            return sanitized_payload
        if isinstance(value, list):
            return [self._sanitize_debug_value(item) for item in value]
        if isinstance(value, tuple):
            return [self._sanitize_debug_value(item) for item in value]
        return value

    def _truncate_debug_text(self, text: str) -> str:
        if len(text) <= MAX_DEBUG_TEXT_LENGTH:
            return text
        truncated_length = MAX_DEBUG_TEXT_LENGTH - len(DEBUG_TEXT_TRUNCATION_SUFFIX)
        return f"{text[:truncated_length]}{DEBUG_TEXT_TRUNCATION_SUFFIX}"
