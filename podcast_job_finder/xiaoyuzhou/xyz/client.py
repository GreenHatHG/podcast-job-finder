from __future__ import annotations

from http import HTTPStatus
import logging
from typing import Any, Final, NoReturn

import requests

from podcast_job_finder.http.user_agents import DEFAULT_BROWSER_USER_AGENT
from podcast_job_finder.xiaoyuzhou.xyz import debug
from podcast_job_finder.xiaoyuzhou.xyz.models import (
    EpisodeListPage,
    EpisodeLoadMoreKey,
    LoginResult,
    PodcastEpisodeSummary,
    RefreshedTokens,
)


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
DEBUG_UNEXPECTED_RESPONSE_TEMPLATE: Final = (
    "xyz 响应字段异常 path=%s field=%s payload=%s"
)


logger = logging.getLogger(__name__)


class XyzClientError(RuntimeError):
    """Raised when the local xyz service cannot satisfy the request."""


class XyzUnauthorizedError(XyzClientError):
    """Raised when the local xyz service reports an expired auth session."""


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

        debug.log_request(logger, path=path, payload=payload)
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
        if response.status_code == HTTPStatus.UNAUTHORIZED:
            self._raise_http_request_error(
                path=path,
                status_code=response.status_code,
                response_data=response_data,
                error_type=XyzUnauthorizedError,
            )
        if response.status_code >= HTTPStatus.BAD_REQUEST:
            self._raise_http_request_error(
                path=path,
                status_code=response.status_code,
                response_data=response_data,
                error_type=XyzClientError,
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
            response_body = debug.truncate_text(response.text)
            debug.log_parse_failure(
                logger,
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

        debug.log_response(
            logger,
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
        normalized_value = value.strip() if isinstance(value, str) else ""
        if not normalized_value:
            self._raise_unexpected_response(
                path=path,
                field_name=field_name,
                expected_description="非空字符串",
                payload=value,
            )
        return normalized_value

    def _raise_http_request_error(
        self,
        *,
        path: str,
        status_code: int,
        response_data: dict[str, Any],
        error_type: type[XyzClientError],
    ) -> NoReturn:
        raise error_type(
            HTTP_REQUEST_FAILED_TEMPLATE.format(
                path=path,
                status_code=status_code,
                detail=self._extract_error_detail(response_data),
            )
        )

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
            actual_value=debug.format_payload(payload),
        )
        logger.debug(
            DEBUG_UNEXPECTED_RESPONSE_TEMPLATE,
            path,
            field_name,
            debug.format_payload(payload),
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
