from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final, Sequence

from company_extraction import CompanyExtractionError, extract_companies_from_episode
from extract_xiaoyuzhou_episode import EpisodeParseError, parse_episode_url
from openai_compatible_llm import (
    EmptyLlmResponseError,
    LlmRetryConfig,
    OpenAiCompatibleConfigError,
    OpenAiCompatibleLlmClient,
    OpenAiCompatibleLlmError,
    load_openai_compatible_config_from_env,
    load_llm_retry_config_from_env,
)
from xiaoyuzhou_auth_store import (
    XiaoyuzhouAuthSession,
    XiaoyuzhouAuthStoreError,
    build_auth_session,
    load_auth_session,
    save_auth_session,
    update_auth_session_tokens,
)
from xiaoyuzhou_xyz_client import (
    DEFAULT_AREA_CODE,
    DEFAULT_XYZ_BASE_URL,
    EpisodeLoadMoreKey,
    PodcastEpisodeSummary,
    XyzClient,
    XyzClientError,
    XyzUnauthorizedError,
)


PROGRAM_NAME: Final = "python extract_episode_companies.py"
COMMAND_USAGE_TEXT: Final = "\n".join(
    [
        f"用法：{PROGRAM_NAME} <episode_url>",
        f"      {PROGRAM_NAME} send-code --mobile <手机号> [--area-code +86]",
        f"      {PROGRAM_NAME} login --mobile <手机号> --code <验证码> [--area-code +86]",
        f"      {PROGRAM_NAME} pid --pid <pid> [--all]",
    ]
)
LOG_LEVEL_ENV: Final = "LOG_LEVEL"
DEFAULT_LOG_LEVEL_NAME: Final = "INFO"
COMPANY_BLACKLIST_ENV_NAME: Final = "COMPANY_BLACKLIST"
COMPANY_BLACKLIST_SEPARATOR_PATTERN = re.compile(r"[\n,，]+")

logger = logging.getLogger(__name__)
SEND_CODE_COMMAND: Final = "send-code"
LOGIN_COMMAND: Final = "login"
PID_COMMAND: Final = "pid"
HELP_FLAGS: Final = frozenset({"-h", "--help"})
SUPPORTED_COMMANDS: Final = frozenset(
    {
        SEND_CODE_COMMAND,
        LOGIN_COMMAND,
        PID_COMMAND,
    }
)
EPISODE_URL_TEMPLATE: Final = "https://www.xiaoyuzhoufm.com/episode/{eid}"
OUTPUT_DIR: Final = "output"
OUTPUT_FILE_TEMPLATE: Final = "result_{pid}_{timestamp}.json"
SUMMARY_FILE_TEMPLATE: Final = "summary_{pid}_{timestamp}.json"
OUTPUT_STATUS_SUCCESS: Final = "success"
OUTPUT_STATUS_ERROR: Final = "error"
XYZ_SERVICE_URL_TEXT: Final = DEFAULT_XYZ_BASE_URL
DEFAULT_EPISODE_ORDER: Final = "desc"
SUCCESS_STATUS_TEXT: Final = "ok"


class CliUsageError(ValueError):
    """Raised when the command line arguments are invalid."""


class _CliArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CliUsageError(COMMAND_USAGE_TEXT)


@dataclass(slots=True, frozen=True)
class ExtractionRuntime:
    llm_client: OpenAiCompatibleLlmClient
    retry_config: LlmRetryConfig
    company_blacklist: tuple[str, ...]
    model: str
    base_url: str | None


def main() -> int:
    raw_args = sys.argv[1:]
    _configure_logging()
    if not raw_args:
        print(COMMAND_USAGE_TEXT, file=sys.stderr)
        return 1
    if raw_args[0] in HELP_FLAGS:
        print(COMMAND_USAGE_TEXT)
        return 0

    try:
        if raw_args[0] in SUPPORTED_COMMANDS:
            return _run_command(raw_args)
        if len(raw_args) != 1:
            raise CliUsageError(COMMAND_USAGE_TEXT)
        return _run_single_episode_mode(raw_args[0])
    except (
        CliUsageError,
        CompanyExtractionError,
        EmptyLlmResponseError,
        EpisodeParseError,
        OpenAiCompatibleConfigError,
        OpenAiCompatibleLlmError,
        XiaoyuzhouAuthStoreError,
        XyzClientError,
        ValueError,
    ) as error:
        print(str(error), file=sys.stderr)
        return 1


def _configure_logging() -> None:
    logging.basicConfig(
        level=_resolve_log_level(),
        format="%(levelname)s:%(name)s:%(message)s",
    )


def _resolve_log_level() -> int:
    raw_log_level = os.getenv(LOG_LEVEL_ENV, DEFAULT_LOG_LEVEL_NAME)
    normalized_log_level = raw_log_level.strip().upper()
    if not normalized_log_level:
        normalized_log_level = DEFAULT_LOG_LEVEL_NAME

    resolved_log_level = getattr(logging, normalized_log_level, None)
    if not isinstance(resolved_log_level, int):
        return logging.WARNING
    return resolved_log_level


def _load_company_blacklist() -> tuple[str, ...]:
    raw_company_blacklist = os.getenv(COMPANY_BLACKLIST_ENV_NAME, "")
    normalized_blacklist_text = raw_company_blacklist.strip()
    if not normalized_blacklist_text:
        return ()

    return tuple(
        company_name.strip()
        for company_name in COMPANY_BLACKLIST_SEPARATOR_PATTERN.split(
            normalized_blacklist_text
        )
        if company_name.strip()
    )


def _run_command(raw_args: Sequence[str]) -> int:
    parser = _build_command_parser()
    parsed_args = parser.parse_args(list(raw_args))
    xyz_client = XyzClient()
    if parsed_args.command == SEND_CODE_COMMAND:
        return _run_send_code_mode(parsed_args, xyz_client)
    if parsed_args.command == LOGIN_COMMAND:
        return _run_login_mode(parsed_args, xyz_client)
    if parsed_args.command == PID_COMMAND:
        return _run_pid_mode(parsed_args, xyz_client)
    raise CliUsageError(COMMAND_USAGE_TEXT)


def _build_command_parser() -> argparse.ArgumentParser:
    parser = _CliArgumentParser(add_help=True, prog=PROGRAM_NAME)
    subparsers = parser.add_subparsers(dest="command", required=True)

    send_code_parser = subparsers.add_parser(SEND_CODE_COMMAND)
    send_code_parser.add_argument("--mobile", required=True)
    send_code_parser.add_argument("--area-code", default=DEFAULT_AREA_CODE)

    login_parser = subparsers.add_parser(LOGIN_COMMAND)
    login_parser.add_argument("--mobile", required=True)
    login_parser.add_argument("--code", required=True)
    login_parser.add_argument("--area-code", default=DEFAULT_AREA_CODE)

    pid_parser = subparsers.add_parser(PID_COMMAND)
    pid_parser.add_argument("--pid", required=True)
    pid_parser.add_argument("--all", action="store_true", dest="fetch_all")
    return parser


def _run_send_code_mode(parsed_args: argparse.Namespace, xyz_client: XyzClient) -> int:
    xyz_client.send_code(
        mobile_phone_number=parsed_args.mobile,
        area_code=parsed_args.area_code,
    )
    _print_json(
        {
            "status": SUCCESS_STATUS_TEXT,
            "mobile_phone_number": parsed_args.mobile,
            "area_code": parsed_args.area_code,
            "xyz_service_url": XYZ_SERVICE_URL_TEXT,
        },
        indent=2,
    )
    return 0


def _run_login_mode(parsed_args: argparse.Namespace, xyz_client: XyzClient) -> int:
    login_result = xyz_client.login(
        mobile_phone_number=parsed_args.mobile,
        verify_code=parsed_args.code,
        area_code=parsed_args.area_code,
    )
    auth_session = build_auth_session(
        mobile_phone_number=parsed_args.mobile,
        area_code=parsed_args.area_code,
        uid=login_result.uid,
        access_token=login_result.access_token,
        refresh_token=login_result.refresh_token,
    )
    save_auth_session(auth_session)
    _print_json(
        {
            "status": SUCCESS_STATUS_TEXT,
            "uid": login_result.uid,
            "mobile_phone_number": parsed_args.mobile,
            "area_code": parsed_args.area_code,
            "xyz_service_url": XYZ_SERVICE_URL_TEXT,
        },
        indent=2,
    )
    return 0


def _run_pid_mode(parsed_args: argparse.Namespace, xyz_client: XyzClient) -> int:
    auth_session = load_auth_session()
    extraction_runtime = _load_extraction_runtime()
    logger.info("开始抓取播客节目列表，pid=%s", parsed_args.pid)
    episodes = _list_podcast_episodes(
        xyz_client=xyz_client,
        auth_session=auth_session,
        pid=parsed_args.pid,
        fetch_all=parsed_args.fetch_all,
    )
    logger.info("抓取到 %d 个节目", len(episodes))

    episode_results: list[dict] = []
    success_count = 0
    fail_count = 0
    for episode_index, episode_summary in enumerate(episodes, start=1):
        episode_url = _build_episode_url(episode_summary.eid)
        logger.info(
            "正在处理第 %d/%d 个节目：%s",
            episode_index,
            len(episodes),
            episode_summary.title,
        )
        try:
            extraction_result = _extract_companies_by_episode_url(
                episode_url,
                extraction_runtime,
            )
            logger.info(
                "节目处理完成：提取到 %d 家公司，过滤 %d 家",
                len(extraction_result.companies),
                extraction_result.filtered_count,
            )
            episode_results.append(
                {
                    "status": OUTPUT_STATUS_SUCCESS,
                    "eid": episode_summary.eid,
                    "title": episode_summary.title,
                    "pub_date": episode_summary.pub_date,
                    "episode_url": episode_url,
                    "companies": [
                        company.to_dict() for company in extraction_result.companies
                    ],
                    "filtered_count": extraction_result.filtered_count,
                }
            )
            success_count += 1
        except (
            CompanyExtractionError,
            EmptyLlmResponseError,
            EpisodeParseError,
            OpenAiCompatibleConfigError,
            OpenAiCompatibleLlmError,
            ValueError,
        ) as error:
            logger.info("节目处理失败：%s", error)
            episode_results.append(
                {
                    "status": OUTPUT_STATUS_ERROR,
                    "eid": episode_summary.eid,
                    "title": episode_summary.title,
                    "pub_date": episode_summary.pub_date,
                    "episode_url": episode_url,
                    "error": str(error),
                }
            )
            fail_count += 1

    output_path = _save_result_file(
        pid=parsed_args.pid,
        model=extraction_runtime.model,
        base_url=extraction_runtime.base_url,
        total=len(episodes),
        success=success_count,
        failed=fail_count,
        episodes=episode_results,
    )
    logger.info("结果已保存到 %s", output_path)
    summary_path = _save_summary_file(
        pid=parsed_args.pid,
        model=extraction_runtime.model,
        base_url=extraction_runtime.base_url,
        total=len(episodes),
        success=success_count,
        failed=fail_count,
        episodes=episode_results,
    )
    logger.info("公司汇总已保存到 %s", summary_path)
    return 1 if fail_count > 0 else 0


def _run_single_episode_mode(episode_url: str) -> int:
    logger.info("处理单个节目：%s", episode_url)
    extraction_runtime = _load_extraction_runtime()
    extraction_result = _extract_companies_by_episode_url(
        episode_url,
        extraction_runtime,
    )
    _print_json(extraction_result.to_dict(), indent=2)
    return 0


def _load_extraction_runtime() -> ExtractionRuntime:
    llm_config = load_openai_compatible_config_from_env()
    retry_config = load_llm_retry_config_from_env()
    company_blacklist = _load_company_blacklist()
    llm_client = OpenAiCompatibleLlmClient(llm_config)
    return ExtractionRuntime(
        llm_client=llm_client,
        retry_config=retry_config,
        company_blacklist=company_blacklist,
        model=llm_config.model,
        base_url=llm_config.base_url,
    )


def _extract_companies_by_episode_url(
    episode_url: str,
    extraction_runtime: ExtractionRuntime,
):
    logger.info("抓取节目页面：%s", episode_url)
    episode = parse_episode_url(episode_url)
    return extract_companies_from_episode(
        episode,
        extraction_runtime.llm_client,
        company_blacklist=extraction_runtime.company_blacklist,
        retry_config=extraction_runtime.retry_config,
    )


def _list_podcast_episodes(
    *,
    xyz_client: XyzClient,
    auth_session: XiaoyuzhouAuthSession,
    pid: str,
    fetch_all: bool,
) -> tuple[PodcastEpisodeSummary, ...]:
    episodes: list[PodcastEpisodeSummary] = []
    current_load_more_key: EpisodeLoadMoreKey | None = None
    current_auth_session = auth_session
    while True:
        page, current_auth_session = _fetch_episode_page_with_refresh(
            xyz_client=xyz_client,
            auth_session=current_auth_session,
            pid=pid,
            load_more_key=current_load_more_key,
        )
        episodes.extend(page.episodes)
        if not fetch_all or page.load_more_key is None:
            return tuple(episodes)
        current_load_more_key = page.load_more_key


def _fetch_episode_page_with_refresh(
    *,
    xyz_client: XyzClient,
    auth_session: XiaoyuzhouAuthSession,
    pid: str,
    load_more_key: EpisodeLoadMoreKey | None,
) -> tuple[EpisodeListPage, XiaoyuzhouAuthSession]:
    try:
        page = xyz_client.list_podcast_episodes(
            pid=pid,
            access_token=auth_session.access_token,
            load_more_key=load_more_key,
            order=DEFAULT_EPISODE_ORDER,
        )
        return page, auth_session
    except XyzUnauthorizedError:
        refreshed_session = _refresh_auth_session(
            xyz_client=xyz_client,
            auth_session=auth_session,
        )
        page = xyz_client.list_podcast_episodes(
            pid=pid,
            access_token=refreshed_session.access_token,
            load_more_key=load_more_key,
            order=DEFAULT_EPISODE_ORDER,
        )
        return page, refreshed_session


def _refresh_auth_session(
    *,
    xyz_client: XyzClient,
    auth_session: XiaoyuzhouAuthSession,
) -> XiaoyuzhouAuthSession:
    refreshed_tokens = xyz_client.refresh_token(
        access_token=auth_session.access_token,
        refresh_token=auth_session.refresh_token,
    )
    refreshed_session = update_auth_session_tokens(
        auth_session,
        access_token=refreshed_tokens.access_token,
        refresh_token=refreshed_tokens.refresh_token,
    )
    save_auth_session(refreshed_session)
    return refreshed_session


def _build_episode_url(eid: str) -> str:
    return EPISODE_URL_TEMPLATE.format(eid=eid)


def _build_output_path(template: str, pid: str, timestamp_label: str) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(
        OUTPUT_DIR, template.format(pid=pid, timestamp=timestamp_label)
    )


def _save_summary_file(
    *,
    pid: str,
    model: str,
    base_url: str | None,
    total: int,
    success: int,
    failed: int,
    episodes: list[dict],
) -> str:
    now = datetime.now(tz=timezone.utc)
    timestamp_label = now.strftime("%Y%m%d_%H%M%S")
    output_path = _build_output_path(SUMMARY_FILE_TEMPLATE, pid, timestamp_label)
    companies = _aggregate_companies(episodes)
    report = {
        "pid": pid,
        "model": model,
        "base_url": base_url,
        "created_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_episodes": total,
        "success_episodes": success,
        "failed_episodes": failed,
        "unique_company_count": len(companies),
        "companies": companies,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return output_path


def _aggregate_companies(episodes: list[dict]) -> list[dict]:
    grouped: dict[str, dict] = {}
    for episode in episodes:
        if episode.get("status") != OUTPUT_STATUS_SUCCESS:
            continue
        episode_ref = {
            "eid": episode.get("eid"),
            "title": episode.get("title"),
            "pub_date": episode.get("pub_date"),
            "episode_url": episode.get("episode_url"),
        }
        for company in episode.get("companies", ()):
            raw_name = company.get("name", "")
            normalized_name = raw_name.strip()
            if not normalized_name:
                continue
            entry = grouped.setdefault(
                normalized_name,
                {"name": normalized_name, "occurrence_count": 0, "episodes": []},
            )
            entry["occurrence_count"] += 1
            entry["episodes"].append(
                {**episode_ref, "evidence": company.get("evidence", "")}
            )

    return sorted(
        grouped.values(),
        key=lambda item: (-item["occurrence_count"], item["name"]),
    )


def _save_result_file(
    *,
    pid: str,
    model: str,
    base_url: str | None,
    total: int,
    success: int,
    failed: int,
    episodes: list[dict],
) -> str:
    now = datetime.now(tz=timezone.utc)
    timestamp_label = now.strftime("%Y%m%d_%H%M%S")
    output_path = _build_output_path(OUTPUT_FILE_TEMPLATE, pid, timestamp_label)
    report = {
        "pid": pid,
        "model": model,
        "base_url": base_url,
        "created_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total": total,
        "success": success,
        "failed": failed,
        "episodes": episodes,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return output_path


def _print_json(payload: object, *, indent: int | None = None) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=indent))


if __name__ == "__main__":
    raise SystemExit(main())
