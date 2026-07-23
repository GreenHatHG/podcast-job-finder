from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final, NoReturn, Sequence

from company_extraction import CompanyExtractionError
from episode_processing_pipeline import (
    load_pipeline_rate_config_from_env,
    run_pid_episode_pipeline,
)
from episode_company_runner import (
    EpisodeExtractionRuntime,
    EpisodeWorkItem,
    build_runtime_signature,
    run_episode_company_extraction,
)
from logging_config import configure_logging
from llm_checkpoint_store import LlmCheckpointStore
from openai_compatible_llm import (
    EmptyLlmResponseError,
    OpenAiCompatibleConfigError,
    OpenAiCompatibleLlmClient,
    OpenAiCompatibleLlmError,
    load_openai_compatible_config_from_env,
    load_llm_retry_config_from_env,
)
from podcast_job_finder.xiaoyuzhou.episode_page import EpisodeParseError
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
    EpisodeListPage,
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
XYZ_SERVICE_URL_TEXT: Final = DEFAULT_XYZ_BASE_URL
DEFAULT_EPISODE_ORDER: Final = "desc"
SUCCESS_STATUS_TEXT: Final = "ok"


class CliUsageError(ValueError):
    """Raised when the command line arguments are invalid."""


class _CliArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise CliUsageError(COMMAND_USAGE_TEXT)


@dataclass(slots=True, frozen=True)
class _PidReportData:
    pid: str
    model: str
    base_url: str | None
    total: int
    success: int
    failed: int
    episodes: list[dict]


def main() -> int:
    raw_args = sys.argv[1:]
    configure_logging()
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
    _print_xyz_success_payload(
        mobile_phone_number=parsed_args.mobile,
        area_code=parsed_args.area_code,
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
    _print_xyz_success_payload(
        mobile_phone_number=parsed_args.mobile,
        area_code=parsed_args.area_code,
        uid=login_result.uid,
    )
    return 0


def _run_pid_mode(parsed_args: argparse.Namespace, xyz_client: XyzClient) -> int:
    auth_session = load_auth_session()
    extraction_runtime = _load_extraction_runtime()
    pipeline_rate_config = load_pipeline_rate_config_from_env()
    checkpoint_store = LlmCheckpointStore()
    logger.info("开始抓取播客节目列表，pid=%s", parsed_args.pid)
    episodes = _list_podcast_episodes(
        xyz_client=xyz_client,
        auth_session=auth_session,
        pid=parsed_args.pid,
        fetch_all=parsed_args.fetch_all,
    )
    logger.info("抓取到 %d 个节目", len(episodes))
    work_items = [
        EpisodeWorkItem(
            episode_url=_build_episode_url(episode_summary.eid),
            eid=episode_summary.eid,
            title=episode_summary.title,
            pub_date=episode_summary.pub_date,
        )
        for episode_summary in episodes
    ]
    pipeline_result = run_pid_episode_pipeline(
        work_items=work_items,
        runtime=extraction_runtime,
        checkpoint_store=checkpoint_store,
        rate_config=pipeline_rate_config,
    )
    report_data = _PidReportData(
        pid=parsed_args.pid,
        model=extraction_runtime.model,
        base_url=extraction_runtime.base_url,
        total=len(episodes),
        success=pipeline_result.success_count,
        failed=pipeline_result.fail_count,
        episodes=pipeline_result.episode_results,
    )

    output_path = _save_result_file(report_data)
    logger.info("结果已保存到 %s", output_path)
    summary_path = _save_summary_file(report_data)
    logger.info("公司汇总已保存到 %s", summary_path)
    return 1 if pipeline_result.fail_count > 0 else 0


def _run_single_episode_mode(episode_url: str) -> int:
    logger.info("处理单个节目：%s", episode_url)
    extraction_runtime = _load_extraction_runtime()
    extraction_outcome = run_episode_company_extraction(
        work_item=EpisodeWorkItem(episode_url=episode_url),
        runtime=extraction_runtime,
        checkpoint_store=LlmCheckpointStore(),
    )
    _print_json(extraction_outcome.extraction_result.to_dict(), indent=2)
    return 0


def _load_extraction_runtime() -> EpisodeExtractionRuntime:
    llm_config = load_openai_compatible_config_from_env()
    retry_config = load_llm_retry_config_from_env()
    company_blacklist = _load_company_blacklist()
    llm_client = OpenAiCompatibleLlmClient(llm_config)
    return EpisodeExtractionRuntime(
        llm_client=llm_client,
        retry_config=retry_config,
        company_blacklist=company_blacklist,
        model=llm_config.model,
        base_url=llm_config.base_url,
        api_style=llm_config.api_style,
        runtime_signature=build_runtime_signature(
            model=llm_config.model,
            base_url=llm_config.base_url,
            api_style=llm_config.api_style,
            company_blacklist=company_blacklist,
        ),
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
    return os.path.join(OUTPUT_DIR, template.format(pid=pid, timestamp=timestamp_label))


def _save_summary_file(report_data: _PidReportData) -> str:
    output_path, created_at = _build_output_file_details(
        SUMMARY_FILE_TEMPLATE,
        report_data.pid,
    )
    companies = _aggregate_companies(report_data.episodes)
    report = _build_base_report(
        report_data=report_data,
        created_at=created_at,
        total_key="total_episodes",
        success_key="success_episodes",
        failed_key="failed_episodes",
    )
    report["unique_company_count"] = len(companies)
    report["companies"] = companies
    _write_report_json(output_path, report)
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


def _save_result_file(report_data: _PidReportData) -> str:
    output_path, created_at = _build_output_file_details(
        OUTPUT_FILE_TEMPLATE,
        report_data.pid,
    )
    report = _build_base_report(
        report_data=report_data,
        created_at=created_at,
        total_key="total",
        success_key="success",
        failed_key="failed",
    )
    report["episodes"] = report_data.episodes
    _write_report_json(output_path, report)
    return output_path


def _print_xyz_success_payload(
    *,
    mobile_phone_number: str,
    area_code: str,
    uid: str | None = None,
) -> None:
    payload = {
        "status": SUCCESS_STATUS_TEXT,
        "mobile_phone_number": mobile_phone_number,
        "area_code": area_code,
        "xyz_service_url": XYZ_SERVICE_URL_TEXT,
    }
    if uid is not None:
        payload["uid"] = uid
    _print_json(payload, indent=2)


def _build_output_file_details(template: str, pid: str) -> tuple[str, str]:
    now = datetime.now(tz=timezone.utc)
    timestamp_label = now.strftime("%Y%m%d_%H%M%S")
    created_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    return _build_output_path(template, pid, timestamp_label), created_at


def _build_base_report(
    *,
    report_data: _PidReportData,
    created_at: str,
    total_key: str,
    success_key: str,
    failed_key: str,
) -> dict[str, object]:
    return {
        "pid": report_data.pid,
        "model": report_data.model,
        "base_url": report_data.base_url,
        "created_at": created_at,
        total_key: report_data.total,
        success_key: report_data.success,
        failed_key: report_data.failed,
    }


def _write_report_json(path: str, payload: object) -> None:
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")


def _print_json(payload: object, *, indent: int | None = None) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=indent))


if __name__ == "__main__":
    raise SystemExit(main())
