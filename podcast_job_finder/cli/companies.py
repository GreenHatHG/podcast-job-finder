from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Final, NoReturn, Sequence

from podcast_job_finder.companies.checkpoint import LlmCheckpointStore
from podcast_job_finder.companies.episode_runner import (
    EpisodeWorkItem,
    run_episode_company_extraction,
)
from podcast_job_finder.companies.models import CompanyExtractionError
from podcast_job_finder.companies.pipeline import run_pid_episode_pipeline
from podcast_job_finder.companies.rate_limit import (
    load_pipeline_rate_config_from_env,
)
from podcast_job_finder.companies.reporting import PidReportData, save_pid_reports
from podcast_job_finder.companies.runtime import load_extraction_runtime_from_env
from podcast_job_finder.llm import (
    EmptyLlmResponseError,
    OpenAiCompatibleConfigError,
    OpenAiCompatibleLlmError,
)
from podcast_job_finder.logging import configure_logging
from podcast_job_finder.xiaoyuzhou.episode_client import build_episode_url
from podcast_job_finder.xiaoyuzhou.episode_parser import EpisodeParseError
from podcast_job_finder.xiaoyuzhou.xyz.auth_store import (
    XiaoyuzhouAuthStoreError,
    build_auth_session,
    load_auth_session,
    save_auth_session,
)
from podcast_job_finder.xiaoyuzhou.xyz.client import (
    DEFAULT_AREA_CODE,
    DEFAULT_XYZ_BASE_URL,
    XyzClient,
    XyzClientError,
)
from podcast_job_finder.xiaoyuzhou.xyz.podcast_service import list_podcast_episodes


PROGRAM_NAME: Final = "podcast-find-jobs"
COMMAND_USAGE_TEXT: Final = "\n".join(
    [
        f"用法：{PROGRAM_NAME} <episode_url>",
        f"      {PROGRAM_NAME} send-code --mobile <手机号> [--area-code +86]",
        f"      {PROGRAM_NAME} login --mobile <手机号> --code <验证码> [--area-code +86]",
        f"      {PROGRAM_NAME} pid --pid <pid> [--all]",
    ]
)
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
XYZ_SERVICE_URL_TEXT: Final = DEFAULT_XYZ_BASE_URL
SUCCESS_STATUS_TEXT: Final = "ok"


class CliUsageError(ValueError):
    """Raised when the command line arguments are invalid."""


class _CliArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise CliUsageError(COMMAND_USAGE_TEXT)


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
    extraction_runtime = load_extraction_runtime_from_env()
    pipeline_rate_config = load_pipeline_rate_config_from_env()
    checkpoint_store = LlmCheckpointStore()
    logger.info("开始抓取播客节目列表，pid=%s", parsed_args.pid)
    episodes = list_podcast_episodes(
        xyz_client=xyz_client,
        auth_session=auth_session,
        pid=parsed_args.pid,
        fetch_all=parsed_args.fetch_all,
    )
    logger.info("抓取到 %d 个节目", len(episodes))
    work_items = [
        EpisodeWorkItem(
            episode_url=build_episode_url(episode_summary.eid),
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
    report_data = PidReportData(
        pid=parsed_args.pid,
        model=extraction_runtime.model,
        base_url=extraction_runtime.base_url,
        total=len(episodes),
        success=pipeline_result.success_count,
        failed=pipeline_result.fail_count,
        episodes=pipeline_result.episode_results,
    )

    output_path, summary_path = save_pid_reports(report_data)
    logger.info("结果已保存到 %s", output_path)
    logger.info("公司汇总已保存到 %s", summary_path)
    return 1 if pipeline_result.fail_count > 0 else 0


def _run_single_episode_mode(episode_url: str) -> int:
    logger.info("处理单个节目：%s", episode_url)
    extraction_runtime = load_extraction_runtime_from_env()
    extraction_outcome = run_episode_company_extraction(
        work_item=EpisodeWorkItem(episode_url=episode_url),
        runtime=extraction_runtime,
        checkpoint_store=LlmCheckpointStore(),
    )
    _print_json(extraction_outcome.extraction_result.to_dict(), indent=2)
    return 0


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


def _print_json(payload: object, *, indent: int | None = None) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=indent))


if __name__ == "__main__":
    raise SystemExit(main())
