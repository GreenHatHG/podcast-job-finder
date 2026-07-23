from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Final, NoReturn, Sequence

from episode_processing_pipeline import (
    EXPECTED_EPISODE_ERRORS,
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
    AUDIO_REQUIRES_CHAT_COMPLETIONS_ERROR,
    CHAT_COMPLETIONS_API_STYLE,
    OpenAiCompatibleConfigError,
    OpenAiCompatibleLlmClient,
    load_openai_compatible_config_from_env,
    load_llm_retry_config_from_env,
)
from pid_audio_transcription import (
    PidAudioTranscriptionError,
    PidAudioTranscriptionRuntime,
    run_pid_audio_transcription,
    save_pid_audio_transcription_report,
)
from pid_company_report import PidCompanyReportData, save_pid_company_reports
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
        f"      {PROGRAM_NAME} pid --pid <pid> [--all] [--source page|audio]",
    ]
)
COMPANY_BLACKLIST_ENV_NAME: Final = "COMPANY_BLACKLIST"
COMPANY_BLACKLIST_SEPARATOR_PATTERN = re.compile(r"[\n,，]+")

logger = logging.getLogger(__name__)
SEND_CODE_COMMAND: Final = "send-code"
LOGIN_COMMAND: Final = "login"
PID_COMMAND: Final = "pid"
PAGE_SOURCE: Final = "page"
AUDIO_SOURCE: Final = "audio"
SUPPORTED_PID_SOURCES: Final = (PAGE_SOURCE, AUDIO_SOURCE)
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
XYZ_SERVICE_URL_TEXT: Final = DEFAULT_XYZ_BASE_URL
DEFAULT_EPISODE_ORDER: Final = "desc"
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
        *EXPECTED_EPISODE_ERRORS,
        PidAudioTranscriptionError,
        XiaoyuzhouAuthStoreError,
        XyzClientError,
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
    pid_parser.add_argument(
        "--source",
        choices=SUPPORTED_PID_SOURCES,
        default=PAGE_SOURCE,
    )
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
    if parsed_args.source == AUDIO_SOURCE:
        return _run_pid_audio_mode(parsed_args.pid, work_items)
    return _run_pid_page_mode(parsed_args.pid, work_items)


def _run_pid_page_mode(pid: str, work_items: Sequence[EpisodeWorkItem]) -> int:
    extraction_runtime = _load_extraction_runtime()
    pipeline_rate_config = load_pipeline_rate_config_from_env()
    pipeline_result = run_pid_episode_pipeline(
        work_items=work_items,
        runtime=extraction_runtime,
        checkpoint_store=LlmCheckpointStore(),
        rate_config=pipeline_rate_config,
    )
    report_data = PidCompanyReportData(
        pid=pid,
        model=extraction_runtime.model,
        base_url=extraction_runtime.base_url,
        total=len(work_items),
        success=pipeline_result.success_count,
        failed=pipeline_result.fail_count,
        episodes=pipeline_result.episode_results,
    )

    output_path, summary_path = save_pid_company_reports(report_data)
    logger.info("结果已保存到 %s", output_path)
    logger.info("公司汇总已保存到 %s", summary_path)
    return 1 if pipeline_result.fail_count > 0 else 0


def _run_pid_audio_mode(pid: str, work_items: Sequence[EpisodeWorkItem]) -> int:
    runtime = _load_audio_transcription_runtime()
    result = run_pid_audio_transcription(
        pid=pid,
        work_items=work_items,
        runtime=runtime,
    )
    report_path = save_pid_audio_transcription_report(
        pid=pid,
        runtime=runtime,
        result=result,
        output_dir=Path(OUTPUT_DIR),
    )
    logger.info("音频转写批次报告已保存到 %s", report_path)
    return 1 if result.fail_count > 0 else 0


def _run_single_episode_mode(episode_url: str) -> int:
    logger.info("处理单个节目：%s", episode_url)
    extraction_runtime = _load_extraction_runtime()
    completed_extraction = run_episode_company_extraction(
        work_item=EpisodeWorkItem(episode_url=episode_url),
        runtime=extraction_runtime,
        checkpoint_store=LlmCheckpointStore(),
    )
    _print_json(completed_extraction.extraction_result.to_dict(), indent=2)
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


def _load_audio_transcription_runtime() -> PidAudioTranscriptionRuntime:
    llm_config = load_openai_compatible_config_from_env()
    if llm_config.api_style != CHAT_COMPLETIONS_API_STYLE:
        raise OpenAiCompatibleConfigError(AUDIO_REQUIRES_CHAT_COMPLETIONS_ERROR)
    return PidAudioTranscriptionRuntime(
        llm_client=OpenAiCompatibleLlmClient(llm_config),
        retry_config=load_llm_retry_config_from_env(),
        llm_config=llm_config,
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
