from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Final, NoReturn, Sequence

from podcast_job_finder.transcription.batch import (
    delete_completed_episode_audio_files,
    load_existing_batch_transcription_result,
    run_batch_audio_transcription,
    save_batch_audio_transcription_report,
)
from podcast_job_finder.output_paths import REPORT_OUTPUT_DIR
from podcast_job_finder.transcription.pipeline_results import (
    BatchAudioTranscriptionResult,
)
from podcast_job_finder.transcription.schedule import (
    DEFAULT_AUDIO_PROCESSING_MODE,
    SUPPORTED_AUDIO_PROCESSING_MODES,
    AudioProcessingMode,
)
from podcast_job_finder.transcription.runtime import (
    load_audio_transcription_runtime_from_env,
)
from podcast_job_finder.companies.audio_pipeline import (
    run_batch_audio_company_extraction,
)
from podcast_job_finder.companies.checkpoint import LlmCheckpointStore
from podcast_job_finder.companies.episode_runner import (
    run_episode_company_extraction,
)
from podcast_job_finder.episode.models import EpisodeWorkItem
from podcast_job_finder.companies.pipeline import (
    run_batch_episode_pipeline,
)
from podcast_job_finder.companies.pipeline_results import BatchEpisodePipelineResult
from podcast_job_finder.companies.rate_limit import (
    load_episode_page_fetch_rate_from_env,
)
from podcast_job_finder.companies.reporting import FeedReportData, save_feed_reports
from podcast_job_finder.companies.runtime import (
    EpisodeExtractionRuntime,
    load_audio_extraction_runtime_from_env,
    load_page_extraction_runtime_from_env,
)
from podcast_job_finder.logging import configure_logging
from podcast_job_finder.podcast_catalog import get_podcast_feed_url
from podcast_job_finder.rss.feed import RssFeed, fetch_rss_feed
from podcast_job_finder.episode import build_episode_url
from podcast_job_finder.errors import PodcastJobFinderError


PROGRAM_NAME: Final = "podcast-find-jobs"
PAGE_SOURCE: Final = "page"
AUDIO_SOURCE: Final = "audio"
SUPPORTED_FEED_SOURCES: Final = (PAGE_SOURCE, AUDIO_SOURCE)
HELP_FLAGS: Final = frozenset({"-h", "--help"})
COMMAND_USAGE_TEXT: Final = "\n".join(
    [
        f"用法：{PROGRAM_NAME} <episode_url>",
        (
            f"      {PROGRAM_NAME} (--podcast <播客名> | --feed-url <RSS地址>) "
            "[--max-episodes <正整数>] [--source page|audio] "
            "[--transcribe-only|--extract-only] "
            "[--audio-processing-mode sequential|download-first] [--resume]"
            " [--delete-audio]"
        ),
    ]
)
TRANSCRIBE_ONLY_SOURCE_ERROR: Final = (
    "--transcribe-only 只能与 --source audio 一起使用。"
)
EXTRACT_ONLY_SOURCE_ERROR: Final = "--extract-only 只能与 --source audio 一起使用。"
EXCLUSIVE_AUDIO_MODE_ERROR: Final = "--transcribe-only 和 --extract-only 不能同时使用。"
RESUME_SOURCE_ERROR: Final = "--resume 只能与 --source audio 一起使用。"
DELETE_AUDIO_SOURCE_ERROR: Final = "--delete-audio 只能与 --source audio 一起使用。"

logger = logging.getLogger(__name__)


class CliUsageError(PodcastJobFinderError, ValueError):
    """命令行参数无效；命令入口打印用法后停止。"""


class _CliArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        del message
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
        if raw_args[0].startswith("-"):
            return _run_feed_command(raw_args)
        if len(raw_args) != 1:
            raise CliUsageError(COMMAND_USAGE_TEXT)
        return _run_single_episode_mode(raw_args[0])
    except PodcastJobFinderError as error:
        print(str(error), file=sys.stderr)
        return 1


def _run_feed_command(raw_args: Sequence[str]) -> int:
    parsed_args = _build_feed_parser().parse_args(list(raw_args))
    if parsed_args.transcribe_only and parsed_args.extract_only:
        raise CliUsageError(EXCLUSIVE_AUDIO_MODE_ERROR)
    if parsed_args.transcribe_only and parsed_args.source != AUDIO_SOURCE:
        raise CliUsageError(TRANSCRIBE_ONLY_SOURCE_ERROR)
    if parsed_args.extract_only and parsed_args.source != AUDIO_SOURCE:
        raise CliUsageError(EXTRACT_ONLY_SOURCE_ERROR)
    if parsed_args.resume and parsed_args.source != AUDIO_SOURCE:
        raise CliUsageError(RESUME_SOURCE_ERROR)
    if parsed_args.delete_audio and parsed_args.source != AUDIO_SOURCE:
        raise CliUsageError(DELETE_AUDIO_SOURCE_ERROR)
    feed_url = (
        get_podcast_feed_url(parsed_args.podcast)
        if parsed_args.podcast is not None
        else parsed_args.feed_url.strip()
    )
    logger.info("从 RSS 获取播客节目列表：feed=%s", feed_url)
    feed = fetch_rss_feed(feed_url)
    episodes = feed.episodes
    if parsed_args.max_episodes is not None:
        episodes = episodes[: parsed_args.max_episodes]
    work_items = [
        EpisodeWorkItem(
            episode_url=build_episode_url(episode.episode_id),
            eid=episode.episode_id,
            podcast_title=feed.title,
            title=episode.title,
            pub_date=episode.published_at,
            audio_url=episode.audio_url,
        )
        for episode in episodes
    ]
    logger.info(
        "RSS 节目列表读取完成：podcast=%s episodes=%d", feed.title, len(work_items)
    )

    if parsed_args.source == AUDIO_SOURCE:
        if parsed_args.extract_only:
            exit_code = _run_feed_audio_extraction_only(
                feed,
                work_items,
                resume=parsed_args.resume,
            )
        else:
            exit_code = _run_feed_audio_mode(
                feed,
                work_items,
                transcribe_only=parsed_args.transcribe_only,
                processing_mode=parsed_args.audio_processing_mode,
                resume=parsed_args.resume,
            )
        if parsed_args.delete_audio:
            delete_completed_episode_audio_files(work_items)
        return exit_code
    return _run_feed_page_mode(feed, work_items)


def _build_feed_parser() -> argparse.ArgumentParser:
    parser = _CliArgumentParser(add_help=True, prog=PROGRAM_NAME)
    feed_input = parser.add_mutually_exclusive_group(required=True)
    feed_input.add_argument("--podcast")
    feed_input.add_argument("--feed-url")
    parser.add_argument("--max-episodes", type=_parse_positive_int)
    parser.add_argument(
        "--source",
        choices=SUPPORTED_FEED_SOURCES,
        default=PAGE_SOURCE,
    )
    parser.add_argument("--transcribe-only", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument(
        "--audio-processing-mode",
        choices=SUPPORTED_AUDIO_PROCESSING_MODES,
        default=DEFAULT_AUDIO_PROCESSING_MODE,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--delete-audio", action="store_true")
    return parser


def _parse_positive_int(value: str) -> int:
    try:
        parsed_value = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("必须是正整数") from error
    if parsed_value < 1:
        raise argparse.ArgumentTypeError("必须是正整数")
    return parsed_value


def _run_feed_page_mode(feed: RssFeed, work_items: Sequence[EpisodeWorkItem]) -> int:
    extraction_runtime = load_page_extraction_runtime_from_env()
    pipeline_result = run_batch_episode_pipeline(
        work_items=work_items,
        runtime=extraction_runtime,
        checkpoint_store=LlmCheckpointStore(),
        page_fetch_rate_per_minute=load_episode_page_fetch_rate_from_env(),
    )
    output_path, summary_path = _save_company_extraction_reports(
        feed,
        runtime=extraction_runtime,
        total=len(work_items),
        pipeline_result=pipeline_result,
    )
    logger.info("结果已保存到 %s", output_path)
    logger.info("公司汇总已保存到 %s", summary_path)
    return 1 if pipeline_result.fail_count > 0 else 0


def _run_feed_audio_mode(
    feed: RssFeed,
    work_items: Sequence[EpisodeWorkItem],
    *,
    transcribe_only: bool,
    processing_mode: AudioProcessingMode,
    resume: bool,
) -> int:
    transcription_runtime = load_audio_transcription_runtime_from_env()
    try:
        transcription_result = run_batch_audio_transcription(
            work_items=work_items,
            runtime=transcription_runtime,
            processing_mode=processing_mode,
            resume=resume,
        )
        report_path = save_batch_audio_transcription_report(
            feed_id=feed.feed_id,
            podcast_title=feed.title,
            runtime=transcription_runtime,
            result=transcription_result,
            output_dir=REPORT_OUTPUT_DIR,
        )
    finally:
        transcription_runtime.close()
    logger.info("音频转写批次报告已保存到 %s", report_path)
    if transcribe_only:
        return 1 if transcription_result.fail_count > 0 else 0

    return _run_audio_company_extraction(
        feed,
        transcription_result,
        resume=resume,
    )


def _run_feed_audio_extraction_only(
    feed: RssFeed,
    work_items: Sequence[EpisodeWorkItem],
    *,
    resume: bool,
) -> int:
    transcription_result, skipped_count = load_existing_batch_transcription_result(
        work_items
    )
    logger.info(
        "已有音频转写扫描完成：可提取=%d 跳过=%d",
        transcription_result.success_count,
        skipped_count,
    )
    return _run_audio_company_extraction(
        feed,
        transcription_result,
        resume=resume,
    )


def _run_audio_company_extraction(
    feed: RssFeed,
    transcription_result: BatchAudioTranscriptionResult,
    *,
    resume: bool,
) -> int:
    extraction_runtime = load_audio_extraction_runtime_from_env()
    extraction_result = run_batch_audio_company_extraction(
        transcription_result=transcription_result,
        runtime=extraction_runtime,
        resume=resume,
    )
    output_path, summary_path = _save_company_extraction_reports(
        feed,
        runtime=extraction_runtime,
        total=len(transcription_result.episode_results),
        pipeline_result=extraction_result,
    )
    logger.info("音频公司提取结果已保存到 %s", output_path)
    logger.info("音频公司汇总已保存到 %s", summary_path)
    return 1 if extraction_result.fail_count > 0 else 0


def _save_company_extraction_reports(
    feed: RssFeed,
    *,
    runtime: EpisodeExtractionRuntime,
    total: int,
    pipeline_result: BatchEpisodePipelineResult,
) -> tuple[str, str]:
    return save_feed_reports(
        FeedReportData(
            feed_id=feed.feed_id,
            podcast_title=feed.title,
            model=runtime.llm.model,
            base_url=runtime.llm.base_url,
            total=total,
            success=pipeline_result.success_count,
            failed=pipeline_result.fail_count,
            episodes=pipeline_result.episode_results,
        )
    )


def _run_single_episode_mode(episode_url: str) -> int:
    logger.info("处理单个节目：%s", episode_url)
    extraction_runtime = load_page_extraction_runtime_from_env()
    extraction_outcome = run_episode_company_extraction(
        work_item=EpisodeWorkItem(episode_url=episode_url),
        runtime=extraction_runtime,
        checkpoint_store=LlmCheckpointStore(),
    )
    print(
        json.dumps(
            extraction_outcome.extraction_result.to_dict(),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
