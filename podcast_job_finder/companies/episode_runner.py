from __future__ import annotations

import logging
from dataclasses import dataclass, replace

from podcast_job_finder.companies.extraction import (
    LlmClientProtocol,
    build_company_extraction_input,
    build_company_extraction_prompt,
    get_company_extraction_prompt_template,
    run_company_extraction_from_prompt,
)
from podcast_job_finder.companies.models import (
    CompanyExtractionResult,
)
from podcast_job_finder.companies.checkpoint import (
    STATUS_FAILED,
    STATUS_PREPARED,
    STATUS_SUCCESS,
    LlmCheckpoint,
    LlmCheckpointSavePayload,
    LlmCheckpointStore,
)
from podcast_job_finder.llm import LlmRetryConfig
from podcast_job_finder.xiaoyuzhou.episode_client import (
    extract_episode_id_from_url,
    parse_episode_url,
)
from podcast_job_finder.runtime_signature import build_runtime_signature_hash


logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class EpisodeExtractionRuntime:
    llm_client: LlmClientProtocol
    retry_config: LlmRetryConfig
    company_blacklist: tuple[str, ...]
    model: str
    base_url: str | None
    api_style: str
    runtime_signature: str


@dataclass(slots=True, frozen=True)
class EpisodeWorkItem:
    episode_url: str
    eid: str | None = None
    title: str | None = None
    pub_date: str | None = None

    def resolve_episode_id(self) -> str | None:
        normalized_eid = (self.eid or "").strip()
        if normalized_eid:
            return normalized_eid
        return extract_episode_id_from_url(self.episode_url)

    def to_result_metadata(self, *, eid: str | None = None) -> dict[str, object]:
        return _build_episode_result_metadata(
            episode_url=self.episode_url,
            eid=eid or self.eid,
            title=self.title,
            pub_date=self.pub_date,
        )


@dataclass(slots=True, frozen=True)
class CompletedEpisodeExtraction:
    episode: EpisodeWorkItem
    episode_key: str
    extraction_result: CompanyExtractionResult


@dataclass(slots=True, frozen=True)
class PreparedEpisodeLlmWork:
    episode: EpisodeWorkItem
    episode_key: str
    prompt_text: str

    def to_checkpoint_payload(self, runtime_signature: str) -> LlmCheckpointSavePayload:
        return LlmCheckpointSavePayload(
            episode_key=self.episode_key,
            episode_url=self.episode.episode_url,
            title=self.episode.title,
            pub_date=self.episode.pub_date,
            runtime_signature=runtime_signature,
            prompt_text=self.prompt_text,
        )


@dataclass(slots=True, frozen=True)
class _EpisodeCheckpointContext:
    episode: EpisodeWorkItem
    episode_key: str
    checkpoint: LlmCheckpoint | None

    def build_prepared_work(
        self,
        *,
        title: str | None,
        pub_date: str | None,
        prompt_text: str,
    ) -> PreparedEpisodeLlmWork:
        return PreparedEpisodeLlmWork(
            episode=replace(
                self.episode,
                title=title,
                pub_date=pub_date,
            ),
            episode_key=self.episode_key,
            prompt_text=prompt_text,
        )


def build_runtime_signature(
    *,
    model: str,
    base_url: str | None,
    api_style: str,
    company_blacklist: tuple[str, ...],
) -> str:
    normalized_blacklist = sorted(
        {
            company_name.strip().casefold()
            for company_name in company_blacklist
            if company_name.strip()
        }
    )
    signature_payload = {
        "model": model,
        "base_url": base_url,
        "api_style": api_style,
        "company_blacklist": normalized_blacklist,
        "prompt_template": get_company_extraction_prompt_template(),
    }
    return build_runtime_signature_hash(signature_payload)


def run_episode_company_extraction(
    *,
    work_item: EpisodeWorkItem,
    runtime: EpisodeExtractionRuntime,
    checkpoint_store: LlmCheckpointStore,
) -> CompletedEpisodeExtraction:
    episode_work = restore_or_prepare_episode_work(
        work_item=work_item,
        runtime=runtime,
        checkpoint_store=checkpoint_store,
    )
    if isinstance(episode_work, CompletedEpisodeExtraction):
        return episode_work
    return run_prepared_episode_llm_work(
        prepared_work=episode_work,
        runtime=runtime,
        checkpoint_store=checkpoint_store,
    )


def restore_or_prepare_episode_work(
    *,
    work_item: EpisodeWorkItem,
    runtime: EpisodeExtractionRuntime,
    checkpoint_store: LlmCheckpointStore,
) -> CompletedEpisodeExtraction | PreparedEpisodeLlmWork:
    checkpoint_result = restore_success_checkpoint(
        work_item=work_item,
        runtime=runtime,
        checkpoint_store=checkpoint_store,
    )
    if checkpoint_result is not None:
        return checkpoint_result
    return prepare_episode_llm_work(
        work_item=work_item,
        runtime=runtime,
        checkpoint_store=checkpoint_store,
    )


def restore_success_checkpoint(
    *,
    work_item: EpisodeWorkItem,
    runtime: EpisodeExtractionRuntime,
    checkpoint_store: LlmCheckpointStore,
) -> CompletedEpisodeExtraction | None:
    checkpoint_context = _load_episode_checkpoint_context(
        work_item=work_item,
        checkpoint_store=checkpoint_store,
    )
    checkpoint = checkpoint_context.checkpoint
    if checkpoint is None:
        return None

    if checkpoint.state.runtime_signature != runtime.runtime_signature:
        return None

    if checkpoint.state.status != STATUS_SUCCESS:
        return None

    logger.info(
        "命中成功检查点，直接复用：episode_key=%s", checkpoint_context.episode_key
    )
    extraction_result = CompanyExtractionResult.from_dict(
        {
            "companies": checkpoint.state.companies,
            "filtered_count": checkpoint.state.filtered_count,
        }
    )
    return CompletedEpisodeExtraction(
        episode=replace(
            checkpoint_context.episode,
            title=_resolve_title(checkpoint.state.title, work_item.title),
            pub_date=_resolve_text(checkpoint.state.pub_date, work_item.pub_date),
        ),
        episode_key=checkpoint_context.episode_key,
        extraction_result=extraction_result,
    )


def prepare_episode_llm_work(
    *,
    work_item: EpisodeWorkItem,
    runtime: EpisodeExtractionRuntime,
    checkpoint_store: LlmCheckpointStore,
) -> PreparedEpisodeLlmWork:
    checkpoint_context = _load_episode_checkpoint_context(
        work_item=work_item,
        checkpoint_store=checkpoint_store,
    )
    episode_key = checkpoint_context.episode_key
    checkpoint = checkpoint_context.checkpoint
    if checkpoint is not None:
        if checkpoint.state.runtime_signature != runtime.runtime_signature:
            logger.info(
                "检查点签名已变化，将重新抓取并执行 LLM：episode_key=%s",
                episode_key,
            )
        elif (
            checkpoint.state.status in {STATUS_PREPARED, STATUS_FAILED}
            and checkpoint.prompt_text
        ):
            logger.info("命中未完成检查点，直接继续 LLM：episode_key=%s", episode_key)
            return checkpoint_context.build_prepared_work(
                title=_resolve_title(checkpoint.state.title, work_item.title),
                pub_date=_resolve_text(
                    checkpoint.state.pub_date,
                    work_item.pub_date,
                ),
                prompt_text=checkpoint.prompt_text,
            )

    logger.info("抓取节目页面：%s", work_item.episode_url)
    episode = parse_episode_url(work_item.episode_url)
    prompt_text = build_company_extraction_prompt(
        build_company_extraction_input(episode)
    )
    title = _resolve_title(episode.title, work_item.title)
    prepared_work = checkpoint_context.build_prepared_work(
        title=title,
        pub_date=work_item.pub_date,
        prompt_text=prompt_text,
    )
    checkpoint_store.save_prepared(
        prepared_work.to_checkpoint_payload(runtime.runtime_signature)
    )
    return prepared_work


def run_prepared_episode_llm_work(
    *,
    prepared_work: PreparedEpisodeLlmWork,
    checkpoint_store: LlmCheckpointStore,
    runtime: EpisodeExtractionRuntime,
) -> CompletedEpisodeExtraction:
    checkpoint_payload = prepared_work.to_checkpoint_payload(runtime.runtime_signature)
    attempt = run_company_extraction_from_prompt(
        prepared_work.prompt_text,
        runtime.llm_client,
        company_blacklist=runtime.company_blacklist,
        retry_config=runtime.retry_config,
    )
    if attempt.error is not None:
        checkpoint_store.save_failed(
            checkpoint_payload,
            error_message=str(attempt.error),
            response_text=attempt.response_text,
        )
        raise attempt.error

    if attempt.extraction_result is None or attempt.response_text is None:
        raise ValueError("LLM 成功结果缺少必要字段。")

    checkpoint_store.save_success(
        checkpoint_payload,
        response_text=attempt.response_text,
        extraction_result=attempt.extraction_result,
    )
    return CompletedEpisodeExtraction(
        episode=prepared_work.episode,
        episode_key=prepared_work.episode_key,
        extraction_result=attempt.extraction_result,
    )


def _load_episode_checkpoint_context(
    *,
    work_item: EpisodeWorkItem,
    checkpoint_store: LlmCheckpointStore,
) -> _EpisodeCheckpointContext:
    resolved_eid = work_item.resolve_episode_id()
    resolved_episode = replace(work_item, eid=resolved_eid)
    episode_key = checkpoint_store.build_episode_key(
        eid=resolved_eid,
        episode_url=work_item.episode_url,
    )
    return _EpisodeCheckpointContext(
        episode=resolved_episode,
        episode_key=episode_key,
        checkpoint=checkpoint_store.load(episode_key),
    )


def _resolve_title(primary: str | None, fallback: str | None) -> str | None:
    return _resolve_text(primary, fallback)


def _resolve_text(primary: str | None, fallback: str | None) -> str | None:
    normalized_primary = (primary or "").strip()
    if normalized_primary:
        return normalized_primary
    normalized_fallback = (fallback or "").strip()
    if normalized_fallback:
        return normalized_fallback
    return None


def _build_episode_result_metadata(
    *,
    episode_url: str,
    eid: str | None,
    title: str | None,
    pub_date: str | None,
) -> dict[str, object]:
    return {
        "eid": eid,
        "title": title,
        "pub_date": pub_date,
        "episode_url": episode_url,
    }
