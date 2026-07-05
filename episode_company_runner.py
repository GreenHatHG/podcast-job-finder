from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass

from company_extraction import (
    CompanyExtractionResult,
    build_company_extraction_input,
    build_company_extraction_prompt,
    get_company_extraction_prompt_template,
    run_company_extraction_from_prompt,
)
from extract_xiaoyuzhou_episode import extract_episode_id_from_url, parse_episode_url
from llm_checkpoint_store import (
    STATUS_FAILED,
    STATUS_PREPARED,
    STATUS_SUCCESS,
    LlmCheckpoint,
    LlmCheckpointStore,
)
from openai_compatible_llm import LlmRetryConfig, OpenAiCompatibleLlmClient


logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class EpisodeExtractionRuntime:
    llm_client: OpenAiCompatibleLlmClient
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


@dataclass(slots=True, frozen=True)
class EpisodeExtractionOutcome:
    episode_key: str
    episode_url: str
    eid: str | None
    title: str | None
    pub_date: str | None
    extraction_result: CompanyExtractionResult

    def to_pid_result_record(self) -> dict:
        return {
            "status": STATUS_SUCCESS,
            "eid": self.eid,
            "title": self.title,
            "pub_date": self.pub_date,
            "episode_url": self.episode_url,
            "companies": [
                company.to_dict() for company in self.extraction_result.companies
            ],
            "filtered_count": self.extraction_result.filtered_count,
        }


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
    serialized_payload = json.dumps(
        signature_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized_payload.encode("utf-8")).hexdigest()


def run_episode_company_extraction(
    *,
    work_item: EpisodeWorkItem,
    runtime: EpisodeExtractionRuntime,
    checkpoint_store: LlmCheckpointStore,
) -> EpisodeExtractionOutcome:
    resolved_eid = _resolve_episode_id(work_item)
    episode_key = checkpoint_store.build_episode_key(
        eid=resolved_eid,
        episode_url=work_item.episode_url,
    )
    checkpoint = checkpoint_store.load(episode_key)
    if checkpoint is not None:
        if checkpoint.state.runtime_signature != runtime.runtime_signature:
            logger.info(
                "检查点签名已变化，将重新抓取并执行 LLM：episode_key=%s",
                episode_key,
            )
        elif checkpoint.state.status == STATUS_SUCCESS:
            logger.info("命中成功检查点，直接复用：episode_key=%s", episode_key)
            return _build_success_outcome_from_checkpoint(
                checkpoint=checkpoint,
                work_item=work_item,
                episode_key=episode_key,
                resolved_eid=resolved_eid,
            )
        elif (
            checkpoint.state.status in {STATUS_PREPARED, STATUS_FAILED}
            and checkpoint.prompt_text
        ):
            logger.info("命中未完成检查点，直接继续 LLM：episode_key=%s", episode_key)
            return _run_llm_with_prompt(
                checkpoint_store=checkpoint_store,
                runtime=runtime,
                episode_key=episode_key,
                episode_url=work_item.episode_url,
                eid=resolved_eid,
                title=_resolve_title(checkpoint.state.title, work_item.title),
                pub_date=_resolve_text(checkpoint.state.pub_date, work_item.pub_date),
                prompt_text=checkpoint.prompt_text,
            )

    logger.info("抓取节目页面：%s", work_item.episode_url)
    episode = parse_episode_url(work_item.episode_url)
    prompt_text = build_company_extraction_prompt(
        build_company_extraction_input(episode)
    )
    title = _resolve_title(episode.title, work_item.title)
    checkpoint_store.save_prepared(
        episode_key=episode_key,
        episode_url=work_item.episode_url,
        title=title,
        pub_date=work_item.pub_date,
        runtime_signature=runtime.runtime_signature,
        prompt_text=prompt_text,
    )
    return _run_llm_with_prompt(
        checkpoint_store=checkpoint_store,
        runtime=runtime,
        episode_key=episode_key,
        episode_url=work_item.episode_url,
        eid=resolved_eid,
        title=title,
        pub_date=work_item.pub_date,
        prompt_text=prompt_text,
    )


def _run_llm_with_prompt(
    *,
    checkpoint_store: LlmCheckpointStore,
    runtime: EpisodeExtractionRuntime,
    episode_key: str,
    episode_url: str,
    eid: str | None,
    title: str | None,
    pub_date: str | None,
    prompt_text: str,
) -> EpisodeExtractionOutcome:
    attempt = run_company_extraction_from_prompt(
        prompt_text,
        runtime.llm_client,
        company_blacklist=runtime.company_blacklist,
        retry_config=runtime.retry_config,
    )
    if attempt.error is not None:
        checkpoint_store.save_failed(
            episode_key=episode_key,
            episode_url=episode_url,
            title=title,
            pub_date=pub_date,
            runtime_signature=runtime.runtime_signature,
            prompt_text=prompt_text,
            error_message=str(attempt.error),
            response_text=attempt.response_text,
        )
        raise attempt.error

    if attempt.extraction_result is None or attempt.response_text is None:
        raise ValueError("LLM 成功结果缺少必要字段。")

    checkpoint_store.save_success(
        episode_key=episode_key,
        episode_url=episode_url,
        title=title,
        pub_date=pub_date,
        runtime_signature=runtime.runtime_signature,
        prompt_text=prompt_text,
        response_text=attempt.response_text,
        extraction_result=attempt.extraction_result,
    )
    return EpisodeExtractionOutcome(
        episode_key=episode_key,
        episode_url=episode_url,
        eid=eid,
        title=title,
        pub_date=pub_date,
        extraction_result=attempt.extraction_result,
    )


def _build_success_outcome_from_checkpoint(
    *,
    checkpoint: LlmCheckpoint,
    work_item: EpisodeWorkItem,
    episode_key: str,
    resolved_eid: str | None,
) -> EpisodeExtractionOutcome:
    extraction_result = CompanyExtractionResult.from_dict(
        {
            "companies": checkpoint.state.companies,
            "filtered_count": checkpoint.state.filtered_count,
        }
    )
    return EpisodeExtractionOutcome(
        episode_key=episode_key,
        episode_url=work_item.episode_url,
        eid=resolved_eid,
        title=_resolve_title(checkpoint.state.title, work_item.title),
        pub_date=_resolve_text(checkpoint.state.pub_date, work_item.pub_date),
        extraction_result=extraction_result,
    )


def _resolve_episode_id(work_item: EpisodeWorkItem) -> str | None:
    normalized_eid = _resolve_text(work_item.eid, None)
    if normalized_eid is not None:
        return normalized_eid
    return extract_episode_id_from_url(work_item.episode_url)


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
