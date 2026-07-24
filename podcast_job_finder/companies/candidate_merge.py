from __future__ import annotations

import json
from collections import defaultdict
from typing import Final, Sequence

from podcast_job_finder.companies.models import (
    CompanyExtractionError,
    CompanyExtractionResult,
    CompanyMention,
)


MAX_EVIDENCE_CHARS_PER_CANDIDATE: Final = 300
MAX_EVIDENCE_COUNT_PER_NAME: Final = 2
INVALID_MERGED_EVIDENCE_ERROR: Final = "候选合并结果包含来源不明的证据：{evidence}"
INVALID_MERGED_NAME_ERROR: Final = "候选合并结果包含来源不明的公司名称：{name}"
CANDIDATE_MERGE_PROMPT_TEMPLATE: Final = """你是一个公司候选结果合并助手。

以下候选结果来自同一集播客的多个连续文本块，已经按统一规则完成初步提取。同一名称只提供少量代表性原文证据。

任务：
1. 合并指向同一招聘主体的全称、简称、别名和残缺名称。
2. 名称优先使用候选证据直接支持的最完整表达。
3. 每家公司只保留一个结果。
4. evidence 必须从候选结果的 evidence 中原样选择，不要改写或拼接。
5. 不要增加候选结果中没有的公司。
6. 输出必须是严格 JSON，且顶层必须是对象。
7. JSON 结构固定为：
{{"companies":[{{"name":"公司名","evidence":"原文片段"}}]}}
8. 没有候选时返回：
{{"companies":[]}}

候选结果：
{candidate_results}
"""


def build_candidate_merge_prompt(
    candidates: Sequence[CompanyMention],
) -> tuple[str, tuple[CompanyMention, ...]]:
    compact_candidates = _compact_candidates(candidates)
    candidate_results = [candidate.to_dict() for candidate in compact_candidates]
    prompt = CANDIDATE_MERGE_PROMPT_TEMPLATE.format(
        candidate_results=json.dumps(
            candidate_results,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    return prompt, compact_candidates


def validate_merged_result(
    result: CompanyExtractionResult,
    candidates: Sequence[CompanyMention],
) -> None:
    candidate_names = {candidate.name.casefold() for candidate in candidates}
    candidate_evidence = {candidate.evidence for candidate in candidates}
    for company in result.companies:
        if company.evidence not in candidate_evidence:
            raise CompanyExtractionError(
                INVALID_MERGED_EVIDENCE_ERROR.format(evidence=company.evidence)
            )
        normalized_name = company.name.casefold()
        evidence_supports_name = normalized_name in company.evidence.casefold()
        if normalized_name not in candidate_names and not evidence_supports_name:
            raise CompanyExtractionError(
                INVALID_MERGED_NAME_ERROR.format(name=company.name)
            )


def _compact_candidates(
    candidates: Sequence[CompanyMention],
) -> tuple[CompanyMention, ...]:
    evidence_by_name: dict[str, set[str]] = defaultdict(set)
    compact_candidates: list[CompanyMention] = []
    for candidate in candidates:
        normalized_name = candidate.name.casefold()
        selected_evidence = evidence_by_name[normalized_name]
        if len(selected_evidence) >= MAX_EVIDENCE_COUNT_PER_NAME:
            continue

        evidence_excerpt = _extract_evidence_excerpt(candidate)
        if evidence_excerpt in selected_evidence:
            continue

        selected_evidence.add(evidence_excerpt)
        compact_candidates.append(
            CompanyMention(name=candidate.name, evidence=evidence_excerpt)
        )
    return tuple(compact_candidates)


def _extract_evidence_excerpt(candidate: CompanyMention) -> str:
    """从较长的证据中取出一段适合交给模型判断的原文。

    一家公司可能在播客中被反复提到，每次提取出的证据也可能很长。把这些长文本
    全部放进合并请求会占用很多输入空间，所以这里只保留公司名附近的一小段内容。
    这段内容仍然逐字来自原证据，方便模型判断公司全称、简称和别名之间的关系。

    较短的证据会完整保留。较长的证据会尽量以公司名为中心截取；证据中找不到
    公司名时，则保留开头的一段，避免整条证据被丢弃。
    """
    evidence = candidate.evidence
    # 原文已经足够短，完整保留能提供更多上下文。
    if len(evidence) <= MAX_EVIDENCE_CHARS_PER_CANDIDATE:
        return evidence

    # 优先寻找公司名的位置，让截取结果尽量同时包含公司名及其前后内容。
    name_index = evidence.find(candidate.name)
    if name_index < 0:
        # 有些候选名称经过了整理，可能无法在原文中逐字找到，此时保留证据开头。
        return evidence[:MAX_EVIDENCE_CHARS_PER_CANDIDATE]

    # 从公司名前方开始截取，为公司名两侧都留出一些内容，便于理解提及场景。
    excerpt_start = max(
        0,
        name_index - MAX_EVIDENCE_CHARS_PER_CANDIDATE // 2,
    )
    excerpt_end = min(
        len(evidence),
        excerpt_start + MAX_EVIDENCE_CHARS_PER_CANDIDATE,
    )
    # 公司名靠近原文末尾时向前补足长度，避免截取结果无故变短。
    excerpt_start = max(0, excerpt_end - MAX_EVIDENCE_CHARS_PER_CANDIDATE)
    return evidence[excerpt_start:excerpt_end]
