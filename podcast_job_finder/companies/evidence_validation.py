from __future__ import annotations

from typing import Final, Sequence

from podcast_job_finder.companies.models import (
    CompanyExtractionError,
    CompanyMention,
)


INVALID_EVIDENCE_SOURCE_ERROR: Final = (
    "公司提取结果的证据不是任一允许原文中的连续逐字片段"
    "（比较时仅忽略空白字符）：name={name} evidence={evidence}"
)
EVIDENCE_MISSING_NAME_ERROR: Final = "公司提取结果的证据未直接包含公司名称：{name}"


def validate_company_evidence(
    company: CompanyMention,
    allowed_sources: Sequence[str],
) -> None:
    normalized_evidence = _remove_whitespace(company.evidence)
    evidence_has_source = any(
        normalized_evidence in _remove_whitespace(source_text)
        for source_text in allowed_sources
    )
    if not evidence_has_source:
        raise CompanyExtractionError(
            INVALID_EVIDENCE_SOURCE_ERROR.format(
                name=company.name,
                evidence=company.evidence,
            )
        )

    if find_company_name(company.evidence, company.name) is None:
        raise CompanyExtractionError(
            EVIDENCE_MISSING_NAME_ERROR.format(name=company.name)
        )


def find_company_name(source_text: str, company_name: str) -> int | None:
    folded_name = company_name.casefold()
    folded_source_parts: list[str] = []
    source_indexes: list[int] = []
    for source_index, character in enumerate(source_text):
        folded_character = character.casefold()
        folded_source_parts.append(folded_character)
        source_indexes.extend([source_index] * len(folded_character))

    folded_index = "".join(folded_source_parts).find(folded_name)
    if folded_index < 0:
        return None
    return source_indexes[folded_index]


def _remove_whitespace(text: str) -> str:
    return "".join(text.split())
