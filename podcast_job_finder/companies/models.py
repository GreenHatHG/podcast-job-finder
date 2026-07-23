from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Final


COMPANIES_FIELD: Final = "companies"
NAME_FIELD: Final = "name"
EVIDENCE_FIELD: Final = "evidence"
INVALID_TOP_LEVEL_ERROR: Final = "LLM 返回结果的顶层结构必须是对象。"
INVALID_COMPANIES_FIELD_ERROR: Final = "LLM 返回结果缺少 companies 数组。"
INVALID_COMPANY_ITEM_ERROR: Final = "LLM 返回结果中的公司项必须是对象。"
MISSING_COMPANY_FIELD_ERROR: Final = "LLM 返回结果中的公司项缺少必要字段。"
EMPTY_COMPANY_FIELD_ERROR: Final = "LLM 返回结果中的公司项字段不能为空。"


class CompanyExtractionError(ValueError):
    """Raised when the company extraction result is invalid."""


@dataclass(slots=True)
class CompanyMention:
    name: str
    evidence: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: object) -> "CompanyMention":
        if not isinstance(payload, dict):
            raise CompanyExtractionError(INVALID_COMPANY_ITEM_ERROR)
        if NAME_FIELD not in payload or EVIDENCE_FIELD not in payload:
            raise CompanyExtractionError(MISSING_COMPANY_FIELD_ERROR)

        name = str(payload[NAME_FIELD]).strip()
        evidence = str(payload[EVIDENCE_FIELD]).strip()
        if not name or not evidence:
            raise CompanyExtractionError(EMPTY_COMPANY_FIELD_ERROR)
        return cls(name=name, evidence=evidence)


@dataclass(slots=True)
class CompanyExtractionResult:
    companies: list[CompanyMention] = field(default_factory=list)
    filtered_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: object) -> "CompanyExtractionResult":
        if not isinstance(payload, dict):
            raise CompanyExtractionError(INVALID_TOP_LEVEL_ERROR)

        companies_data = payload.get(COMPANIES_FIELD, [])
        if not isinstance(companies_data, list):
            raise CompanyExtractionError(INVALID_COMPANIES_FIELD_ERROR)

        filtered_count = payload.get("filtered_count", 0)
        if not isinstance(filtered_count, int):
            raise CompanyExtractionError("filtered_count 必须是整数。")
        return cls(
            companies=[
                CompanyMention.from_dict(company_data)
                for company_data in companies_data
            ],
            filtered_count=filtered_count,
        )


@dataclass(slots=True)
class CompanyExtractionAttempt:
    response_text: str | None = None
    extraction_result: CompanyExtractionResult | None = None
    error: Exception | None = None
