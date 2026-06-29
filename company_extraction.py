from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Protocol

from extract_xiaoyuzhou_episode import CommentInfo, EpisodeInfo


TITLE_SECTION_LABEL = "标题"
BODY_SECTION_LABEL = "正文"
COMMENTS_SECTION_LABEL = "评论"
NO_COMMENTS_LABEL = "无评论"
COMMENT_BLOCK_TEMPLATE = "评论 {index}"
REPLY_BLOCK_TEMPLATE = "回复 {index}"
AUTHOR_LINE_TEMPLATE = "作者：{author}"
CONTENT_LABEL = "内容："
PROMPT_TEMPLATE = """你是一个信息抽取助手。

任务：
从播客的标题、正文、评论中提取明确提到的公司、机构或品牌名称。

提取规则：
1. 只保留文本里明确出现的名称。
2. 纯人物名、书名、泛行业词、模糊指代不要返回。
3. 每个结果必须附带一段原文证据，证据必须直接来自输入文本。
4. 同一家公司只保留一个结果。
5. 输出必须是严格 JSON，且顶层必须是对象。
6. JSON 结构固定为：
{{"companies":[{{"name":"公司名","evidence":"原文片段"}}]}}
7. 没有命中时返回：
{{"companies":[]}}

待处理文本：
{episode_text}
"""
COMPANIES_FIELD = "companies"
NAME_FIELD = "name"
EVIDENCE_FIELD = "evidence"
INVALID_RESPONSE_ERROR = "LLM 返回结果不是合法 JSON。"
INVALID_TOP_LEVEL_ERROR = "LLM 返回结果的顶层结构必须是对象。"
INVALID_COMPANIES_FIELD_ERROR = "LLM 返回结果缺少 companies 数组。"
INVALID_COMPANY_ITEM_ERROR = "LLM 返回结果中的公司项必须是对象。"
MISSING_COMPANY_FIELD_ERROR = "LLM 返回结果中的公司项缺少必要字段。"
EMPTY_COMPANY_FIELD_ERROR = "LLM 返回结果中的公司项字段不能为空。"
JSON_CODE_BLOCK_PATTERN = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


class CompanyExtractionError(ValueError):
    """Raised when the company extraction result is invalid."""


class LlmClientProtocol(Protocol):
    def generate(self, prompt: str) -> str:
        """Generates a text response for the provided prompt."""


@dataclass(slots=True)
class CompanyMention:
    name: str
    evidence: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class CompanyExtractionResult:
    companies: list[CompanyMention] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def build_company_extraction_input(episode: EpisodeInfo) -> str:
    sections = [
        TITLE_SECTION_LABEL,
        episode.title,
        "",
        BODY_SECTION_LABEL,
        episode.content,
        "",
        COMMENTS_SECTION_LABEL,
    ]
    if not episode.comments:
        sections.append(NO_COMMENTS_LABEL)
        return "\n".join(sections)

    comment_lines: list[str] = []
    for comment_index, comment in enumerate(episode.comments, start=1):
        comment_lines.extend(_comment_to_input_lines(comment, str(comment_index), 0))
    sections.append("\n".join(comment_lines))
    return "\n".join(sections)


def build_company_extraction_prompt(episode_text: str) -> str:
    return PROMPT_TEMPLATE.format(episode_text=episode_text)


def parse_company_extraction_output(response_text: str) -> CompanyExtractionResult:
    normalized_response_text = _strip_json_code_block(response_text).strip()
    try:
        payload = json.loads(normalized_response_text)
    except json.JSONDecodeError as error:
        raise CompanyExtractionError(INVALID_RESPONSE_ERROR) from error

    if not isinstance(payload, dict):
        raise CompanyExtractionError(INVALID_TOP_LEVEL_ERROR)

    companies_data = payload.get(COMPANIES_FIELD)
    if not isinstance(companies_data, list):
        raise CompanyExtractionError(INVALID_COMPANIES_FIELD_ERROR)

    seen_company_names: set[str] = set()
    companies: list[CompanyMention] = []
    for company_data in companies_data:
        company = _parse_company_item(company_data)
        if company is None:
            continue

        normalized_name = company.name.strip()
        if normalized_name in seen_company_names:
            continue

        seen_company_names.add(normalized_name)
        companies.append(company)
    return CompanyExtractionResult(companies=companies)


def extract_companies_from_episode(
    episode: EpisodeInfo,
    llm_client: LlmClientProtocol,
) -> CompanyExtractionResult:
    episode_text = build_company_extraction_input(episode)
    prompt = build_company_extraction_prompt(episode_text)
    response_text = llm_client.generate(prompt)
    return parse_company_extraction_output(response_text)


def _comment_to_input_lines(
    comment: CommentInfo,
    index_label: str,
    depth: int,
) -> list[str]:
    block_title = (
        COMMENT_BLOCK_TEMPLATE if depth == 0 else REPLY_BLOCK_TEMPLATE
    ).format(index=index_label)
    lines = [
        block_title,
        AUTHOR_LINE_TEMPLATE.format(author=comment.author),
        CONTENT_LABEL,
        comment.text,
        "",
    ]
    for reply_index, reply in enumerate(comment.replies, start=1):
        reply_label = f"{index_label}.{reply_index}"
        lines.extend(_comment_to_input_lines(reply, reply_label, depth + 1))
    return lines


def _parse_company_item(company_data: object) -> CompanyMention:
    if not isinstance(company_data, dict):
        raise CompanyExtractionError(INVALID_COMPANY_ITEM_ERROR)

    if NAME_FIELD not in company_data or EVIDENCE_FIELD not in company_data:
        raise CompanyExtractionError(MISSING_COMPANY_FIELD_ERROR)

    name = str(company_data[NAME_FIELD]).strip()
    evidence = str(company_data[EVIDENCE_FIELD]).strip()
    if not name or not evidence:
        raise CompanyExtractionError(EMPTY_COMPANY_FIELD_ERROR)

    return CompanyMention(name=name, evidence=evidence)


def _strip_json_code_block(response_text: str) -> str:
    matched_block = JSON_CODE_BLOCK_PATTERN.match(response_text.strip())
    if matched_block is None:
        return response_text
    return matched_block.group(1)
