from __future__ import annotations

import json
import logging
import re
from typing import Collection, Final, Protocol

from podcast_job_finder.companies.models import (
    COMPANIES_FIELD,
    INVALID_COMPANIES_FIELD_ERROR,
    INVALID_TOP_LEVEL_ERROR,
    CompanyExtractionAttempt,
    CompanyExtractionError,
    CompanyExtractionResult,
    CompanyMention,
)
from podcast_job_finder.llm import (
    EmptyLlmResponseError,
    LlmRetryConfig,
    LlmRetryExhaustedError,
    OpenAiCompatibleLlmError,
    RetryableOpenAiCompatibleLlmError,
    execute_llm_with_retry,
)
from podcast_job_finder.xiaoyuzhou.models import CommentInfo, EpisodeInfo


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
从播客的标题、正文、评论中提取明确提到的招聘主体名称。

提取规则：
1. 只保留文本里明确出现的名称。
2. 公司、企业、工作室、事务所、基金管理人可以返回。
3. 品牌、App、平台、播客厂牌在文本中被明确当作出品方、运营方、招聘方、任职方、团队主体时可以返回。
4. 学校、学院、研究院、医院、政府部门、协会、基金会这类非招聘主体不要返回。
5. 纯平台曝光、分发渠道、榜单、奖项、收听入口、节目名称不要返回。
6. 纯人物名、书名、泛行业词、模糊指代不要返回。
7. 每个结果必须附带一段原文证据，证据必须直接来自输入文本。
8. 同一家公司只保留一个结果。
9. 输出必须是严格 JSON，且顶层必须是对象。
10. JSON 结构固定为：
{{"companies":[{{"name":"公司名","evidence":"原文片段"}}]}}
11. 没有命中时返回：
{{"companies":[]}}

待处理文本：
{episode_text}
"""
INVALID_RESPONSE_ERROR = "LLM 返回结果不是合法 JSON。"
LLM_REQUEST_RETRY_EXHAUSTED_TEMPLATE = (
    "LLM 调用连续 {max_attempts} 次失败，最后一次错误：{error_message}"
)
LLM_INVALID_RESULT_RETRY_EXHAUSTED_TEMPLATE = (
    "LLM 连续 {max_attempts} 次返回无效结果，最后一次错误：{error_message}"
)
JSON_CODE_BLOCK_PATTERN = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
REQUEST_FAILURE_CATEGORY = "request_failure"
INVALID_RESULT_CATEGORY = "invalid_result"

logger = logging.getLogger(__name__)


COMPANY_EXTRACTION_RETRYABLE_ERRORS: Final[tuple[type[Exception], ...]] = (
    RetryableOpenAiCompatibleLlmError,
    EmptyLlmResponseError,
    CompanyExtractionError,
)


class LlmClientProtocol(Protocol):
    def generate(self, prompt: str) -> str:
        """Generates a text response for the provided prompt."""
        ...


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


def get_company_extraction_prompt_template() -> str:
    return PROMPT_TEMPLATE


def parse_company_extraction_output(
    response_text: str,
    company_blacklist: Collection[str] | None = None,
) -> CompanyExtractionResult:
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

    normalized_company_blacklist = _normalize_company_blacklist(company_blacklist)
    seen_company_names: set[str] = set()
    companies: list[CompanyMention] = []
    filtered_count = 0
    for company_data in companies_data:
        company = CompanyMention.from_dict(company_data)
        normalized_name = _normalize_company_name(company.name)
        if normalized_name in seen_company_names:
            continue

        seen_company_names.add(normalized_name)
        if normalized_name in normalized_company_blacklist:
            filtered_count += 1
            logger.debug(
                "公司命中黑名单，已过滤：name=%s evidence=%s",
                company.name,
                company.evidence,
            )
            continue

        companies.append(company)
    return CompanyExtractionResult(
        companies=companies,
        filtered_count=filtered_count,
    )


def run_company_extraction_from_prompt(
    prompt: str,
    llm_client: LlmClientProtocol,
    company_blacklist: Collection[str] | None = None,
    retry_config: LlmRetryConfig | None = None,
) -> CompanyExtractionAttempt:
    last_response_text: str | None = None

    def request_company_extraction() -> CompanyExtractionAttempt:
        nonlocal last_response_text
        last_response_text = None
        response_text = llm_client.generate(prompt)
        last_response_text = response_text
        extraction_result = parse_company_extraction_output(
            response_text,
            company_blacklist=company_blacklist,
        )
        return CompanyExtractionAttempt(
            response_text=response_text,
            extraction_result=extraction_result,
        )

    try:
        result, attempt = execute_llm_with_retry(
            request_company_extraction,
            retry_config=retry_config,
            retryable_errors=COMPANY_EXTRACTION_RETRYABLE_ERRORS,
        )
    except LlmRetryExhaustedError as error:
        return _build_failed_company_extraction_attempt(last_response_text, error)
    except OpenAiCompatibleLlmError as error:
        return CompanyExtractionAttempt(error=error)

    _log_company_extraction_result(result, attempt)
    return result


def _build_failed_company_extraction_attempt(
    last_response_text: str | None,
    retry_error: LlmRetryExhaustedError,
) -> CompanyExtractionAttempt:
    last_error = retry_error.last_error
    failure_category = (
        REQUEST_FAILURE_CATEGORY
        if isinstance(last_error, RetryableOpenAiCompatibleLlmError)
        else INVALID_RESULT_CATEGORY
    )
    return CompanyExtractionAttempt(
        response_text=last_response_text,
        error=_build_retry_exhausted_error(
            max_attempts=retry_error.max_attempts,
            failure_category=failure_category,
            last_error=last_error,
        ),
    )


def _log_company_extraction_result(
    result: CompanyExtractionAttempt,
    attempt: int,
) -> None:
    extraction_result = result.extraction_result
    if extraction_result is None:
        return
    company_names = [company.name for company in extraction_result.companies]
    logger.info(
        "LLM 返回结果：%d 家公司 %s",
        len(extraction_result.companies),
        company_names,
    )
    if attempt > 1:
        logger.debug("LLM 第 %s 次尝试成功。", attempt)


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


def _normalize_company_blacklist(
    company_blacklist: Collection[str] | None,
) -> frozenset[str]:
    if not company_blacklist:
        return frozenset()

    return frozenset(
        _normalize_company_name(company_name)
        for company_name in company_blacklist
        if company_name.strip()
    )


def _normalize_company_name(company_name: str) -> str:
    return company_name.strip().casefold()


def _strip_json_code_block(response_text: str) -> str:
    matched_block = JSON_CODE_BLOCK_PATTERN.match(response_text.strip())
    if matched_block is None:
        return response_text
    return matched_block.group(1)


def _build_retry_exhausted_error(
    max_attempts: int,
    failure_category: str,
    last_error: Exception,
) -> Exception:
    if failure_category == REQUEST_FAILURE_CATEGORY:
        return OpenAiCompatibleLlmError(
            LLM_REQUEST_RETRY_EXHAUSTED_TEMPLATE.format(
                max_attempts=max_attempts,
                error_message=str(last_error),
            )
        )

    return CompanyExtractionError(
        LLM_INVALID_RESULT_RETRY_EXHAUSTED_TEMPLATE.format(
            max_attempts=max_attempts,
            error_message=str(last_error),
        )
    )
