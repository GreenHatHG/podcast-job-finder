from __future__ import annotations

from pathlib import Path
from typing import Final, Protocol

from podcast_job_finder.audio.segment_export import (
    ExportedSpeechSegment,
    get_segment_audio_format,
)
from podcast_job_finder.audio.transcription import (
    AudioTranscriptionError,
    TranscribedSpeechSegment,
    TranscriptionOutput,
)
from podcast_job_finder.llm import (
    AudioFormat,
    EmptyLlmResponseError,
    LlmRetryConfig,
    LlmRetryExhaustedError,
    RetryableOpenAiCompatibleLlmError,
    execute_llm_with_retry,
)


PREVIOUS_CONTEXT_MAX_CHARS: Final = 200
NO_PREVIOUS_CONTEXT_TEXT: Final = "无"
TRANSCRIPTION_PROMPT_TEMPLATE: Final = """你是专业的中文音频转写助手。

请准确转写这段播客音频，保留人名、公司名、产品名和英文词的原始表达。

上一片段末尾文本（仅供理解上下文，不要重复输出）：
{previous_context}

要求：
1. 只输出当前音频对应的转写正文。
2. 使用自然的中文标点，不添加解释、标题或 Markdown。
3. 音频开头与上一片段重复的内容只保留一次。
4. 无法确认的内容保留原始发音，不要编造。
"""
READ_AUDIO_ERROR_TEMPLATE: Final = "无法读取待识别音频：{path}，{error_message}"
TRANSCRIPTION_OPERATION_NAME_TEMPLATE: Final = "音频片段 {index} 转写"
TRANSCRIPTION_RETRY_EXHAUSTED_TEMPLATE: Final = (
    "音频片段 {index} 连续 {max_attempts} 次转写失败，最后一次错误：{error_message}"
)
TRANSCRIPTION_RETRYABLE_ERRORS: Final[tuple[type[Exception], ...]] = (
    RetryableOpenAiCompatibleLlmError,
    EmptyLlmResponseError,
)


class AudioTranscriptionClientProtocol(Protocol):
    def transcribe_audio(
        self,
        audio_data: bytes,
        *,
        audio_format: AudioFormat,
        prompt: str,
    ) -> str: ...


class LlmAudioTranscriber:
    def __init__(
        self,
        client: AudioTranscriptionClientProtocol,
        *,
        retry_config: LlmRetryConfig | None = None,
    ) -> None:
        self._client = client
        self._retry_config = retry_config

    def transcribe(
        self,
        segment: ExportedSpeechSegment,
        *,
        previous_segment: TranscribedSpeechSegment | None,
    ) -> TranscriptionOutput:
        audio_data = _read_audio(segment.file_path)
        # 上一段文字帮助模型理解当前片段开头，减少断句和重复。
        previous_text = previous_segment.text if previous_segment is not None else ""
        prompt = _build_transcription_prompt(previous_text)
        try:
            text, _ = execute_llm_with_retry(
                lambda: self._client.transcribe_audio(
                    audio_data,
                    audio_format=get_segment_audio_format(segment.file_path),
                    prompt=prompt,
                ),
                retry_config=self._retry_config,
                retryable_errors=TRANSCRIPTION_RETRYABLE_ERRORS,
                operation_name=TRANSCRIPTION_OPERATION_NAME_TEMPLATE.format(
                    index=segment.index
                ),
            )
        except LlmRetryExhaustedError as error:
            raise AudioTranscriptionError(
                TRANSCRIPTION_RETRY_EXHAUSTED_TEMPLATE.format(
                    index=segment.index,
                    max_attempts=error.max_attempts,
                    error_message=str(error.last_error),
                )
            ) from error
        return TranscriptionOutput(text=text)

    def close(self) -> None:
        return


def _build_transcription_prompt(previous_text: str) -> str:
    normalized_previous_text = " ".join(previous_text.split())
    previous_context = (
        normalized_previous_text[-PREVIOUS_CONTEXT_MAX_CHARS:]
        if normalized_previous_text
        else NO_PREVIOUS_CONTEXT_TEXT
    )
    return TRANSCRIPTION_PROMPT_TEMPLATE.format(previous_context=previous_context)


def _read_audio(audio_path: Path) -> bytes:
    try:
        return audio_path.read_bytes()
    except OSError as error:
        raise AudioTranscriptionError(
            READ_AUDIO_ERROR_TEMPLATE.format(
                path=audio_path,
                error_message=str(error),
            )
        ) from error
