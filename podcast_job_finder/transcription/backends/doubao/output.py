from __future__ import annotations

from typing import Final

from podcast_job_finder.audio.segmentation.segment_export import ExportedSpeechSegment
from podcast_job_finder.transcription.models import (
    AudioTranscriptionError,
    CharacterAlignment,
    TimedTranscriptionText,
    TranscribedSpeechSegment,
    TranscriptionOutput,
)
from podcast_job_finder.transcription.diagnostics import (
    TranscriptionDiagnostics,
)


SENTENCE_END_CHARACTERS: Final = frozenset("。！？!?；;")
DOUBAO_ALIGNMENT_MISSING_ERROR: Final = "豆包返回了文字，但没有生成逐字时间：{path}"
DOUBAO_OVERLAP_ONLY_RESULT_ERROR: Final = (
    "豆包返回的文字全部落在已处理的重叠音频内："
    "{path}（最晚时间 {latest_end_ms}ms，去重边界 {cutoff_ms}ms）"
)
DOUBAO_ALIGNMENT_AT_SEGMENT_START_ERROR: Final = (
    "豆包逐字时间没有超过当前片段起点："
    "{path}（最晚时间 {latest_end_ms}ms，片段起点 {cutoff_ms}ms）"
)


def build_doubao_transcription_output(  # pylint: disable=too-many-arguments
    text: str,
    alignments: tuple[CharacterAlignment, ...],
    *,
    segment: ExportedSpeechSegment,
    previous_segment: TranscribedSpeechSegment | None,
    silence_padding_ms: int,
    diagnostics: TranscriptionDiagnostics,
) -> TranscriptionOutput:
    """整理一次豆包识别结果，去掉与上一音频片段重复的文字。

    较长的录音会被切成多个小片段。为了避免在切片处漏掉一句话的开头或结尾，
    相邻片段可能包含一小段相同的音频。本函数先把每个字在当前片段中的时间
    换算成它在原录音中的时间，再删掉上一片段已经处理过的部分，最后生成可保存
    的文字、每个字的时间和每句话的时间。

    Args:
        text: 豆包识别出的完整文字。
        alignments: 可以从声音中定位的每个字，以及它在当前片段中的开始和结束
            时间。标点等没有声音的字符可能不在其中。
        segment: 当前片段在原录音中的位置，以及导出后的音频文件路径。
        previous_segment: 上一片段的转写结果。它的结束时间用于判断当前片段开头
            有多少音频已经处理过。
        silence_padding_ms: 导出音频时在片段开头额外补入的静音长度。
        diagnostics: 供后续查看问题的检查记录，会原样放入最终结果。

    Returns:
        去掉重复内容后的文字、每个字的时间、每句话的时间和检查记录。

    Raises:
        AudioTranscriptionError: 豆包返回了文字却没有任何逐字时间，或者所有文字
            都位于上一片段已经处理过的音频中，或者逐字时间全部停在片段起点。
    """
    # 文字和逐字时间同时为空，说明这个片段没有可保存的识别内容。这属于有效
    # 结果，因此返回空文字，同时保留检查记录，方便之后确认本次识别发生了什么。
    if not text and not alignments:
        return TranscriptionOutput(
            text="",
            diagnostics=diagnostics,
        )

    # 调用本函数前，上游已经完成必要的重试，并选出了最终采用的识别结果。
    # 此时有文字却没有任何逐字时间，程序便无法判断哪些文字与上一片段重复，
    # 继续保存可能产生重复文字，所以在这里明确报告错误。
    if text and not alignments:
        raise AudioTranscriptionError(
            DOUBAO_ALIGNMENT_MISSING_ERROR.format(path=segment.file_path)
        )

    # alignments 中的时间从导出音频的开头算起，其中包含额外补入的静音。
    # 这里先减去静音长度，再加上当前片段在原录音中的开始时间。换算后，所有
    # 片段的逐字时间都表示它们在同一份原录音中的实际位置，后面才能正确去重。
    absolute_alignments = tuple(
        _to_absolute_alignment(
            alignment,
            segment=segment,
            silence_padding_ms=silence_padding_ms,
        )
        for alignment in alignments
    )

    # cutoff_ms 表示当前片段从哪个时间点之后才算新内容。存在重叠时，它通常是
    # 上一片段的结束时间；没有上一片段或两段没有重叠时，它是当前片段的起点。
    cutoff_ms = _build_overlap_cutoff_ms(
        segment,
        previous_segment=previous_segment,
    )

    # 结束时间没有超过 cutoff_ms 的字完全位于已处理区域，因此可以丢弃。一个字
    # 即使从边界之前开始，只要它在边界之后才结束，仍会保留，避免删掉跨越边界
    # 的发音。
    retained = tuple(item for item in absolute_alignments if item.end_ms > cutoff_ms)

    # absolute_alignments 在这里一定有内容。retained 为空表示所有字的结束时间都
    # 早于或等于 cutoff_ms。两段确实重叠时，豆包识别出的文字全部来自已经处理过
    # 的重叠音频；没有实际重叠时，逐字时间全部停在片段起点，说明时间数据异常。
    if not retained:
        latest_end_ms = max(item.end_ms for item in absolute_alignments)
        if cutoff_ms > segment.segment.start_ms:
            message = DOUBAO_OVERLAP_ONLY_RESULT_ERROR.format(
                path=segment.file_path,
                latest_end_ms=latest_end_ms,
                cutoff_ms=cutoff_ms,
            )
        else:
            message = DOUBAO_ALIGNMENT_AT_SEGMENT_START_ERROR.format(
                path=segment.file_path,
                latest_end_ms=latest_end_ms,
                cutoff_ms=cutoff_ms,
            )
        raise AudioTranscriptionError(message)

    # source_start 是第一个保留字在完整 text 中的字符位置。文字也从这个位置开始
    # 截取，确保最终文字与 retained 中的逐字时间指向同一段内容。
    source_start = retained[0].source_start

    # 每个字的时间和每句话的时间都使用 retained 生成，这样最终结果中的文字与
    # 时间信息经过了相同的去重处理。
    return TranscriptionOutput(
        text=text[source_start:].strip(),
        character_timestamps=_build_character_timestamps(retained),
        sentences=_build_sentences(text, retained, source_start=source_start),
        diagnostics=diagnostics,
    )


def _build_character_timestamps(
    alignments: tuple[CharacterAlignment, ...],
) -> tuple[TimedTranscriptionText, ...]:
    return tuple(
        TimedTranscriptionText(
            text=item.text,
            start_ms=item.start_ms,
            end_ms=item.end_ms,
            confidence=item.confidence,
        )
        for item in alignments
    )


def _to_absolute_alignment(
    alignment: CharacterAlignment,
    *,
    segment: ExportedSpeechSegment,
    silence_padding_ms: int,
) -> CharacterAlignment:
    return CharacterAlignment(
        text=alignment.text,
        source_start=alignment.source_start,
        source_end=alignment.source_end,
        start_ms=_to_source_timestamp(
            alignment.start_ms,
            segment=segment,
            silence_padding_ms=silence_padding_ms,
        ),
        end_ms=_to_source_timestamp(
            alignment.end_ms,
            segment=segment,
            silence_padding_ms=silence_padding_ms,
        ),
        confidence=alignment.confidence,
    )


def _to_source_timestamp(
    timestamp_ms: int,
    *,
    segment: ExportedSpeechSegment,
    silence_padding_ms: int,
) -> int:
    relative_ms = max(0, timestamp_ms - silence_padding_ms)
    return min(
        segment.segment.end_ms,
        segment.segment.start_ms + relative_ms,
    )


def _build_overlap_cutoff_ms(
    segment: ExportedSpeechSegment,
    *,
    previous_segment: TranscribedSpeechSegment | None,
) -> int:
    if previous_segment is None:
        return segment.segment.start_ms
    return max(
        segment.segment.start_ms,
        previous_segment.end_ms,
    )


def _build_sentences(
    text: str,
    alignments: tuple[CharacterAlignment, ...],
    *,
    source_start: int,
) -> tuple[TimedTranscriptionText, ...]:
    """按句末标点切分文字，并为每句话补上它在原录音中的时间。

    source_start 之前的文字已经在相邻音频片段去重时被丢弃。本函数只处理这个
    位置之后的文字，并使用同一范围内的逐字时间确定每句话的开始和结束时间。
    最后一段文字即使没有句末标点，也会作为最后一个句子片段保留下来，原文字不变。

    Args:
        text: 豆包识别出的完整文字，包含可能用于分句的标点。
        alignments: 去重后保留的逐字时间。标点等没有声音的字符可能不在其中。
        source_start: 第一个保留字在完整文字中的字符位置。

    Returns:
        按原顺序排列的句子，以及每句话在原录音中的开始和结束时间。
    """
    sentences = []
    sentence_start = source_start

    # _sentence_end_positions 返回句末标点后一个字符的位置，因此 text 的切片会
    # 包含这个标点。每处理完一句，就从它的结尾继续查找下一句。
    for sentence_end in _sentence_end_positions(text, source_start=source_start):
        sentence = _build_sentence(
            text,
            alignments,
            start=sentence_start,
            end=sentence_end,
        )
        if sentence is not None:
            sentences.append(sentence)
        sentence_start = sentence_end

    # 最后一个句末标点到文字结尾之间可能还有内容。单独处理这一段，避免没有
    # 句末标点的最后一段文字被遗漏；这里不会补标点，若末尾只有空白，
    # _build_sentence 会将其忽略。
    trailing_sentence = _build_sentence(
        text,
        alignments,
        start=sentence_start,
        end=len(text),
    )
    if trailing_sentence is not None:
        sentences.append(trailing_sentence)
    return tuple(sentences)


def _sentence_end_positions(text: str, *, source_start: int) -> tuple[int, ...]:
    return tuple(
        index + 1
        for index, character in enumerate(text)
        if index >= source_start and character in SENTENCE_END_CHARACTERS
    )


def _build_sentence(
    text: str,
    alignments: tuple[CharacterAlignment, ...],
    *,
    start: int,
    end: int,
) -> TimedTranscriptionText | None:
    sentence_alignments = [
        item
        for item in alignments
        if item.source_start >= start and item.source_end <= end
    ]
    sentence_text = text[start:end].strip()
    if not sentence_text or not sentence_alignments:
        return None
    return TimedTranscriptionText(
        text=sentence_text,
        start_ms=sentence_alignments[0].start_ms,
        end_ms=sentence_alignments[-1].end_ms,
    )
