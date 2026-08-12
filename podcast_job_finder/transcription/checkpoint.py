from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from podcast_job_finder.audio.segmentation.segment_export import ExportedSpeechSegment
from podcast_job_finder.transcription.checkpoint_store import (
    SegmentTranscriptionCheckpointStore,
)
from podcast_job_finder.transcription.models import (
    AudioTranscriberProtocol,
    AudioTranscriptionResult,
    BatchAudioTranscriberProtocol,
    TranscribedSpeechSegment,
    TranscriptionSegmentResult,
)
from podcast_job_finder.transcription.segments import (
    build_transcribed_speech_segment,
    transcribe_speech_segment,
    validate_previous_segment_order,
)


logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class _DeferredSegmentCheckpoint:
    segment: ExportedSpeechSegment
    transcribed_segment: TranscribedSpeechSegment


@dataclass(slots=True)
class _BatchTranscriptionState:
    transcribed_segments: list[TranscribedSpeechSegment]
    deferred_checkpoints: list[_DeferredSegmentCheckpoint]


def transcribe_speech_segments_with_checkpoints(
    segments: Sequence[ExportedSpeechSegment],
    *,
    transcriber: AudioTranscriberProtocol,
    checkpoint_store: SegmentTranscriptionCheckpointStore,
    resume: bool = False,
) -> tuple[AudioTranscriptionResult, bool]:
    if isinstance(transcriber, BatchAudioTranscriberProtocol):
        return _transcribe_batches_with_checkpoints(
            segments,
            transcriber=transcriber,
            checkpoint_store=checkpoint_store,
            resume=resume,
        )
    return _transcribe_segments_sequentially(
        segments,
        transcriber=transcriber,
        checkpoint_store=checkpoint_store,
        resume=resume,
    )


def _transcribe_segments_sequentially(
    segments: Sequence[ExportedSpeechSegment],
    *,
    transcriber: AudioTranscriberProtocol,
    checkpoint_store: SegmentTranscriptionCheckpointStore,
    resume: bool,
) -> tuple[AudioTranscriptionResult, bool]:
    transcribed_segments: list[TranscribedSpeechSegment] = []
    # 保留上一段的完整结果：下一段既可能用到文字，也可能用到时间信息。
    previous_segment = None
    all_segments_cached = bool(segments)
    for segment in segments:
        validate_previous_segment_order(segment, previous_segment)
        transcription_path = segment.file_path.with_suffix(".json")
        transcribed_segment = None
        if resume:
            transcribed_segment = checkpoint_store.load(
                transcription_path,
                exported_segment=segment,
            )
        if transcribed_segment is None:
            all_segments_cached = False
            transcribed_segment = transcribe_speech_segment(
                segment,
                transcriber=transcriber,
                previous_segment=previous_segment,
            )
            checkpoint_store.save(
                transcription_path,
                exported_segment=segment,
                transcribed_segment=transcribed_segment,
            )
        else:
            logger.info(
                "命中音频片段转写检查点：path=%s index=%d",
                transcription_path,
                segment.index,
            )
        transcribed_segments.append(transcribed_segment)
        previous_segment = transcribed_segment
    return (
        AudioTranscriptionResult(segments=transcribed_segments),
        all_segments_cached,
    )


def _transcribe_batches_with_checkpoints(
    segments: Sequence[ExportedSpeechSegment],
    *,
    transcriber: BatchAudioTranscriberProtocol,
    checkpoint_store: SegmentTranscriptionCheckpointStore,
    resume: bool,
) -> tuple[AudioTranscriptionResult, bool]:
    if transcriber.batch_size <= 0:
        raise ValueError("batch_size 必须大于 0。")
    state = _BatchTranscriptionState([], [])
    if not resume:
        _transcribe_pending_batches(
            segments,
            transcriber=transcriber,
            checkpoint_store=checkpoint_store,
            previous_segment=None,
            state=state,
        )
        _save_deferred_checkpoints(state.deferred_checkpoints, checkpoint_store)
        return AudioTranscriptionResult(segments=state.transcribed_segments), False
    return _transcribe_with_sparse_checkpoints(
        segments,
        transcriber=transcriber,
        checkpoint_store=checkpoint_store,
        state=state,
    )


def _transcribe_with_sparse_checkpoints(
    segments: Sequence[ExportedSpeechSegment],
    *,
    transcriber: BatchAudioTranscriberProtocol,
    checkpoint_store: SegmentTranscriptionCheckpointStore,
    state: _BatchTranscriptionState,
) -> tuple[AudioTranscriptionResult, bool]:
    previous_segment = None
    all_segments_cached = bool(segments)
    index = 0
    while index < len(segments):
        segment = segments[index]
        validate_previous_segment_order(segment, previous_segment)
        cached_segment = _load_cached_segment(
            segment,
            checkpoint_store=checkpoint_store,
        )
        if cached_segment is not None:
            _append_cached_segment(
                segment,
                cached_segment,
                transcribed_segments=state.transcribed_segments,
            )
            previous_segment = cached_segment
            index += 1
            continue

        all_segments_cached = False
        run_end = _find_next_cached_segment(
            segments,
            start=index + 1,
            checkpoint_store=checkpoint_store,
        )
        previous_segment = _transcribe_pending_batches(
            segments[index:run_end],
            transcriber=transcriber,
            checkpoint_store=checkpoint_store,
            previous_segment=previous_segment,
            state=state,
        )
        index = run_end

    _save_deferred_checkpoints(state.deferred_checkpoints, checkpoint_store)
    return AudioTranscriptionResult(
        segments=state.transcribed_segments
    ), all_segments_cached


def _transcribe_pending_batches(
    segments: Sequence[ExportedSpeechSegment],
    *,
    transcriber: BatchAudioTranscriberProtocol,
    checkpoint_store: SegmentTranscriptionCheckpointStore,
    previous_segment: TranscribedSpeechSegment | None,
    state: _BatchTranscriptionState,
) -> TranscribedSpeechSegment | None:
    current_previous = previous_segment
    expected_segments = {
        segment.file_path: (
            position,
            segment,
            segments[position - 1] if position > 0 else None,
        )
        for position, segment in enumerate(segments)
    }
    expected_paths = set(expected_segments)
    processed_paths: set[Path] = set()
    last_processed_position = -1
    for outputs in transcriber.transcribe_batches(
        segments,
        previous_segment=current_previous,
    ):
        if not outputs:
            raise ValueError("批量转写结果数量与音频片段数量不一致。")
        for item in outputs:
            segment = item.segment
            segment_position = _validate_batch_transcription_item(
                item,
                expected_segments=expected_segments,
                processed_paths=processed_paths,
                last_processed_position=last_processed_position,
                current_previous=current_previous,
            )
            transcribed_segment = build_transcribed_speech_segment(
                segment,
                item.output,
            )
            if item.defer_checkpoint:
                state.deferred_checkpoints.append(
                    _DeferredSegmentCheckpoint(segment, transcribed_segment)
                )
            else:
                checkpoint_store.save(
                    segment.file_path.with_suffix(".json"),
                    exported_segment=segment,
                    transcribed_segment=transcribed_segment,
                )
            state.transcribed_segments.append(transcribed_segment)
            processed_paths.add(segment.file_path)
            current_previous = transcribed_segment
            last_processed_position = segment_position
    if processed_paths != expected_paths:
        raise ValueError("批量转写结果数量与音频片段数量不一致。")
    return current_previous


def _save_deferred_checkpoints(
    deferred_checkpoints: Sequence[_DeferredSegmentCheckpoint],
    checkpoint_store: SegmentTranscriptionCheckpointStore,
) -> None:
    for item in deferred_checkpoints:
        logger.info(
            "所有音频片段已处理完成，写入延迟的空转写检查点：index=%d path=%s",
            item.segment.index,
            item.segment.file_path.with_suffix(".json"),
        )
        checkpoint_store.save(
            item.segment.file_path.with_suffix(".json"),
            exported_segment=item.segment,
            transcribed_segment=item.transcribed_segment,
        )


def _validate_batch_transcription_item(
    item: TranscriptionSegmentResult,
    *,
    expected_segments: Mapping[
        Path,
        tuple[int, ExportedSpeechSegment, ExportedSpeechSegment | None],
    ],
    processed_paths: set[Path],
    last_processed_position: int,
    current_previous: TranscribedSpeechSegment | None,
) -> int:
    segment = item.segment
    if segment.file_path not in expected_segments:
        raise ValueError("批量转写结果包含未请求的音频片段。")
    if segment.file_path in processed_paths:
        raise ValueError("批量转写结果包含重复的音频片段。")
    segment_position, expected_segment, expected_previous = expected_segments[
        segment.file_path
    ]
    if segment != expected_segment:
        raise ValueError("批量转写结果与请求的音频片段不一致。")
    if segment_position <= last_processed_position:
        raise ValueError(f"批量转写结果顺序错误：actual_index={segment.index}")

    validate_previous_segment_order(segment, item.previous_segment)
    if segment_position == last_processed_position + 1:
        if item.previous_segment != current_previous:
            raise ValueError(f"批量转写结果使用了错误的上一片段：index={segment.index}")
        return segment_position

    if (
        expected_previous is None
        or item.previous_segment is None
        or item.previous_segment.index != expected_previous.index
        or item.previous_segment.start_ms != expected_previous.segment.start_ms
        or item.previous_segment.end_ms != expected_previous.segment.end_ms
    ):
        raise ValueError(f"批量转写结果使用了错误的上一片段：index={segment.index}")
    return segment_position


def _find_next_cached_segment(
    segments: Sequence[ExportedSpeechSegment],
    *,
    start: int,
    checkpoint_store: SegmentTranscriptionCheckpointStore,
) -> int:
    for index in range(start, len(segments)):
        if (
            _load_cached_segment(
                segments[index],
                checkpoint_store=checkpoint_store,
            )
            is not None
        ):
            return index
    return len(segments)


def _load_cached_segment(
    segment: ExportedSpeechSegment,
    *,
    checkpoint_store: SegmentTranscriptionCheckpointStore,
) -> TranscribedSpeechSegment | None:
    return checkpoint_store.load(
        segment.file_path.with_suffix(".json"),
        exported_segment=segment,
    )


def _append_cached_segment(
    segment: ExportedSpeechSegment,
    cached_segment: TranscribedSpeechSegment,
    *,
    transcribed_segments: list[TranscribedSpeechSegment],
) -> None:
    logger.info(
        "命中音频片段转写检查点：path=%s index=%d",
        segment.file_path.with_suffix(".json"),
        segment.index,
    )
    transcribed_segments.append(cached_segment)
