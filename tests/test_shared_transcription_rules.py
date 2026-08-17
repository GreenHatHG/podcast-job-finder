import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from podcast_job_finder.audio.segmentation.segment_export import (
    ExportedSpeechSegment,
    to_source_audio_timestamp,
)
from podcast_job_finder.audio.segmentation.vad import SpeechSegment
from podcast_job_finder.companies.models import CompanyExtractionResult
from podcast_job_finder.companies.pipeline_results import (
    FailedCompanyEpisodeResult,
    SuccessfulCompanyEpisodeResult,
)
from podcast_job_finder.episode.models import EpisodeWorkItem
from podcast_job_finder.transcription.checkpoint_store import (
    SegmentTranscriptionCheckpointStore,
)
from podcast_job_finder.transcription.manifest import (
    TranscriptionManifestError,
    parse_transcribed_segment,
)
from podcast_job_finder.transcription.pipeline_results import (
    SuccessfulEpisodeTranscriptionResult,
)


class SharedTranscriptionRulesTest(TestCase):
    def test_manifest_parser_restores_optional_timestamp_fields(self) -> None:
        payload = {
            "character_timestamps": [
                {"text": "测", "start_ms": 100, "end_ms": 200, "confidence": 0.8}
            ],
            "sentences": [{"text": "测试。", "start_ms": 100, "end_ms": 400}],
        }

        segment = parse_transcribed_segment(
            {
                "index": 1,
                "start_ms": 100,
                "end_ms": 400,
                "text": "测试。",
                **payload,
            },
            path=Path("transcription.json"),
            index=0,
        )

        self.assertEqual(
            segment.to_dict(),
            {
                "index": 1,
                "start_ms": 100,
                "end_ms": 400,
                "text": "测试。",
                "character_timestamps": [
                    {
                        "text": "测",
                        "start_ms": 100,
                        "end_ms": 200,
                        "confidence": 0.8,
                    }
                ],
                "sentences": [{"text": "测试。", "start_ms": 100, "end_ms": 400}],
            },
        )

    def test_manifest_parser_converts_optional_field_error(self) -> None:
        with self.assertRaises(TranscriptionManifestError):
            parse_transcribed_segment(
                {
                    "index": 1,
                    "start_ms": 100,
                    "end_ms": 400,
                    "text": "测试。",
                    "sentences": "不是数组",
                },
                path=Path("transcription.json"),
                index=0,
            )

    def test_checkpoint_loader_ignores_invalid_optional_field(self) -> None:
        exported_segment = ExportedSpeechSegment(
            index=1,
            segment=SpeechSegment(start_sample=16_000, end_sample=48_000),
            file_path=Path("segment.wav"),
        )
        payload = {
            "index": 1,
            "start_ms": 1_000,
            "end_ms": 3_000,
            "audio_path": "segment.wav",
            "text": "测试。",
            "sentences": "不是数组",
        }
        checkpoint_store = SegmentTranscriptionCheckpointStore(
            metadata={},
            expected_metadata={},
        )
        with TemporaryDirectory() as temporary_dir:
            checkpoint_path = Path(temporary_dir) / "segment.json"
            checkpoint_path.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            with self.assertLogs(
                "podcast_job_finder.transcription.checkpoint",
                level="WARNING",
            ):
                restored_segment = checkpoint_store.load(
                    checkpoint_path,
                    exported_segment=exported_segment,
                )

        self.assertIsNone(restored_segment)

    def test_source_timestamp_removes_padding_and_clamps_to_segment(self) -> None:
        segment = ExportedSpeechSegment(
            index=1,
            segment=SpeechSegment(start_sample=16_000, end_sample=48_000),
            file_path=Path("segment.wav"),
        )

        self.assertEqual(
            to_source_audio_timestamp(
                250,
                segment=segment,
                silence_padding_ms=500,
            ),
            1_000,
        )
        self.assertEqual(
            to_source_audio_timestamp(
                1_750,
                segment=segment,
                silence_padding_ms=500,
            ),
            2_250,
        )
        self.assertEqual(
            to_source_audio_timestamp(
                5_000,
                segment=segment,
                silence_padding_ms=500,
            ),
            3_000,
        )

    def test_company_failure_keeps_transcription_fields_and_overrides_status(
        self,
    ) -> None:
        episode = EpisodeWorkItem(episode_url="https://example.com/episode", eid="1")
        transcription_result = SuccessfulEpisodeTranscriptionResult(
            episode=episode,
            cached=True,
            episode_output_dir="output/1",
            transcription_path="output/1/transcription.json",
        )

        result = FailedCompanyEpisodeResult(
            episode=episode,
            error="提取失败",
            transcription_result=transcription_result,
        )

        self.assertEqual(result.to_dict()["status"], "error")
        self.assertEqual(result.to_dict()["cached"], True)
        self.assertEqual(result.to_dict()["error"], "提取失败")

    def test_company_success_without_transcription_keeps_episode_fields(self) -> None:
        episode = EpisodeWorkItem(episode_url="https://example.com/episode", eid="1")
        result = SuccessfulCompanyEpisodeResult(
            episode=episode,
            extraction_result=CompanyExtractionResult(),
        )

        self.assertEqual(
            result.to_dict(),
            {
                "eid": "1",
                "podcast_title": None,
                "title": None,
                "pub_date": None,
                "episode_url": "https://example.com/episode",
                "status": "success",
                "companies": [],
                "filtered_count": 0,
            },
        )
