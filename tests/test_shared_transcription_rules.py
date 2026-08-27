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
from podcast_job_finder.transcription.completed_transcription import (
    can_restore_completed_transcription,
)
from podcast_job_finder.transcription.formatting.article import (
    build_transcription_article,
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

    def test_checkpoint_loader_accepts_different_audio_path(self) -> None:
        exported_segment = ExportedSpeechSegment(
            index=1,
            segment=SpeechSegment(start_sample=16_000, end_sample=48_000),
            file_path=Path("/vps/segment.wav"),
        )
        payload = {
            "eid": "abc",
            "episode_url": "https://example.com/episode",
            "index": 1,
            "start_ms": 1_000,
            "end_ms": 3_000,
            "audio_path": "/Users/old/segment.wav",
            "text": "测试。",
        }
        checkpoint_store = SegmentTranscriptionCheckpointStore(
            metadata={"eid": "abc"},
            expected_metadata={
                "eid": "abc",
                "episode_url": "https://example.com/episode",
            },
        )
        with TemporaryDirectory() as temporary_dir:
            checkpoint_path = Path(temporary_dir) / "segment.json"
            checkpoint_path.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            restored_segment = checkpoint_store.load(
                checkpoint_path,
                exported_segment=exported_segment,
            )

        assert restored_segment is not None
        self.assertEqual(restored_segment.text, "测试。")
        self.assertEqual(restored_segment.start_ms, 1_000)


class CompletedTranscriptionResumeTest(TestCase):
    def test_restore_succeeds_without_local_audio_files(self) -> None:
        title = "节目标题"
        body = "你好"
        with TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            transcription_path = output_dir / "transcription.json"
            article_path = output_dir / "transcription.md"
            quality_report_path = output_dir / "transcription_quality_report.json"
            transcription_path.write_text(
                json.dumps(
                    {
                        "eid": "abc",
                        "episode_url": "https://example.com/episode",
                        "title": title,
                        "audio_path": "/Users/old/source.m4a",
                        "segment_count": 1,
                        "text": body,
                        "segments": [
                            {
                                "index": 1,
                                "start_ms": 0,
                                "end_ms": 1000,
                                "text": body,
                                "audio_path": "/Users/old/segment.wav",
                                "transcription_path": "/Users/old/segment.json",
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            article_path.write_text(
                build_transcription_article(title=title, body=body),
                encoding="utf-8",
            )
            quality_report_path.write_text(
                json.dumps({"segment_count": 1}, ensure_ascii=False),
                encoding="utf-8",
            )
            restored = can_restore_completed_transcription(
                transcription_path=transcription_path,
                article_path=article_path,
                quality_report_path=quality_report_path,
                article_title=title,
                expected_metadata={
                    "eid": "abc",
                    "episode_url": "https://example.com/episode",
                },
            )
        self.assertTrue(restored)

    def test_restore_rejects_episode_metadata_mismatch(self) -> None:
        title = "节目标题"
        body = "你好"
        with TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            transcription_path = output_dir / "transcription.json"
            article_path = output_dir / "transcription.md"
            quality_report_path = output_dir / "transcription_quality_report.json"
            transcription_path.write_text(
                json.dumps(
                    {
                        "eid": "abc",
                        "episode_url": "https://example.com/episode",
                        "title": title,
                        "segment_count": 1,
                        "text": body,
                        "segments": [
                            {
                                "index": 1,
                                "start_ms": 0,
                                "end_ms": 1000,
                                "text": body,
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            article_path.write_text(
                build_transcription_article(title=title, body=body),
                encoding="utf-8",
            )
            quality_report_path.write_text(
                json.dumps({"segment_count": 1}, ensure_ascii=False),
                encoding="utf-8",
            )
            restored = can_restore_completed_transcription(
                transcription_path=transcription_path,
                article_path=article_path,
                quality_report_path=quality_report_path,
                article_title=title,
                expected_metadata={
                    "eid": "abc",
                    "episode_url": "https://example.com/other",
                },
            )
        self.assertFalse(restored)
