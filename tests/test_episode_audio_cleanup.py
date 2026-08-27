import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from podcast_job_finder.audio.episode_audio.errors import EpisodeAudioCleanupError
from podcast_job_finder.audio.episode_audio.files import delete_episode_audio_directory
from podcast_job_finder.cli.companies import (
    DELETE_AUDIO_SOURCE_ERROR,
    CliUsageError,
    _run_feed_command,
)
from podcast_job_finder.episode.models import EpisodeWorkItem
from podcast_job_finder.output_paths import (
    COMPANY_EXTRACTION_DIR_NAME,
    EPISODE_AUDIO_DIR_NAME,
    EPISODE_TRANSCRIPTION_DIR_NAME,
    build_episode_output_dir,
)
from podcast_job_finder.transcription.batch import delete_completed_episode_audio_files
from podcast_job_finder.transcription.formatting.article import (
    TRANSCRIPTION_ARTICLE_FILE_NAME,
    build_transcription_article,
)
from podcast_job_finder.transcription.manifest import TRANSCRIPTION_FILE_NAME
from podcast_job_finder.transcription.quality_report import (
    TRANSCRIPTION_QUALITY_REPORT_FILE_NAME,
)


class DeleteEpisodeAudioDirectoryTest(TestCase):
    def test_missing_directory_returns_false(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            audio_dir = Path(temporary_dir) / "audio"
            self.assertFalse(delete_episode_audio_directory(audio_dir))

    def test_symlink_directory_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            real_dir = output_dir / "real-audio"
            real_dir.mkdir()
            (real_dir / "source.m4a").write_bytes(b"abc")
            audio_dir = output_dir / "audio"
            audio_dir.symlink_to(real_dir)
            with self.assertRaises(EpisodeAudioCleanupError):
                delete_episode_audio_directory(audio_dir)
            self.assertTrue((real_dir / "source.m4a").exists())


class DeleteCompletedEpisodeAudioFilesTest(TestCase):
    def test_deletes_completed_episode_audio_and_keeps_other_files(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            completed_item = _build_work_item("complete-eid", "已完成节目")
            pending_item = _build_work_item("pending-eid", "未完成节目")
            completed_dir = _create_episode_dir(output_dir, completed_item)
            pending_dir = _create_episode_dir(output_dir, pending_item)
            _write_completed_transcription(completed_dir, completed_item)
            _write_audio_files(completed_dir)
            _write_audio_files(pending_dir)
            company_dir = completed_dir / COMPANY_EXTRACTION_DIR_NAME
            company_dir.mkdir()
            (company_dir / "keep.json").write_text("{}", encoding="utf-8")

            deleted_count = delete_completed_episode_audio_files(
                [completed_item, pending_item],
                audio_output_dir=output_dir,
            )

            self.assertEqual(deleted_count, 1)
            self.assertFalse((completed_dir / EPISODE_AUDIO_DIR_NAME).exists())
            self.assertTrue(
                (
                    completed_dir
                    / EPISODE_TRANSCRIPTION_DIR_NAME
                    / TRANSCRIPTION_FILE_NAME
                ).exists()
            )
            self.assertTrue((company_dir / "keep.json").exists())
            self.assertTrue(
                (pending_dir / EPISODE_AUDIO_DIR_NAME / "source.m4a").exists()
            )


class DeleteAudioCliTest(TestCase):
    def test_delete_audio_requires_audio_source(self) -> None:
        with self.assertRaises(CliUsageError) as error:
            _run_feed_command(
                [
                    "--feed-url",
                    "https://example.com/rss",
                    "--source",
                    "page",
                    "--delete-audio",
                ]
            )
        self.assertEqual(str(error.exception), DELETE_AUDIO_SOURCE_ERROR)


def _build_work_item(eid: str, title: str) -> EpisodeWorkItem:
    return EpisodeWorkItem(
        episode_url=f"https://example.com/episode/{eid}",
        eid=eid,
        title=title,
        podcast_title="示例播客",
    )


def _create_episode_dir(output_dir: Path, work_item: EpisodeWorkItem) -> Path:
    episode_dir = build_episode_output_dir(
        output_dir,
        work_item.eid or "",
        podcast_title=work_item.podcast_title,
        episode_title=work_item.title,
    )
    episode_dir.mkdir(parents=True)
    return episode_dir


def _write_audio_files(episode_dir: Path) -> None:
    audio_dir = episode_dir / EPISODE_AUDIO_DIR_NAME
    segment_dir = audio_dir / "segments"
    segment_dir.mkdir(parents=True)
    (audio_dir / "source.m4a").write_bytes(b"source-audio")
    (segment_dir / "segment-001.wav").write_bytes(b"segment-audio")


def _write_completed_transcription(
    episode_dir: Path,
    work_item: EpisodeWorkItem,
) -> None:
    transcription_dir = episode_dir / EPISODE_TRANSCRIPTION_DIR_NAME
    transcription_dir.mkdir(parents=True)
    title = work_item.title or work_item.eid or ""
    body = "你好"
    transcription_path = transcription_dir / TRANSCRIPTION_FILE_NAME
    transcription_path.write_text(
        json.dumps(
            {
                "eid": work_item.eid,
                "episode_url": work_item.episode_url,
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
    (transcription_dir / TRANSCRIPTION_ARTICLE_FILE_NAME).write_text(
        build_transcription_article(title=title, body=body),
        encoding="utf-8",
    )
    (transcription_dir / TRANSCRIPTION_QUALITY_REPORT_FILE_NAME).write_text(
        json.dumps({"segment_count": 1}, ensure_ascii=False),
        encoding="utf-8",
    )
