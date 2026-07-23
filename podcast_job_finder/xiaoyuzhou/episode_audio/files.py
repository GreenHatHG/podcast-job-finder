from __future__ import annotations

import fcntl
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Final, Iterator

from podcast_job_finder.xiaoyuzhou.episode_audio.errors import (
    EpisodeAudioDownloadError,
)
from podcast_job_finder.xiaoyuzhou.episode_audio.http import download_audio_content


SOURCE_FILE_STEM: Final = "source"
PARTIAL_FILE_PREFIX: Final = ".source."
PARTIAL_FILE_SUFFIX: Final = ".part"
DOWNLOAD_LOCK_FILE_NAME: Final = ".download.lock"
EMPTY_AUDIO_ERROR: Final = "下载到的节目音频为空：{url}"
EXISTING_AUDIO_SYMLINK_ERROR: Final = "目标音频文件是符号链接，已拒绝操作：{path}"
EXISTING_AUDIO_NOT_FILE_ERROR: Final = "目标音频路径不是普通文件：{path}"
INSPECT_EXISTING_AUDIO_ERROR_TEMPLATE: Final = (
    "检查或清理已有音频文件失败：{path}，{error_message}"
)
AUDIO_PUBLISH_CONFLICT_ERROR: Final = (
    "发布节目音频时目标路径被其他进程占用或替换：{path}"
)
EPISODE_DIR_SYMLINK_ERROR: Final = "节目音频目录是符号链接，已拒绝操作：{path}"
EPISODE_DIR_REDIRECT_ERROR_TEMPLATE: Final = (
    "节目音频目录真实位置异常：期望 {expected_path}，实际 {actual_path}"
)
PREPARE_OUTPUT_DIR_ERROR_TEMPLATE: Final = (
    "创建或解析节目音频输出目录失败：{path}，{error_message}"
)
OPEN_DOWNLOAD_LOCK_ERROR_TEMPLATE: Final = (
    "打开节目音频锁文件失败：{path}，{error_message}"
)
ACQUIRE_DOWNLOAD_LOCK_ERROR_TEMPLATE: Final = (
    "获取节目音频文件锁失败：{path}，{error_message}"
)
CLOSE_DOWNLOAD_LOCK_ERROR_TEMPLATE: Final = (
    "关闭节目音频锁文件失败：{path}，{error_message}"
)
CREATE_PARTIAL_AUDIO_ERROR_TEMPLATE: Final = (
    "创建节目音频临时文件失败：{path}，{error_message}"
)
CLOSE_PARTIAL_AUDIO_ERROR_TEMPLATE: Final = (
    "关闭节目音频临时文件失败：{path}，{error_message}"
)
PUBLISH_AUDIO_ERROR_TEMPLATE: Final = "发布节目音频文件失败：{path}，{error_message}"
REMOVE_PARTIAL_AUDIO_ERROR_TEMPLATE: Final = (
    "清理节目音频临时文件失败：{path}，{error_message}"
)


def build_audio_target_path(output_dir: Path, eid: str, extension: str) -> Path:
    episode_dir = prepare_episode_audio_directory(output_dir, eid)
    return episode_dir / f"{SOURCE_FILE_STEM}{extension}"


def prepare_episode_audio_directory(output_dir: Path, eid: str) -> Path:
    try:
        resolved_output_dir = output_dir.resolve()
        episode_dir = resolved_output_dir / eid
        if episode_dir.is_symlink():
            raise EpisodeAudioDownloadError(
                EPISODE_DIR_SYMLINK_ERROR.format(path=episode_dir)
            )
        episode_dir.mkdir(parents=True, exist_ok=True)
        if episode_dir.is_symlink():
            raise EpisodeAudioDownloadError(
                EPISODE_DIR_SYMLINK_ERROR.format(path=episode_dir)
            )
        actual_episode_dir = episode_dir.resolve()
        if actual_episode_dir != episode_dir:
            raise EpisodeAudioDownloadError(
                EPISODE_DIR_REDIRECT_ERROR_TEMPLATE.format(
                    expected_path=episode_dir,
                    actual_path=actual_episode_dir,
                )
            )
        return episode_dir
    except OSError as error:
        raise EpisodeAudioDownloadError(
            PREPARE_OUTPUT_DIR_ERROR_TEMPLATE.format(
                path=output_dir,
                error_message=str(error),
            )
        ) from error


def store_episode_audio(
    source_url: str,
    target_path: Path,
    *,
    overwrite: bool,
) -> bool:
    with _acquire_download_lock(target_path.parent):
        if _should_skip_existing_file(target_path, overwrite=overwrite):
            return True
        return _download_audio_file(
            source_url,
            target_path,
            overwrite=overwrite,
        )


@contextmanager
def _acquire_download_lock(episode_dir: Path) -> Iterator[None]:
    lock_path = episode_dir / DOWNLOAD_LOCK_FILE_NAME
    open_flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        open_flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        open_flags |= os.O_NOFOLLOW

    file_descriptor = _open_lock_file(lock_path, open_flags)
    try:
        lock_file = os.fdopen(file_descriptor, "a+b")
    except BaseException as error:  # pylint: disable=broad-exception-caught
        if isinstance(error, OSError):
            download_error = EpisodeAudioDownloadError(
                OPEN_DOWNLOAD_LOCK_ERROR_TEMPLATE.format(
                    path=lock_path,
                    error_message=str(error),
                )
            )
            _close_file_descriptor(
                file_descriptor,
                lock_path,
                download_error,
                CLOSE_DOWNLOAD_LOCK_ERROR_TEMPLATE,
            )
            raise download_error from error
        _close_file_descriptor(
            file_descriptor,
            lock_path,
            error,
            CLOSE_DOWNLOAD_LOCK_ERROR_TEMPLATE,
        )
        raise

    active_error: BaseException | None = None
    try:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        except OSError as error:
            raise EpisodeAudioDownloadError(
                ACQUIRE_DOWNLOAD_LOCK_ERROR_TEMPLATE.format(
                    path=lock_path,
                    error_message=str(error),
                )
            ) from error
        yield
    except BaseException as error:  # pylint: disable=broad-exception-caught
        active_error = error
        raise
    finally:
        _close_file(
            lock_file,
            lock_path,
            active_error,
            CLOSE_DOWNLOAD_LOCK_ERROR_TEMPLATE,
        )


def _open_lock_file(lock_path: Path, open_flags: int) -> int:
    try:
        return os.open(lock_path, open_flags, 0o600)
    except OSError as error:
        raise EpisodeAudioDownloadError(
            OPEN_DOWNLOAD_LOCK_ERROR_TEMPLATE.format(
                path=lock_path,
                error_message=str(error),
            )
        ) from error


def _should_skip_existing_file(target_path: Path, *, overwrite: bool) -> bool:
    try:
        if target_path.is_symlink():
            raise EpisodeAudioDownloadError(
                EXISTING_AUDIO_SYMLINK_ERROR.format(path=target_path)
            )
        if not target_path.exists():
            return False
        if not target_path.is_file():
            raise EpisodeAudioDownloadError(
                EXISTING_AUDIO_NOT_FILE_ERROR.format(path=target_path)
            )
        if overwrite:
            return False
        if target_path.stat().st_size > 0:
            return True

        target_path.unlink()
        return False
    except OSError as error:
        raise EpisodeAudioDownloadError(
            INSPECT_EXISTING_AUDIO_ERROR_TEMPLATE.format(
                path=target_path,
                error_message=str(error),
            )
        ) from error


def _download_audio_file(
    source_url: str,
    target_path: Path,
    *,
    overwrite: bool,
) -> bool:
    partial_path: Path | None = None
    active_error: BaseException | None = None
    try:
        partial_path, partial_file = _create_partial_file(target_path.parent)
        downloaded_bytes = _write_to_partial_file(
            source_url,
            partial_path,
            partial_file,
        )
        if downloaded_bytes == 0:
            raise EpisodeAudioDownloadError(EMPTY_AUDIO_ERROR.format(url=source_url))
        try:
            return _publish_downloaded_file(
                partial_path,
                target_path,
                overwrite=overwrite,
            )
        except OSError as error:
            raise EpisodeAudioDownloadError(
                PUBLISH_AUDIO_ERROR_TEMPLATE.format(
                    path=target_path,
                    error_message=str(error),
                )
            ) from error
    except BaseException as error:  # pylint: disable=broad-exception-caught
        active_error = error
        raise
    finally:
        if partial_path is not None:
            _remove_partial_file(partial_path, active_error)


def _create_partial_file(parent_dir: Path) -> tuple[Path, BinaryIO]:
    file_descriptor: int | None = None
    partial_path: Path | None = None
    try:
        file_descriptor, partial_file_name = tempfile.mkstemp(
            dir=parent_dir,
            prefix=PARTIAL_FILE_PREFIX,
            suffix=PARTIAL_FILE_SUFFIX,
        )
        partial_path = Path(partial_file_name)
        return partial_path, os.fdopen(file_descriptor, "wb")
    except BaseException as error:  # pylint: disable=broad-exception-caught
        active_error = _build_partial_file_creation_error(parent_dir, error)
        if file_descriptor is not None:
            _close_file_descriptor(
                file_descriptor,
                partial_path or parent_dir,
                active_error,
                CLOSE_PARTIAL_AUDIO_ERROR_TEMPLATE,
            )
        if partial_path is not None:
            _remove_partial_file(partial_path, active_error)
        if active_error is error:
            raise
        raise active_error from error


def _build_partial_file_creation_error(
    parent_dir: Path,
    error: BaseException,
) -> BaseException:
    if not isinstance(error, OSError):
        return error
    return EpisodeAudioDownloadError(
        CREATE_PARTIAL_AUDIO_ERROR_TEMPLATE.format(
            path=parent_dir,
            error_message=str(error),
        )
    )


def _write_to_partial_file(
    source_url: str,
    partial_path: Path,
    partial_file: BinaryIO,
) -> int:
    active_error: BaseException | None = None
    try:
        return download_audio_content(source_url, partial_path, partial_file)
    except BaseException as error:  # pylint: disable=broad-exception-caught
        active_error = error
        raise
    finally:
        _close_file(
            partial_file,
            partial_path,
            active_error,
            CLOSE_PARTIAL_AUDIO_ERROR_TEMPLATE,
        )


def _close_file(
    file_obj: BinaryIO,
    path: Path,
    active_error: BaseException | None,
    error_template: str,
) -> None:
    try:
        file_obj.close()
    except OSError as error:
        message = error_template.format(path=path, error_message=str(error))
        if active_error is not None:
            active_error.add_note(message)
            return
        raise EpisodeAudioDownloadError(message) from error


def _close_file_descriptor(
    file_descriptor: int,
    path: Path,
    active_error: BaseException,
    error_template: str,
) -> None:
    try:
        os.close(file_descriptor)
    except OSError as error:
        active_error.add_note(
            error_template.format(path=path, error_message=str(error))
        )


def _publish_downloaded_file(
    partial_path: Path,
    target_path: Path,
    *,
    overwrite: bool,
) -> bool:
    if overwrite:
        partial_path.replace(target_path)
        return False

    try:
        os.link(partial_path, target_path)
    except FileExistsError:
        if _is_non_empty_regular_file(target_path):
            return True
        raise EpisodeAudioDownloadError(
            AUDIO_PUBLISH_CONFLICT_ERROR.format(path=target_path)
        ) from None
    return False


def _is_non_empty_regular_file(path: Path) -> bool:
    return not path.is_symlink() and path.is_file() and path.stat().st_size > 0


def _remove_partial_file(
    partial_path: Path,
    active_error: BaseException | None = None,
) -> None:
    try:
        partial_path.unlink(missing_ok=True)
    except OSError as error:
        message = REMOVE_PARTIAL_AUDIO_ERROR_TEMPLATE.format(
            path=partial_path,
            error_message=str(error),
        )
        if active_error is not None:
            active_error.add_note(message)
            return
        raise EpisodeAudioDownloadError(message) from error
