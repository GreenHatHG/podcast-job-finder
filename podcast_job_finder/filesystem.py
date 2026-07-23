from __future__ import annotations

import json
import os
import secrets
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Final, Iterator


DEFAULT_TEXT_ENCODING: Final = "utf-8"
# 普通输出文件不允许执行。
# 常见结果是文件所有者可读写，其他用户只读。
# 系统可以根据自身安全设置继续限制哪些用户能够读写。
DEFAULT_FILE_CREATION_MODE: Final = 0o666
# 登录凭据等敏感文件仅允许文件所有者读写。
# 系统还可以进一步限制。
OWNER_READ_WRITE_MODE: Final = 0o600
TEMPORARY_FILE_NAME_TEMPLATE: Final = ".{target_name}.{random_token}.tmp"
TEMPORARY_FILE_RANDOM_BYTES: Final = 16
TEMPORARY_FILE_CREATION_ATTEMPTS: Final = 100
CREATE_TEMPORARY_FILE_ERROR_TEMPLATE: Final = "无法创建唯一临时文件：{target_path}"
REMOVE_TEMPORARY_FILE_ERROR_TEMPLATE: Final = (
    "清理临时文件失败：{path}，{error_message}"
)


class AtomicWriteConflictError(FileExistsError):
    """禁止覆盖时目标文件已存在。"""


class TemporaryFileError(OSError):
    """临时文件无法创建、关闭或清理。"""


def atomic_write_file(
    target_path: Path,
    *,
    write: Callable[[Path], None],
    overwrite: bool,
    mode: int,
) -> None:
    """写完同级临时文件后，按覆盖策略原子发布到目标路径。"""
    with temporary_sibling_path(target_path, mode=mode) as temporary_path:
        write(temporary_path)
        _publish_temporary_file(
            temporary_path,
            target_path,
            overwrite=overwrite,
        )


def atomic_write_text(
    target_path: Path,
    content: str,
    *,
    encoding: str = DEFAULT_TEXT_ENCODING,
    mode: int,
) -> None:
    """完整写入文本后，原子替换目标文件。"""

    def write_text(temporary_path: Path) -> None:
        temporary_path.write_text(content, encoding=encoding)

    atomic_write_file(
        target_path,
        write=write_text,
        overwrite=True,
        mode=mode,
    )


def atomic_write_json(
    target_path: Path,
    payload: object,
    *,
    mode: int,
) -> None:
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    atomic_write_text(target_path, content, mode=mode)


@contextmanager
def temporary_sibling_path(
    target_path: Path,
    *,
    mode: int,
    error_factory: Callable[[str], BaseException] = TemporaryFileError,
) -> Iterator[Path]:
    """创建与目标文件位于同一目录的唯一临时文件，并在使用后清理。"""
    try:
        temporary_path = _create_temporary_sibling_path(target_path, mode=mode)
    except OSError as error:
        raise error_factory(str(error)) from error

    active_error: BaseException | None = None
    try:
        yield temporary_path
    except BaseException as error:  # pylint: disable=broad-exception-caught
        active_error = error
        raise
    finally:
        try:
            _remove_temporary_file(temporary_path, active_error)
        except OSError as error:
            raise error_factory(str(error)) from error


def _publish_temporary_file(
    temporary_path: Path,
    target_path: Path,
    *,
    overwrite: bool,
) -> None:
    if overwrite:
        temporary_path.replace(target_path)
        return
    try:
        os.link(temporary_path, target_path)
    except FileExistsError:
        raise AtomicWriteConflictError(target_path) from None


def _create_temporary_sibling_path(target_path: Path, *, mode: int) -> Path:
    file_descriptor, temporary_path = _open_temporary_sibling_file(
        target_path,
        mode=mode,
    )
    try:
        os.close(file_descriptor)
    except BaseException as error:  # pylint: disable=broad-exception-caught
        _remove_temporary_file(temporary_path, error)
        raise
    return temporary_path


def _open_temporary_sibling_file(
    target_path: Path,
    *,
    mode: int,
) -> tuple[int, Path]:
    open_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        open_flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        open_flags |= os.O_NOFOLLOW

    for _ in range(TEMPORARY_FILE_CREATION_ATTEMPTS):
        temporary_path = target_path.with_name(
            TEMPORARY_FILE_NAME_TEMPLATE.format(
                target_name=target_path.name,
                random_token=secrets.token_hex(TEMPORARY_FILE_RANDOM_BYTES),
            )
        )
        try:
            return os.open(temporary_path, open_flags, mode), temporary_path
        except FileExistsError:
            continue
    raise FileExistsError(
        CREATE_TEMPORARY_FILE_ERROR_TEMPLATE.format(target_path=target_path)
    )


def _remove_temporary_file(
    temporary_path: Path,
    active_error: BaseException | None,
) -> None:
    try:
        temporary_path.unlink(missing_ok=True)
    except OSError as error:
        message = REMOVE_TEMPORARY_FILE_ERROR_TEMPLATE.format(
            path=temporary_path,
            error_message=str(error),
        )
        if active_error is not None:
            active_error.add_note(message)
            return
        raise
