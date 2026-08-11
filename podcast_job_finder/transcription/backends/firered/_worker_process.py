from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Sequence
from typing import IO, Final


WORKER_READY_STATUS: Final = "ready"
WORKER_RESULT_STATUS: Final = "result"
WORKER_SHUTDOWN_COMMAND: Final = "shutdown"
WORKER_CLOSE_TIMEOUT_SECONDS: Final = 5


class WorkerExitedError(RuntimeError):
    def __init__(self, returncode: int | None) -> None:
        super().__init__(f"进程退出：{returncode}")
        self.returncode = returncode


class WorkerResponseError(RuntimeError):
    def __init__(self, response: object) -> None:
        super().__init__(str(response))
        self.response = response


class JsonLineWorkerProcess:
    """管理使用标准输入输出交换 JSON 行的 FireRed 工作进程。"""

    def __init__(
        self,
        *,
        command: Sequence[str],
        stdin_unavailable_error: str,
    ) -> None:
        self._command = tuple(command)
        self._stdin_unavailable_error = stdin_unavailable_error
        self._process: subprocess.Popen[str] | None = None

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def returncode(self) -> int | None:
        if self._process is None:
            return None
        return self._process.poll()

    def start(self) -> None:
        environment = os.environ.copy()
        environment.setdefault("TOKENIZERS_PARALLELISM", "false")
        process = subprocess.Popen(  # pylint: disable=consider-using-with
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=environment,
        )
        self._process = process

    def write_request(self, payload: dict[str, object]) -> None:
        process = self._require_process()
        _write_request(
            process.stdin,
            payload,
            stdin_unavailable_error=self._stdin_unavailable_error,
        )

    def read_response(self) -> dict[str, object]:
        return _read_response(self._require_process())

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        try:
            _write_request(
                process.stdin,
                {"command": WORKER_SHUTDOWN_COMMAND},
                stdin_unavailable_error=self._stdin_unavailable_error,
            )
            process.wait(timeout=WORKER_CLOSE_TIMEOUT_SECONDS)
        except BrokenPipeError, OSError, subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=WORKER_CLOSE_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def _require_process(self) -> subprocess.Popen[str]:
        process = self._process
        if process is None:
            raise WorkerExitedError(None)
        return process


def _write_request(
    stream: IO[str] | None,
    payload: dict[str, object],
    *,
    stdin_unavailable_error: str,
) -> None:
    if stream is None:
        raise BrokenPipeError(stdin_unavailable_error)
    stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
    stream.flush()


def _read_response(process: subprocess.Popen[str]) -> dict[str, object]:
    if process.stdout is None:
        raise WorkerResponseError("标准输出不可用")
    response_line = process.stdout.readline()
    if not response_line:
        raise WorkerExitedError(process.poll())
    try:
        response = json.loads(response_line)
    except json.JSONDecodeError as error:
        raise WorkerResponseError(response_line.strip()) from error
    if not isinstance(response, dict):
        raise WorkerResponseError(response)
    return response
