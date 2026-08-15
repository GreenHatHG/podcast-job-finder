"""音频转写流水线的共享结果类型和状态。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


RESULT_STATUS_SUCCESS: Final = "success"
RESULT_STATUS_ERROR: Final = "error"


@dataclass(slots=True, frozen=True)
class BatchAudioTranscriptionResult:
    """保存一次批量音频转写产生的节目记录和数量统计。

    普通音频模式创建的结果包含本次选中的全部节目：转写成功和失败的节目都会
    保留一条记录。失败记录只用于保存失败原因，不代表该节目已经得到转写文本。

    ``--extract-only`` 模式从已有文件创建结果，只加入存在 ``transcription.json``
    的节目；缺少该文件的节目在创建对象前已经跳过。这个对象不保存 RSS feed
    本身，也不记录这些被提前跳过的节目。
    """

    # 每个字典对应一个节目，status 表示转写是否成功；成功记录还包含
    # transcription_path，供后续流程读取该节目的转写文本。
    episode_results: list[dict[str, object]]
    # episode_results 中 status 为 success 的记录数量。
    success_count: int
    # episode_results 中除 success 以外的记录数量，不包括创建对象前跳过的节目。
    fail_count: int
