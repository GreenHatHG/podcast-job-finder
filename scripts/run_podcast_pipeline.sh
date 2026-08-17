#!/usr/bin/env bash

# 按“下载全部音频 -> 转写全部音频 -> 提取公司”的顺序处理一档播客。
# 用法：./scripts/run_podcast_pipeline.sh <播客名或RSS地址>
# 播客名来自项目根目录的 podcasts.toml；也可以继续直接传 RSS 地址。请在项目
# 根目录运行脚本，脚本会从当前目录的 .env 读取模型地址、密钥等运行配置。
# 每个阶段失败后最多再尝试到第 5 次；成功后立即进入下一阶段。全部终端输出会
# 追加到 podcast_pipeline.log。重复运行时会复用已经下载的文件，以及程序为
# 已完成转写和公司提取保存的检查点结果。

# 后面的命令会通过 tee 同时输出到终端和日志。pipefail 可以避免 tee 成功时
# 掩盖前面实际处理命令的失败状态。
set -o pipefail

# 用户按 Ctrl+C 时使用 130 作为退出码，让调用方能够区分手动中断和普通失败。
trap 'exit 130' INT

MAX_STAGE_ATTEMPTS=5
LOG_FILE="podcast_pipeline.log"

if [[ $# -ne 1 ]]; then
  echo "用法：$0 <播客名或RSS地址>" >&2
  exit 2
fi

if [[ ! -f .env ]]; then
  echo "未找到 .env，请在项目根目录运行该脚本。" >&2
  exit 1
fi

PODCAST_REFERENCE=${1#"${1%%[![:space:]]*}"}
PODCAST_REFERENCE=${PODCAST_REFERENCE%"${PODCAST_REFERENCE##*[![:space:]]}"}
if [[ -z "$PODCAST_REFERENCE" ]]; then
  echo "播客名或 RSS 地址不能为空。" >&2
  exit 2
fi

shopt -s nocasematch
if [[ "$PODCAST_REFERENCE" == http://* || "$PODCAST_REFERENCE" == https://* ]]; then
  FEED_INPUT=(--feed-url "$PODCAST_REFERENCE")
else
  FEED_INPUT=(--podcast "$PODCAST_REFERENCE")
fi
shopt -u nocasematch

# 执行一个完整阶段，并根据命令退出码决定进入下一阶段还是重试。
# 第一个参数是写入日志的阶段名称，其余参数组成实际执行的命令。函数成功时返回
# 0；连续 5 次失败时返回最后一次命令的退出码，脚本因此不会继续执行后续阶段。
run_stage() {
  local stage_name=$1
  shift

  local attempt
  local command_status=1

  for ((attempt = 1; attempt <= MAX_STAGE_ATTEMPTS; attempt++)); do
    echo "===== ${stage_name}：第 ${attempt}/${MAX_STAGE_ATTEMPTS} 次执行 $(date '+%Y-%m-%d %H:%M:%S %z') =====" \
      | tee -a "$LOG_FILE"

    # 在子 shell 中加载 .env，使其中的配置只影响本次阶段命令。set -a 会把
    # .env 中赋值的变量导出给 uv 启动的 Python 程序。
    (
      set -a
      if ! source .env; then
        echo "加载 .env 失败，当前阶段无法继续。" >&2
        exit 1
      fi
      set +a
      "$@"
    ) 2>&1 | tee -a "$LOG_FILE"

    # PIPESTATUS[0] 是 tee 前面子 shell 的退出码，也就是本次阶段的真实结果。
    command_status=${PIPESTATUS[0]}

    if [[ $command_status -eq 0 ]]; then
      echo "===== ${stage_name}执行成功 =====" | tee -a "$LOG_FILE"
      return 0
    fi

    if [[ $command_status -eq 130 ]]; then
      # 手动中断不属于可恢复的普通失败，不继续重复执行当前阶段。
      return 130
    fi

    echo "===== ${stage_name}执行失败，退出码=${command_status} =====" \
      | tee -a "$LOG_FILE"
  done

  # 只有 5 次执行全部失败，程序才会走到这里。command_status 在每次执行后都会
  # 保存新的退出码，因此此时保存的是第 5 次命令的退出码。
  echo "===== ${stage_name}连续 ${MAX_STAGE_ATTEMPTS} 次执行失败，流程停止 =====" \
    | tee -a "$LOG_FILE"
  return "$command_status"
}

# 下载器会跳过已经存在的非空音频。脚本中断后重新运行时，只需继续处理尚未
# 完成的节目，不会重新下载已经完成的节目。
# 每个阶段末尾的“|| exit $?”表示：run_stage 成功返回 0 时继续下一阶段；返回
# 非零退出码时，exit 使用这个退出码结束整个脚本，因此后续阶段不会执行。
run_stage "下载全部音频" \
  uv run podcast-download-rss "$PODCAST_REFERENCE" || exit $?

# --transcribe-only 让命令只生成转写结果；--resume 会复用已经完成的语音切分、
# 片段转写和完整转写结果。只有整个转写阶段成功后，脚本才会进入公司提取阶段。
run_stage "转写全部音频" \
  uv run podcast-find-jobs \
  "${FEED_INPUT[@]}" \
  --source audio \
  --transcribe-only \
  --resume || exit $?

# --extract-only 只读取已有转写，不重新转写音频；--resume 会复用通过校验的公司
# 提取检查点，因此重试时主要继续处理之前失败或尚未完成的内容。
run_stage "从全部转写结果提取公司" \
  uv run podcast-find-jobs \
  "${FEED_INPUT[@]}" \
  --source audio \
  --extract-only \
  --resume || exit $?
