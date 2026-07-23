# Podcast Job Finder

从小宇宙播客的标题、正文和评论中提取明确出现的招聘主体，并保留对应原文证据。项目支持处理单个节目，也支持按播客 `pid` 批量抓取节目、调用兼容 OpenAI 接口的大语言模型，并生成逐集结果与公司汇总。

## 主要功能

- 解析小宇宙单集页面，读取标题、正文、评论和音频地址。
- 调用 `responses` 或 `chat.completions` 接口提取公司名称与原文证据。
- 按公司黑名单过滤结果，并对同一集中的公司名称去重。
- 按播客 `pid` 批量处理节目，支持分页抓取、请求限速和失败重试。
- 断点续跑：配置和提示词未变化时可复用成功结果，失败任务或者中断任务重新执行时可接着之前的进度跑。
- 下载单集音频，并通过 TEN VAD 检测、切分和导出语音片段。

## 运行要求

- Python 3.14 或更高版本
- [uv](https://docs.astral.sh/uv/)
- 可用的 OpenAI 或 OpenAI 兼容接口
- `ffmpeg`，仅在使用语音片段切分功能时需要
- 可选：本地 [`xyz`](https://github.com/ultrazg/xyz/releases) 服务，批量处理播客和小宇宙登录时需要；程序固定访问 `http://localhost:23020`

## 安装

```bash
git clone https://github.com/GreenHatHG/podcast-job-finder.git
cd podcast-job-finder
uv sync
```

需要批量处理播客时，从 [`xyz` Releases](https://github.com/ultrazg/xyz/releases) 下载适合当前操作系统的版本，并按照对应版本的发布说明完成安装。

### 启动 `xyz` 的时机

以下命令会调用 `xyz`，执行前需要启动服务，并确保它正在监听 `http://localhost:23020`：

- `send-code`：发送小宇宙登录验证码。
- `login`：使用验证码登录并保存凭据。
- `pid`：获取播客节目列表、翻页和刷新登录令牌。

处理单集 URL、查看单集页面内容、下载音频以及切分本地语音片段时无需启动 `xyz`。`send-code` 和 `login` 完成后可以关闭服务；运行 `pid` 批量任务前需要重新启动，并保持服务运行到命令结束。

## 配置大语言模型

处理公司信息前，需要在当前终端设置以下环境变量：

- `OPENAI_API_KEY`：接口密钥，必填。
- `OPENAI_MODEL`：模型名称，必填。
- `OPENAI_API_STYLE`：接口类型，必填，可选值为 `responses` 或 `chat.completions`。
- `OPENAI_BASE_URL`：兼容接口地址，可选；使用 OpenAI 官方接口时可省略。

例如创建一个不会提交到 Git 的 `.env` 文件：

```dotenv
OPENAI_API_KEY=your-api-key
OPENAI_MODEL=your-model
OPENAI_API_STYLE=responses
# OPENAI_BASE_URL=https://your-provider.example/v1
```

程序不会自动读取 `.env`。在 `zsh` 或 `bash` 中加载配置：

```bash
set -a
source .env
set +a
```

以下变量用于调整运行行为：

- `COMPANY_BLACKLIST`：需要过滤的公司名称，以英文逗号、中文逗号或换行分隔；匹配时忽略大小写和名称两端空白。
- `OPENAI_MAX_ATTEMPTS`：单集最大尝试次数，默认为 `3`。
- `OPENAI_RETRY_BASE_DELAY_SECONDS`：首次重试等待秒数，默认为 `1.0`。
- `OPENAI_RETRY_MAX_DELAY_SECONDS`：重试等待秒数上限，默认为 `8.0`。
- `LLM_PIPELINE_PRODUCER_RATE_PER_MINUTE`：批量模式中提示词任务进入队列的每分钟上限。
- `LLM_PIPELINE_CONSUMER_RATE_PER_MINUTE`：批量模式中大语言模型请求的每分钟上限。
- `LOG_LEVEL`：日志级别，默认为 `INFO`。

## 使用方法

### 提取单集中的公司

```bash
uv run podcast-find-jobs \
  "https://www.xiaoyuzhoufm.com/episode/<eid>"
```

命令会把 JSON 结果写到标准输出：

```json
{
  "companies": [
    {
      "name": "示例公司",
      "evidence": "示例公司正在招聘产品经理"
    }
  ],
  "filtered_count": 0
}
```

单集处理无需登录，也无需启动本地 `xyz` 服务。生成的提示词与大语言模型调用结果会保存到 `output/checkpoints/episodes/<eid>/`，后续使用相同模型配置、黑名单和提示词时会直接复用成功结果。

### 查看单集页面内容

这条命令只解析并输出节目标题、正文、评论和音频地址，不调用大语言模型：

```bash
uv run podcast-inspect-episode \
  "https://www.xiaoyuzhoufm.com/episode/<eid>"
```

### 批量处理一个播客

批量模式需要本地 `xyz` 服务运行在 `http://localhost:23020`。先获取验证码并登录：

```bash
uv run podcast-find-jobs send-code \
  --mobile <手机号> \
  --area-code +86

uv run podcast-find-jobs login \
  --mobile <手机号> \
  --code <验证码> \
  --area-code +86
```

登录成功后，凭据保存在项目根目录的 `.xiaoyuzhou_auth.json`，文件权限为仅当前用户可读写。处理播客最新一页的节目：

```bash
uv run podcast-find-jobs pid --pid <pid>
```

加入 `--all` 可抓取全部分页：

```bash
uv run podcast-find-jobs pid --pid <pid> --all
```

批量处理完成后会在 `output/` 生成两个文件：

- `result_<pid>_<UTC时间>.json`：逐集状态、公司、证据和错误信息。
- `summary_<pid>_<UTC时间>.json`：去重后的公司汇总、出现次数及关联节目。

只要有一集处理失败，批量命令就会返回非零退出码；成功节目及已经写入的检查点仍会保留，重新执行命令可继续处理。

### 下载单集音频

```bash
uv run podcast-download-audio \
  "https://www.xiaoyuzhoufm.com/episode/<eid>"
```

音频默认保存到 `output/audio/<eid>/source.<扩展名>`。已有文件默认跳过，使用 `--overwrite` 覆盖，使用 `--output-dir <目录>` 修改输出目录。

### 检测并导出语音片段

语音切分目前提供 Python 接口。它会调用 `ffmpeg` 把输入音频规范化为 16 kHz 单声道 WAV，使用 TEN VAD 检测人声，并把片段导出为 WAV：

```python
from pathlib import Path

from podcast_job_finder.audio.speech_pipeline import (
    detect_and_export_speech_segments,
)

segments = detect_and_export_speech_segments(
    Path("output/audio/<eid>/source.m4a"),
    output_dir=Path("output/audio/<eid>/segments"),
)

for segment in segments:
    print(segment.to_dict())
```

## 输出与缓存

运行产生的登录凭据、音频、报告和检查点均已加入 `.gitignore`。主要目录结构如下：

```text
output/
├── audio/
│   └── <eid>/
│       ├── source.<扩展名>
│       └── segments/
├── checkpoints/
│   └── episodes/
│       └── <eid>/
│           ├── llm_prompt.txt
│           ├── llm_response.txt
│           └── llm_state.json
├── result_<pid>_<UTC时间>.json
└── summary_<pid>_<UTC时间>.json
```

检查点签名包含模型、接口地址、接口类型、公司黑名单和提示词模板。任一内容变化后，对应节目会重新抓取并调用大语言模型。

## 项目结构

```text
podcast_job_finder/
├── audio/                  # 音频规范化、VAD 检测、片段导出与转录
├── cli/                    # 命令行入口
├── companies/              # 公司提取、单集任务、批量处理与报告
├── http/                   # 共享 HTTP 配置
├── llm/                    # OpenAI 兼容客户端、配置与重试
└── xiaoyuzhou/             # 页面解析、音频下载与 xyz 服务集成
```

## 开发检查

安装开发依赖并运行项目配置的检查：

```bash
uv sync --group dev
uv run pre-commit run --all-files
```

单独运行静态检查：

```bash
uv run mypy .
uv run pyright
```
