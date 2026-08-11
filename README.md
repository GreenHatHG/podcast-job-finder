# Podcast Job Finder

从小宇宙播客的标题、正文和评论中提取明确出现的招聘主体，并保留对应原文证据。项目也可以通过公开 RSS 获取完整节目清单并批量下载音频。

## 主要功能

- 解析小宇宙单集页面，读取标题、正文、评论和音频地址。
- 调用 `responses` 或 `chat.completions` 接口提取公司名称与原文证据。
- 按公司黑名单过滤结果，并对同一集中的公司名称去重。
- 按公开 RSS 获取全量节目，并批量转写或提取公司。
- 本地音频转写默认从头处理；需要复用检查点时显式添加 `--resume`。
- 对音频通过 TEN VAD 检测、切分和导出语音片段。
- 从公开 RSS 获取完整节目清单，并免登录批量下载原始音频。

## 运行要求

- Python 3.14 或更高版本
- [uv](https://docs.astral.sh/uv/)
- 可用的 OpenAI 或 OpenAI 兼容接口
- `ffmpeg`，仅在使用语音片段切分功能时需要

## 安装

```bash
git clone https://github.com/GreenHatHG/podcast-job-finder.git
cd podcast-job-finder
uv sync
```

## 配置大语言模型

页面公司提取、音频转写、转写文本整理和音频公司提取分别使用独立的大语言模型配置。

例如创建一个不会提交到 Git 的 `.env` 文件：

```dotenv
# 页面公司提取
LLM_PAGE_COMPANY_EXTRACTION_API_KEY=your-page-extraction-api-key
LLM_PAGE_COMPANY_EXTRACTION_MODEL=your-page-extraction-model
LLM_PAGE_COMPANY_EXTRACTION_API_STYLE=responses
# LLM_PAGE_COMPANY_EXTRACTION_BASE_URL=https://provider.example/v1
LLM_PAGE_COMPANY_EXTRACTION_RATE_PER_MINUTE=10

# 小宇宙单集页面请求速率
EPISODE_PAGE_FETCH_RATE_PER_MINUTE=10

# 音频转写，仅支持 chat.completions
LLM_AUDIO_TRANSCRIPTION_API_KEY=your-transcription-api-key
LLM_AUDIO_TRANSCRIPTION_MODEL=your-transcription-model
LLM_AUDIO_TRANSCRIPTION_API_STYLE=chat.completions
# LLM_AUDIO_TRANSCRIPTION_BASE_URL=https://audio-provider.example/v1
LLM_AUDIO_TRANSCRIPTION_RATE_PER_MINUTE=10

# 转写文本整理，支持 responses 或 chat.completions
LLM_TRANSCRIPTION_FORMATTING_API_KEY=your-formatting-api-key
LLM_TRANSCRIPTION_FORMATTING_MODEL=your-formatting-model
LLM_TRANSCRIPTION_FORMATTING_API_STYLE=responses
# LLM_TRANSCRIPTION_FORMATTING_BASE_URL=https://provider.example/v1
LLM_TRANSCRIPTION_FORMATTING_RATE_PER_MINUTE=10

# 从音频转写文本提取公司
LLM_AUDIO_COMPANY_EXTRACTION_API_KEY=your-audio-extraction-api-key
LLM_AUDIO_COMPANY_EXTRACTION_MODEL=your-audio-extraction-model
LLM_AUDIO_COMPANY_EXTRACTION_API_STYLE=responses
# LLM_AUDIO_COMPANY_EXTRACTION_BASE_URL=https://provider.example/v1
LLM_AUDIO_COMPANY_EXTRACTION_RATE_PER_MINUTE=10
```

程序不会自动读取 `.env`。在 `zsh` 或 `bash` 中加载配置：

```bash
set -a
source .env
set +a
```

以下变量用于调整运行行为：

- `COMPANY_BLACKLIST`：需要过滤的公司名称，以英文逗号、中文逗号或换行分隔；匹配时忽略大小写和名称两端空白。
- `LLM_<场景>_MAX_ATTEMPTS`：单次操作最大尝试次数，默认为 `3`。
- `LLM_<场景>_RETRY_BASE_DELAY_SECONDS`：首次重试等待秒数，默认为 `1.0`。
- `LLM_<场景>_RETRY_MAX_DELAY_SECONDS`：重试等待秒数上限，默认为 `8.0`。
- `LLM_<场景>_RATE_PER_MINUTE`：该场景大语言模型请求的每分钟上限；留空时不限速，每次重试也计入限速。
- `EPISODE_PAGE_FETCH_RATE_PER_MINUTE`：批量页面公司提取中，小宇宙单集页面 HTTP 请求的每分钟上限。
- `LOG_LEVEL`：日志级别，默认为 `INFO`。

`<场景>` 支持 `PAGE_COMPANY_EXTRACTION`、`AUDIO_TRANSCRIPTION`、`TRANSCRIPTION_FORMATTING` 和 `AUDIO_COMPANY_EXTRACTION`。`podcast-find-jobs --feed-url ... --source page` 需要页面公司提取配置；`podcast-find-jobs --feed-url ... --source audio` 需要音频转写配置，未使用 `--transcribe-only` 时还需要音频公司提取配置；`podcast-transcribe` 需要音频转写和转写文本整理配置；`podcast-merge-transcriptions` 只需要转写文本整理配置。地址、密钥和模型相同时，可以在 `.env` 中引用前面定义的变量。

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

单集处理无需登录。生成的提示词与大语言模型调用结果会保存到 `output/checkpoints/episodes/<eid>/`，后续使用相同模型配置、黑名单和提示词时会直接复用成功结果。

### 查看单集页面内容

这条命令只解析并输出节目标题、正文、评论和音频地址，不调用大语言模型：

```bash
uv run podcast-inspect-episode \
  "https://www.xiaoyuzhoufm.com/episode/<eid>"
```

### 通过 RSS 批量处理一个播客

批量处理播客不需要登录。RSS 本身始终返回全量节目：

```bash
uv run podcast-find-jobs \
  --feed-url <RSS地址> \
  --source audio \
  --transcribe-only \
  --resume
```

RSS 的节目 ID 和音频地址会直接用于检查 `output/audio/<eid>/`，已有音频和有效转写缓存不会重新下载或转写。批次报告使用由 RSS 地址 SHA-256 哈希前 16 位生成的稳定 `feed_id` 命名，不再需要手工指定播客 ID。

### 从 RSS 下载完整播客

公开 RSS 不需要小宇宙账号。下面的命令会先保存每档播客的完整节目清单，再按清单顺序下载全部音频：

```bash
uv run podcast-download-rss \
  https://feed.xyzfm.space/j8yp8gxkmgqr \
  https://feed.xyzfm.space/jtvfkcxqmnkg \
  https://feed.xyzfm.space/ypn9dydpbxpc \
  https://feed.xyzfm.space/6hpdgggtxpxb \
  https://proxy.wavpub.com/35huan.xml
```

每档播客的清单会保存在 `output/podcasts/<播客名>-<RSS地址摘要>/manifest.json`，其中包含节目标题、发布日期、原始音频 URL、声明的文件大小、本地路径和下载状态。音频默认保存在 `output/audio/<节目ID>/source.<扩展名>`，节目 ID 直接使用 RSS 的 `guid`；小宇宙 RSS 中该值与原有 `eid` 相同，因此会直接复用已有音频和转写目录。

已有的非空音频文件默认跳过，因此任务中断后可以直接重新执行同一命令。只生成完整清单并在下载前检查预计空间时使用：

```bash
uv run podcast-download-rss \
  https://feed.xyzfm.space/j8yp8gxkmgqr \
  --list-only
```

需要重新下载已有文件时添加 `--overwrite`。`--output-dir <目录>` 修改清单目录，`--audio-output-dir <目录>` 修改音频目录。

转写本地音频时，使用 `uv run podcast-transcribe <音频路径> --resume` 复用有效的片段检查点。

### 检测并导出语音片段

语音切分目前提供 Python 接口。它会调用 `ffmpeg` 把输入音频规范化为 16 kHz 单声道 WAV，使用 TEN VAD 检测人声，并把片段导出为 WAV：

```python
from pathlib import Path

from podcast_job_finder.audio import detect_and_export_speech_segments

segments = detect_and_export_speech_segments(
    Path("output/audio/<eid>/source.m4a"),
    output_dir=Path("output/audio/<eid>/segments"),
)

for segment in segments:
    print(segment.to_dict())
```

## 输出与缓存

运行产生的音频、报告和检查点均已加入 `.gitignore`。主要目录结构如下：

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
├── result_<feed_id>_<UTC时间>.json
├── summary_<feed_id>_<UTC时间>.json
└── podcasts/
    └── <播客名>-<RSS地址摘要>/
        └── manifest.json
```

检查点签名包含模型、接口地址、接口类型、公司黑名单和提示词模板。任一内容变化后，对应节目会重新抓取并调用大语言模型。

## 项目结构

```text
podcast_job_finder/
├── audio/                  # 节目音频下载、规范化、VAD 检测与片段导出
│   ├── episode_audio/      # 节目音频下载和本地文件保存
│   └── segmentation/       # 音频规范化、VAD、切分和片段导出
├── cli/                    # 命令行入口
├── companies/              # 公司提取和处理流程
├── episode/                # 页面解析、节目模型与单集 URL 工具
├── http/                   # 共享 HTTP 配置
├── llm/                    # OpenAI 兼容客户端、配置与重试
├── rss/                    # RSS 清单解析与批量音频下载
└── transcription/          # 转写流程、后端、检查点和文本整理
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
