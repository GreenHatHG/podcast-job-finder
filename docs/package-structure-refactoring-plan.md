# Python 包结构重构计划

## 实施进度

- 阶段 1（节目模型与 URL 处理）：已完成
- 阶段 2（转写模型、片段执行、诊断和质量报告）：已完成
- 阶段 3（OpenAI 兼容后端）：已完成
- 阶段 4（FireRed 后端和 worker）：已完成
- 阶段 5（豆包后端）：已完成
- 阶段 6（运行配置、检查点和清单）：已完成
- 阶段 7（转写文本整理）：已完成
- 阶段 8（批量流程和调度）：已完成
- 阶段 9（音频规范化、VAD、切分和片段导出）：已完成
- 阶段 10（README、旧路径清理和完整验证）：已完成

最后更新：全部阶段已完成；README 和 Python 接口示例已更新，旧模块路径已清理，格式化、静态检查、类型检查、构建、模块与命令入口导入、FireRed worker、wheel 安装包、依赖方向和历史持久化文件兼容验证均已通过。

## 总体方案

项目采用按业务能力分包的方式组织代码：

- 同一工作流的代码放在相邻目录，减少跨目录搜索。
- `audio` 只负责音频获取、规范化和切分。
- 新建顶层 `transcription`，集中管理转写流程、后端、检查点和文本整理。
- 保留职责清楚的 `episode`、`rss`、`companies` 和 `llm`。
- 不引入容易积累无关代码的 `utils`、`common` 等通用目录。
- 调整依赖方向，解除当前 `audio` 与 `companies` 的双向依赖。

目标结构如下：

```text
podcast_job_finder/
├── audio/
│   ├── episode_audio/          # 节目音频下载和本地文件保存
│   └── segmentation/           # 音频规范化、VAD、切分和片段导出
├── transcription/
│   ├── backends/
│   │   ├── openai_compatible.py
│   │   ├── doubao/
│   │   └── firered/
│   │       └── worker/
│   ├── formatting/             # 转写整理、审计、报告和文章输出
│   ├── models.py
│   ├── segments.py
│   ├── runtime.py
│   ├── batch.py
│   ├── schedule.py
│   ├── checkpoint.py
│   ├── manifest.py
│   ├── diagnostics.py
│   ├── quality.py
│   └── quality_report.py
├── companies/                  # 公司提取领域和处理流程
├── episode/                    # 节目数据、URL 处理、页面请求和页面解析
│   └── urls.py                 # 节目 URL 构建和节目 ID 解析
├── rss/                        # RSS 获取、解析和批量下载流程
├── llm/                        # 通用 OpenAI 兼容客户端
└── cli/                        # 参数解析和执行入口
```

包初始化文件按以下约定创建或迁移：

- 新建 `audio/segmentation/__init__.py`，只供 `audio.__init__` 重新导出音频处理接口，不把内部切分模块路径作为稳定接口。
- 新建 `transcription/__init__.py`，只导出“公开行为”一节列出的稳定转写接口。
- 新建 `transcription/backends/__init__.py` 和 `transcription/formatting/__init__.py`，不从包级入口导出具体后端或格式化流程。
- 新建 `transcription/backends/firered/__init__.py`；`audio/doubao/__init__.py` 随豆包目录迁移为 `transcription/backends/doubao/__init__.py`。
- `transcription/backends/firered/worker/__init__.py` 保留当前空初始化文件，便于构建产物包含同目录脚本；worker 目录只按脚本路径由独立 Python 解释器启动，不作为主项目环境中的普通模块导入，也不从任何包级 `__init__.py` 导出。

## 包边界和依赖方向

将 `EpisodeWorkItem` 从 `companies.episode_runner` 移到 `episode.models`。这个类型描述节目任务，不属于公司提取功能；移动后，`companies` 和 `transcription` 都可以依赖 `episode`，不再互相反向引用。

当前 `episode.client` 依赖 `episode.models`，而 `EpisodeWorkItem.resolve_episode_id()` 需要节目 ID 解析函数。为避免形成 `episode.models -> episode.client -> episode.models` 循环导入，将纯 URL 处理移到 `episode.urls`：

- `episode.models` 和 `episode.client` 都可以依赖 `episode.urls`。
- `episode.urls` 不导入页面请求、页面解析或其他业务包。
- `EpisodeWorkItem.to_result_metadata()` 的元数据构建逻辑一起移入 `episode.models`，不反向调用 `companies` 中的私有函数。

依赖方向固定为：

```text
cli -> companies / transcription / audio / rss / episode / llm
companies -> transcription / episode / llm
transcription -> audio / episode / llm
rss -> episode / audio.episode_audio
audio -> http / progress
episode -> http
llm -> http
```

`filesystem`、`timestamps`、`logging`、`tracing`、`progress` 和 `runtime_signature` 是职责明确的基础模块，业务包可以按需使用，但基础模块不反向依赖业务包。`audio.segmentation.vad` 使用 `progress` 输出长音频的检测进度，因此该依赖属于允许的基础模块依赖。

具体约束如下：

- `audio` 不导入 `transcription`、`companies` 或 `cli`。
- `transcription` 可以使用 `audio`、`episode` 和 `llm`，但不导入 `companies`。
- `companies` 可以使用 `transcription`、`episode` 和 `llm`。
- `rss` 可以使用 `episode` 和节目音频下载能力，不依赖公司提取或转写流程。
- `cli` 可以直接组织它对应的业务包，但只负责解析参数、加载运行配置和组织调用，不保存核心业务规则。

不新增依赖边界检查工具。先通过明确的目录结构、导入方向和现有静态检查维持边界。

## 主要调整

### 1. 共享节目模型

- 将 `EpisodeWorkItem` 移到 `episode.models`。
- 将 `extract_episode_id_from_url`、`build_episode_url` 及相关 URL 常量移到 `episode.urls`，`episode.client` 继续专注页面请求。
- 将结果元数据构建逻辑移入 `EpisodeWorkItem`，不依赖 `companies.episode_runner`。
- 修改批量页面提取、批量音频转写和 CLI 的导入路径。
- 保留 `EpisodeWorkItem` 的字段、节目 ID 解析和结果元数据行为。

### 2. 音频处理

将以下职责归入 `audio.segmentation`：

- PCM 常量和音频规范化。
- VAD 人声检测。
- 切分候选搜索和切分点选择。
- 语音片段导出。
- 语音片段检查点和切分流程。

为避免实施时临时决定名称，按以下路径迁移，并尽量保留原文件名：

- `audio/_pcm.py` -> `audio/segmentation/_pcm.py`。
- `audio/normalized_audio.py` -> `audio/segmentation/normalized_audio.py`。
- `audio/_segment_candidate_search.py` -> `audio/segmentation/_segment_candidate_search.py`。
- `audio/_segment_candidates.py` -> `audio/segmentation/_segment_candidates.py`。
- `audio/_segment_split.py` -> `audio/segmentation/_segment_split.py`。
- `audio/vad.py` -> `audio/segmentation/vad.py`。
- `audio/segment_export.py` -> `audio/segmentation/segment_export.py`。
- `audio/speech_pipeline.py` -> `audio/segmentation/speech_pipeline.py`。
- `audio/speech_segment_checkpoint.py` -> `audio/segmentation/speech_segment_checkpoint.py`。

`audio.__init__` 继续暴露现有的包级接口：`AudioFileDecodeError`、`AudioSegmentExportError`、`ExportedSpeechSegment`、`SpeechSegment`、`VadConfig` 和 `detect_and_export_speech_segments`。这些名称的内部来源改为 `audio.segmentation`，调用方无需修改 `from podcast_job_finder.audio import ...`。README 中的语音切分示例改用这个稳定的包级接口，不直接依赖 `audio.segmentation` 内部模块路径。

`audio.episode_audio` 继续负责节目音频下载、本地文件发布、重复下载跳过和下载错误，不与音频信号处理混合。

### 3. 转写核心

将现有转写核心模块迁入顶层 `transcription`：

- 将 `audio/transcription.py` 中的转写结果、语音片段、异常、转写器协议和时间字段解析移到 `transcription/models.py`。
- 将 `audio/firered_alignment.py` 中的 `CharacterAlignment` 移到 `transcription/models.py`。
- 将 `audio/transcription.py` 中的 `transcribe_speech_segment`、`validate_previous_segment_order` 和 `build_transcribed_speech_segment` 移到 `transcription/segments.py`，避免 `models.py` 同时承担数据定义和转写执行职责。
- `audio/transcription_diagnostics.py` -> `transcription/diagnostics.py`。
- `audio/transcription_quality.py` -> `transcription/quality.py`。
- `audio/transcription_confidence_report.py` -> `transcription/quality_report.py`。
- `audio/transcription_checkpoint.py` -> `transcription/checkpoint.py`。
- `audio/transcription_manifest.py` -> `transcription/manifest.py`。
- `audio/batch_transcription.py` -> `transcription/batch.py`。
- `audio/batch_transcription_schedule.py` -> `transcription/schedule.py`。
- `audio/transcription_runtime.py` -> `transcription/runtime.py`，根据环境变量选择转写后端并管理资源释放。

将 `CharacterAlignment` 等通用转写数据结构放在 `transcription.models` 中，避免通用质量检查反向依赖 FireRed 后端。`transcription.segments` 可以依赖 `transcription.models` 和 `audio.segmentation`，`transcription.models` 不导入具体后端、检查点、批量流程或 CLI。

### 4. 转写后端

- `audio/llm_transcriber.py` -> `transcription/backends/openai_compatible.py`。
- 将 `audio/doubao/` 整体迁入 `transcription/backends/doubao/`，保留其中 `client.py`、`config.py`、`output.py`、`request_client.py`、`request_scheduler.py`、`response.py`、`transcriber.py` 和 `truncation_probe.py` 的文件名。
- `audio/firered_config.py` -> `transcription/backends/firered/config.py`。
- `audio/firered_transcriber.py` -> `transcription/backends/firered/transcriber.py`。
- `audio/firered_alignment.py` -> `transcription/backends/firered/alignment.py`；`CharacterAlignment` 已按“转写核心”一节移入 `transcription.models`，本模块只保留 FireRed 文字对齐配置、客户端和响应解析。
- 将 `audio/_firered_worker/` 下的全部脚本迁入 `transcription/backends/firered/worker/`，保留现有文件名和同目录导入方式。该目录是由独立 Python 解释器按脚本路径启动的工作进程文件集合，不作为主项目环境中的普通可导入包使用。
- 保留 FireRed 独立 Python 环境及其 Python 版本和依赖约束。
- 同步修改 FireRed 转写和文字对齐客户端中指向工作进程脚本的文件路径。
- 豆包仍可组合使用 FireRed 文字对齐，但通用转写核心不依赖具体后端。

### 5. 转写文本整理

将转写文本整理模块迁入 `transcription.formatting`：

- `audio/transcription_input.py` -> `transcription/formatting/input.py`。
- `audio/transcription_formatter.py` -> `transcription/formatting/formatter.py`。
- `audio/transcription_format_audit.py` -> `transcription/formatting/audit.py`。
- `audio/transcription_format_report.py` -> `transcription/formatting/report.py`。
- `audio/transcription_article.py` -> `transcription/formatting/article.py`。

整理功能继续使用独立的 `LLM_TRANSCRIPTION_FORMATTING_*` 配置，不与音频识别后端配置合并。

## 公开行为

这次允许调整 Python 模块导入路径，不保留旧路径的兼容转发模块。同步修改所有内部导入和 README 中的 Python 示例。

以下用户可观察行为保持不变：

- `pyproject.toml` 中已有的命令名称和入口。
- CLI 参数、默认值、退出状态和错误信息。
- 环境变量名称和含义。
- 输出 JSON 字段、检查点内容和持久化文件格式。
- `output/` 下现有目录和文件命名方式。
- 已有检查点和转写清单的读取行为。

`audio.__init__` 只暴露音频处理接口，并保留本计划“音频处理”一节列出的现有包级名称。`transcription.__init__` 只暴露以下稳定接口：

```text
AudioTranscriptionError
AudioTranscriberProtocol
BatchAudioTranscriberProtocol
AudioTranscriptionResult
TranscriptionOutput
TranscribedSpeechSegment
TranscriptionSegmentResult
TimedTranscriptionText
CharacterAlignment
```

这些名称都从 `transcription.models` 导出并列入 `__all__`。具体后端、片段执行函数、检查点、清单、批量流程和格式化流程使用明确的模块路径导入，不通过 `transcription.__init__` 暴露。

## 分阶段实施

每个阶段完成后保持项目可以导入，并单独运行相关检查。没有提交授权时只规划审核边界，不自动创建提交。

前述依赖方向是全部迁移完成后的最终约束。第 2～8 阶段中，尚未迁出的旧 `audio` 转写模块可以临时导入已经迁入的 `transcription` 模块，因此包级依赖图可能暂时出现 `audio -> transcription`。临时依赖只能用于旧模块继续调用已迁移实现；新的 `transcription` 模块不得反向导入 `audio` 下的旧转写、后端或格式化模块，只能使用仍属于音频处理的模块和 `audio.episode_audio`，避免形成实际循环导入。第 8 阶段结束后必须删除全部 `audio -> transcription` 临时依赖，第 9 阶段再更新最终的音频处理模块路径。

1. 将节目 URL 处理迁入 `episode.urls`，将 `EpisodeWorkItem` 及其元数据逻辑迁入 `episode.models`，同阶段修改全部调用方并确认没有循环导入。
2. 迁移转写模型、片段执行函数、诊断、质量判断和质量报告到 `transcription`，同阶段修改仍在旧位置的后端和流程模块导入。
3. 迁移 OpenAI 兼容后端，同阶段修改运行配置的导入。
4. 迁移 FireRed 后端和工作进程，同阶段修改工作进程文件路径、运行配置和豆包文字对齐导入。
5. 迁移豆包后端，同阶段修改运行配置导入。
6. 迁移运行配置、转写检查点和转写清单到 `transcription`，同阶段更新仍在旧位置的批量流程、`companies` 和 CLI 的直接导入。
7. 迁移转写文本整理到 `transcription.formatting`，同阶段更新仍在旧位置的批量流程和 CLI 的导入。
8. 迁移批量流程和执行顺序到 `transcription.batch` 与 `transcription.schedule`，同阶段更新 `companies` 和 CLI 的全部调用方；阶段结束后确认 `audio` 不再导入 `transcription`。
9. 按“音频处理”一节的文件映射将音频规范化、VAD 和片段处理归入 `audio.segmentation`，同阶段更新所有调用方，并保持 `audio.__init__` 的现有包级名称可用。
10. 更新 README 项目结构和 Python 接口示例，扫描并清理全仓库的旧模块路径。

每次移动模块时，必须在同一阶段修改该模块的全部内部调用方、相关 `__init__.py` 导出和命令入口。不依赖旧路径转发模块维持中间状态。

文件移动存在机械耦合时，可以在同一阶段修改超过五个文件，但不得混入业务行为修改、无关格式化或顺手清理。

## 验证方案

每个阶段执行与当前改动直接相关的语法编译、实际模块导入和类型检查。`compileall` 只用于检查语法，不代替实际导入检查。全部迁移完成后执行：

```bash
git diff --check
uv run pre-commit run --all-files
uv run mypy .
uv run pyright
uv build
```

另外执行以下检查：

- 使用 `compileall` 检查语法，再逐模块导入主项目包，确认没有错误路径、缺失名称和循环导入。FireRed 工作进程模块不在主项目环境中导入，由独立环境检查。
- 对 `pyproject.toml` 中的全部命令运行 `--help`，确认入口模块可以加载。
- 使用 FireRed 独立 Python 环境分别运行新路径下的 `worker.py --help` 和 `alignment_worker.py --help`，确认工作进程的同目录导入和依赖可用，不加载模型。
- 检查 FireRed 转写和文字对齐客户端计算出的工作进程脚本路径存在，并使用 `unzip -l` 或等价方式确认 wheel 中包含全部工作进程脚本。
- 将构建出的 wheel 以 `--no-deps` 安装到临时目录，从仓库目录之外使用项目现有 Python 环境和该临时安装目录导入主项目包。通过已安装模块的 `__file__` 重新计算 FireRed 工作进程路径，确认 `worker.py`、`alignment_worker.py` 及其同目录依赖文件都存在，避免源码目录掩盖安装包缺失文件或相对路径错误。
- 从临时安装目录读取 wheel 的入口点元数据，在独立子进程中把参数设置为 `--help` 并逐个调用 `pyproject.toml` 声明的命令入口。源码目录和临时安装目录中的 `--help` 检查都必须通过，且临时安装检查不得把仓库根目录加入模块搜索路径。
- 使用一次性的 Python AST 导入扫描输出顶层包依赖图，逐条确认 `audio`、`transcription`、`companies`、`rss` 和基础模块符合本计划声明的依赖方向。该扫描只作为验证命令运行，不新增项目依赖、配置或长期维护的检查工具。
- 仅在 `podcast_job_finder/`、`README.md`、`scripts/` 和 `run.sh` 中搜索原有 `podcast_job_finder.audio.transcription*`、`podcast_job_finder.audio.batch_transcription*`、`podcast_job_finder.audio.doubao`、FireRed 旧路径，以及 `podcast_job_finder.audio.normalized_audio`、`segment_export`、`speech_pipeline`、`speech_segment_checkpoint`、`vad` 和 `_segment_*` 等音频切分旧路径，确认没有遗留的内部导入、脚本引用或 README 示例。排除 `docs/package-structure-refactoring-plan.md`，因为该计划本身必须保留旧路径到新路径的迁移映射。
- 优先从现有 `output/` 中选择真实文件并复制到临时目录；如果当前环境没有可用输出，则按现有读取器要求构造最小临时样本。分别验证 `speech_segments.json`、片段转写检查点 JSON、`transcription.json`、`transcription_quality_report.json` 和批量转写报告。执行读取、解析和重新序列化后，对比关键字段、文件名和目录位置，确认持久化格式没有变化；不直接修改原有输出文件。
- 持久化兼容样本必须包含 `start_ms == end_ms` 的旧逐字时间记录，确认迁移后仍可读取和重新保存；负数时间和倒序时间仍保持原有的拒绝行为。
- 检查最终差异，确认没有修改业务逻辑、输出格式或无关文件。

功能验证不请求 RSS、节目页面、LLM、豆包或其他业务外部服务，也不加载 FireRed 模型。`uv`、`pre-commit` 和构建工具在本地缺少依赖、hook 环境或缓存时可能需要下载工具依赖；执行重构前应提前准备这些环境。必须离线执行但本地依赖不完整时，不绕过检查，应记录无法运行的具体命令和未验证范围。默认不新增单元测试；本次是行为保持不变的结构重构，优先通过现有静态检查、类型检查和命令入口检查验证。

## 实施约定

- 按职责移动和命名模块，不顺便重写大文件或调整无关代码风格。
- 不为了减少文件数量合并职责不同的代码。
- 不为了缩短文件行数创建大量只有少量内容的小模块。
- 不新增未来可能使用的接口、配置、兼容层或扩展点。
- 如果实施过程中发现必须改变 CLI、数据格式、检查点或输出目录，应暂停并单独确认，不把行为变化混入目录重构。
