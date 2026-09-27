# AI2-THOR 自动 Rollout → 规则 QA → VLM 润色

用于生成 AI2-THOR 视频和 QA 的 STORM 代码包。安装依赖后运行 `scripts/rollout.py`，再用 `scripts/polish_qa.py` 接入 VLM 润色。

默认流程生成四类房间各 7 段视频，共 28 段、300 条四选一 QA。视频为 640×480、20 FPS、约 60 秒，每段规划 10 个物体消失／重现事件。VLM 只负责文字润色，规则生成器负责答案与证据标签。

原 v2.5 数据使用规则模板改写文字。本包新增了独立的 VLM API 脚本。

- [代码链路与数据集说明](PIPELINE.md)
- [验证记录](VALIDATION.md)
- [原始源码快照清单](source_manifest.json)

## 1. 安装

完整仿真建议使用 **Linux x86_64 + NVIDIA GPU/Vulkan 驱动 + Python 3.11**。仅 QA 导出、VLM 润色和测试不需要 GPU；本包已在 macOS 上测试这些离线功能。未验证 Apple Silicon 上运行 Unity 仿真。

```bash
cd /your/path/ai2thor
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements-rollout.txt
python scripts/check_environment.py
```

`vendor/ai2thor` 保存生成原数据时使用的 Python SDK，保留上游 Apache-2.0 LICENSE。该工作树的包版本显示为 `0.0.1`，不是 PyPI 上的发行版本号；不要执行 `pip install ai2thor==0.0.1`。安装时请在本包根目录执行上述命令。

Unity 构建固定为 `4d2e1f1d04051fafcd9794b810f227551121253a`，首次运行由 SDK 下载，需要网络及磁盘空间。Linux 默认使用 `CloudRendering`。已有 Unity 可执行文件时可跳过下载：

```bash
export STORM_THOR_EXECUTABLE=/absolute/path/to/thor-CloudRendering-4d2e1f1d04051fafcd9794b810f227551121253a
python scripts/check_environment.py --render
```

`--render` 启动 FloorPlan1，并保存一张 RGB 图片，用于检查渲染环境。

多 GPU 机器请按当前资源选择，例如 `CUDA_VISIBLE_DEVICES=0 python scripts/check_environment.py --render`；GPU 编号由使用者决定。服务器需有 Vulkan loader（Ubuntu 通常为 `libvulkan1`）及可用的 NVIDIA Vulkan 驱动。

可选环境变量：`STORM_THOR_PLATFORM`、`STORM_THOR_COMMIT`；设置 `STORM_THOR_EXECUTABLE` 时以本地可执行文件为准。代理、DISPLAY 和 GPU 选择沿用当前环境。

只做后处理：

```bash
python -m pip install -r requirements.txt
```

视频写入和 VLM 抽帧使用 `imageio-ffmpeg` 的 FFmpeg。历史 `keyclip_exporter` 如需单独使用，还需在 PATH 中提供 `ffmpeg` 和 `ffprobe`，默认主链路不使用它。

## 2. 一键生成视频和 QA

```bash
python scripts/rollout.py --output outputs/run01
```

输出目录须为新目录。运行分为三步：

1. 四类房间分批生成并验证 episode。
2. 从 accepted rollout 导出 `qa_reference`。
3. 从相同原始证据重新构建 v2.5 规则 QA，写入 `qa_rule`。

```text
outputs/run01/
├── raw/                         # Room configs, plans, traces, evidence and videos
├── qa_reference/                # Reference QA with source episode paths
└── qa_rule/                     # v2.5 rule QA; input to the VLM script
    ├── meta_data/qa_results/<scene>/<episode>.json
    ├── meta_data/gene_videos/<scene>/<episode>.mp4
    ├── questions.jsonl
    ├── model_inputs.jsonl       # Evaluation inputs without labels
    ├── evaluation_labels.json
    ├── evaluation_splits.json
    ├── evaluation_protocol.json
    └── summary.json
```

`--seed` 控制 rollout，`--qa-seed` 控制 v2.5 QA。使用不同模拟器构建或依赖版本不保证同 seed 字节一致。

先做小规模试跑可以使用 reference 配方：

```bash
python scripts/rollout.py --profile reference \
  --episodes-per-room 1 --question-count 44 --output outputs/small
```

v2.5 配方固定为 **28 视频 / 300 QA**。reference 支持每视频 10–12 题，四房间题数相等，并满足六类问题配额；脚本会在启动仿真前检查。原验证配置为 640×480、约一分钟的视频。

## 3. 复用已有 rollout

房间已经生成成功时，可复用 accepted run，避免重新渲染。其 `manifest.json` 与 `batch_validation.json` 均需通过，episode 数须与参数匹配：

```bash
python scripts/rollout.py --output outputs/run02 \
  --reuse-run kitchen=/path/to/accepted/kitchen/rollouts/run_XXX \
  --reuse-run living_room=/path/to/accepted/living_room/rollouts/run_XXX \
  --reuse-run bedroom=/path/to/accepted/bedroom/rollouts/run_XXX \
  --reuse-run bathroom=/path/to/accepted/bathroom/rollouts/run_XXX
```

已经有 `qa_reference` 或原 v2.x 数据、只想重建规则 QA：

```bash
python scripts/export_qa.py --source-dataset outputs/run01/qa_reference \
  --output outputs/qa_rebuilt --link-mode copy
python scripts/validate_dataset.py outputs/qa_rebuilt
```

重建 QA 需要原始 `episode_plan.json`、`execution_trace.json`、`scene_profile.json` 等文件，路径记录在 `qa_reference/summary.json` 的 `raw_episode_dirs` 中。迁移数据后需更新这些路径。视频默认使用硬链接，跨文件系统时复制；JSON 独立写入。

## 4. 接自己的 VLM API

接口采用常见 Chat Completions JSON 格式：`POST <base-url>/chat/completions`，Bearer key，支持 `image_url` 的 JPEG data URL。任何兼容协议的视觉模型服务均可接入；其他协议只需替换 `polish_qa.py` 的 `call_vlm()`。

```bash
export VLM_BASE_URL=https://your-provider.example/v1
export VLM_API_KEY=your-key
export VLM_MODEL=your-vision-model

python scripts/polish_qa.py --input outputs/run01/qa_rule \
  --output outputs/qa_polished --max-frames 12
python scripts/validate_dataset.py outputs/qa_polished
```

`.env.example` 仅作示例，脚本不会自动加载 `.env`。API key 只从环境读取，不写入文件或命令行参数。base URL 应指向 API 根路径（通常以 `/v1` 结尾），不要包含 `/chat/completions`。

先检查两道题的请求与图片，不调用 API：

```bash
python scripts/polish_qa.py --input outputs/run01/qa_rule \
  --output outputs/vlm_preview --dry-run --limit 2
```

`requests/*.json` 保存抽帧图片和编辑请求，含答案等私有标签，仅用于检查请求。正式润色时使用另一个输出目录。

继续中断任务 / 重试失败题：

```bash
python scripts/polish_qa.py --input outputs/run01/qa_rule \
  --output outputs/qa_polished --max-frames 12 --resume --retry-failed
```

恢复时模型、端点、图片设置、原始 QA、视频内容和脚本版本必须一致。已完成题直接使用 checkpoint；`--retry-failed` 重试 error / flag。模型配置变化请使用新目录。不要同时启动两个进程写同一输出目录。

润色规则：

- 按每题 `query_time` 严格截断可见前缀，优先抽取证据区间边界，再补充前缀覆盖帧。视频必须为该生成器输出的恒定帧率 MP4。
- 每题一次请求。只允许改 `question` 和私有 `video_evidence`。选项本身及顺序、`answer_index`、证据时间、题型、诊断标签都不允许修改。
- VLM 返回 `rewrite`、`keep` 或 `flag`。flag、解析失败、越权字段或网络错误都保留原题；有 error 时退出码为 2，并写明逐题状态。
- 对 429/5xx/网络错误按上限退避重试。`--limit N` 可控制试用量，其余题保留原文；报告将它们标为 `not_selected`。
- `polish_audit.jsonl` 与 `.polish_cache/` 保存编辑结果、理由和使用的帧时间；`polish_run.json` 保存无密钥的配置指纹。
- 重建逐 episode QA、汇总 JSONL、模型输入、评测标签，避免只改其中一个副本。润色结果的视频路径相对于数据集根目录，整个目录可搬迁。

润色会将图片、QA 和编辑用标签发送到配置的服务。结果需检查语义并重新评测，历史 `release_status.json` 不随新数据导出。评测请使用不含标签的 `model_inputs.jsonl`。

## 5. 测试

```bash
python -m pip install pytest
python -m pytest -q
```

测试覆盖规划、QA 导出、FFmpeg 抽帧、HTTP mock、失败处理、恢复和标签一致性。测试无需外部 API 或 Unity。详见 [验证记录](VALIDATION.md)。

## 6. 主要文件

| 路径 | 职责 |
|---|---|
| `scripts/rollout.py` | 对外统一入口，仿真及规则 QA 两阶段 |
| `scripts/export_qa.py` | 复用既有证据，重建 v2.5 QA |
| `scripts/polish_qa.py` | 自有 VLM API，抽帧、编辑、重试、恢复 |
| `scripts/dataset_io.py` | 标签保护、可搬迁导出、评测输入白名单 |
| `scripts/validate_dataset.py` | 结构、时间范围、副本一致性校验 |
| `configs/benchgen_qa_4scene_v1.yaml` | 原四场景基线参数 |
| `src/tools/storm/benchgen/` | 仿真适配、规划、验证、QA 的原核心实现 |
| `vendor/ai2thor/` | 原使用的 AI2-THOR Python SDK；不含 Unity 二进制 |

v2.5 复用历史模块中的 builder，默认入口直接从 reference 导出 v2.5。
