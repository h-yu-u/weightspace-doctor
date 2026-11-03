# WeightSpace Doctor（权重空间诊断与修复）

> **可运行的研究原型 v0.1.0** · MIT · Python 3.10+ · CLI 优先 · 不依赖梯度下降
>
> 目标不是让 LLM 随便改权重，而是提供**快速症状诊断 → 有证据的权重定位 → 受约束的可逆编辑 → 独立验证 → 人工审批**的完整工程链路。

[English](README.md) | [架构设计](docs/ARCHITECTURE.zh-CN.md) | [算法与原始证据](docs/METHODOLOGY.zh-CN.md) | [安全与验证](docs/SECURITY.md) | [论文实验逆向复盘](docs/EXPERIMENT_8_TRACE.zh-CN.md) | [路线图](docs/ROADMAP.md) | [发布指南](docs/RELEASE.zh-CN.md)

## 项目边界与当前能力

| 层级 | 目标 | v0.1.0 状态 |
|---|---|---|
| L0 静态预检 | checkpoint 类型、量化状态、层数、大小 | ✅ 不加载模型的 `inspect` |
| L1 快速行为诊断 | 循环、截断、格式、拒答迹象、正确率、速度 | ✅ `scan` / HTTP `probe` |
| L2 权重健康审计 | NaN/Inf、全零矩阵、异常列范数 | ✅ HF 浮点权重 `weight-audit` |
| L3 机制候选定位 | 成对正常/异常残差差分、层排名、激活×权重 MLP writer 列排名 | ✅ `localize`，**只是候选机制** |
| L4 权重空间编辑 | top-k 列的秩一投影抑制，版本化、校验、回滚 | ✅ `propose` / `export` |
| L5 因果对照 | 编辑前后 teacher-forced log P、层隔离、随机正交等范数对照 | ✅ `causal` |
| L6 Agent 自动循环 | 诊断、候选 α 搜索、独立任务评估、门禁、拒绝无证据编辑 | ✅ `repair`，不自动上线 |
| MLX / 4-bit 直接权重编辑 | 原位量化修改 | ❌ **不支持**；仅通过 API 进行快速行为诊断 |
| 跨架构通用性 | 所有 Transformer / MoE / 视觉模型 | ❌ 当前支持符合 HF Decoder + `mlp.down_proj` 的权重结构，需实际验证 |

**测试状态**：CI 和单元测试覆盖 Python/CPU 的诊断、定位、编辑、回滚及门禁核心逻辑；仓库中的演示数据是**人为构造样本**。**未**在本交付环境中下载/运行 Qwen3.5-0.8B、27B 或 MLX 模型，**不**声称已经复现论文指标，也**不**对修改后模型的能力改善做任何保证。

## 0. 最快体验：离线运行，不要下载模型

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

wspace demo --out runs/demo
wspace scan --traces runs/demo/traces.jsonl --out runs/demo/diagnosis --model synthetic-demo

# 生成机器可读 diagnosis.json 与人类可读 report.md
cat runs/demo/diagnosis/report.md
```

你也可以用自己的推理日志，直接进行扫描：

```bash
wspace scan --traces examples/traces.sample.jsonl --out runs/fast-check
```

`traces.jsonl` 每行至少包含 `id`、`text`；推荐另带 `prompt`、`finish_reason`、`output_tokens`、`max_output_tokens`、`elapsed_ms`、`expected`、`split`（target/guard）、`tags`。示例见 `examples/traces.sample.jsonl`。

> **快速 ≠ 能读出所有模型缺陷**。第一阶段主要发现可观察的症状，不能仅凭权重的统计量判断模型是否具有幻觉、后门、隐藏偏见或新知识缺失。未标注正确答案时，不会虚构“正确率”。

## 1. Mac + MLX 4-bit：直接诊断当前运行的模型

**最省事的命令**：不必提前准备测试问题，使用内置的 `general` / `math` / `code` 小型客观烟雾测试集直接运行：

```bash
wspace quick --profile general \
  --base-url http://127.0.0.1:8080/v1 \
  --model '<服务端真实 model id>' \
  --max-new-tokens 512 \
  --out runs/mlx-quick
```

这会同时保存 `cases.jsonl`、`traces.jsonl`、`diagnosis.json`、`report.md`。内置题库**只是烟雾检测，不代表任何行业 Benchmark 或复杂推理能力**；用于找循环/截断等症状，真正业务评测应使用自定义题库。

若你已经在 MLX Desktop / `mlx_lm.server` 中提供 OpenAI 兼容 HTTP 接口，可用 `probe` 复用，无需重新下载高精度权重。

```bash
# 只允许本机调用；模型路径换成你的 MLX 4-bit 目录
mlx_lm.server --model /path/to/your-mlx-4bit-model --host 127.0.0.1 --port 8080

# 新终端：先查询服务端真实 model id，避免使用错误的名字
curl -s http://127.0.0.1:8080/v1/models

wspace probe \
  --cases examples/cases.sample.jsonl \
  --base-url http://127.0.0.1:8080/v1 \
  --model '<从 /v1/models 读取的真实 ID>' \
  --max-new-tokens 512 \
  --out runs/mlx-quickcheck
```

结果包含 `traces.jsonl`、`diagnosis.json`、`report.md`。如使用企业内部 OpenAI 兼容 API，可将 `--base-url` 换成企业地址、设置 `OPENAI_API_KEY`。**注意** HTTP 后端可能不支持 `seed`：实现会在明确的 HTTP 400 `seed` 错误后重试无 seed，请勿将此类结果当作严格可重复的随机采样实验。

**注意**：模型本身不存在无法仅靠一套固定问题“扫描出来的全部问题”。请替换为自己业务的测试用例。`expected` 是严格字符串比对；含自由表达或数学等价的评测需要扩展 `metrics.correctness` 的领域评测器。

## 2. 用 BF16/FP16 权重定位重复推理故障

```bash
# 仅需要读取权重的任务才安装 PyTorch + Transformers 等依赖
pip install -e '.[hf]'

# 先看本地 checkpoint 元信息（不读取全部权重）
wspace inspect --model-dir /path/to/hf-fp16-model

# 审计权重的数值健康：不会推断行为错误
wspace weight-audit --model /path/to/hf-fp16-model \
  --device cpu --layers 0,1,2 --out runs/audit.json

# contrast pairs 是对应同一个问题的“正常 vs 循环”两种续写
wspace localize --model /path/to/hf-fp16-model \
  --device auto --pairs examples/contrasts.sample.jsonl \
  --layers all --topk 64 --out runs/localized

# 将发现的坏行为方向从排名最高的两个 MLP 层部分列中减去
wspace propose --model /path/to/hf-fp16-model \
  --device auto --localization runs/localized \
  --alpha 1 --top-layers 2 --topk 64 --out runs/candidate_patch

# 量化编辑对正常/异常续写对数概率的选择性影响
wspace causal --model /path/to/hf-fp16-model \
  --device auto --patch runs/candidate_patch \
  --pairs examples/causal_holdout.sample.jsonl \
  --out runs/causal.json
```

`localize` 是无梯度的关联定位：比较同一问题下正确推进/异常循环续写的层输出向量，收集 `down_proj` 输入激活，并按 `max(0, (uᵀW_j) × Δactivation_j)` 给写入列排名，缺少激活数据时才退化为纯投影排名。**它不能单独证明哪层、哪列导致问题**；还需要 `causal` 中的层隔离、剂量实验和随机方向阴性对照。

所有编辑默认**仅限未量化浮点权重**。4-bit/8-bit 模型**不可**通过 `propose`、`repair`、`export` 直接安全修改；推荐工作流是：**在能容纳高精度权重的机器编辑 HF BF16/FP16 基座 → 独立评测 → 导出新检查点 → 使用相应 MLX 转换/量化工具重新量化**。千万不要对当前 MLX `.safetensors` 的 packed bits 直接套用此公式。

## 3. 一条命令跑完整受约束 Repair Agent

准备四份相互独立的数据：发现方向用的 `contrasts`、候选 α 选择用的 `tune`、**不能用于调参**的 `holdout`、检验机制选择性的 `causal-holdout`。

```bash
wspace repair \
  --model /path/to/hf-fp16-model --device auto \
  --pairs examples/contrasts.sample.jsonl \
  --tune examples/tune.sample.jsonl \
  --holdout examples/holdout.sample.jsonl \
  --causal-holdout examples/causal_holdout.sample.jsonl \
  --alphas 0.5,1,2 \
  --topk 64 \
  --max-new-tokens 2048 \
  --out runs/repair-001
```

这个例子展示**文件格式与命令**，不是实证数据，不能用少量示例去判断真实模型能否安全发布。默认门禁至少要求 **20 条目标任务 + 10 条保护任务**，并且要求独立机制对照；不满足时只生成候选或直接拒绝编辑。

Agent 运行规则：

1. 先运行原始模型，确认确有重复异常；若没有，**拒绝盲目编辑**。
2. 使用正常/异常配对数据分析多层残差表征，筛出 MLP writer 候选。
3. 每次只修改少数列，强制限制相对更新范数和改动列比例，比较有限组 `alpha`。
4. 在最终未参与调参的独立任务集验证 target 任务提升、guard 能力无明显退化。
5. 比较正交等范数随机对照、层隔离和独立教师强制概率指标。
6. 结果仅为 `recommend_manual_review` / `not_approved` / `abstained_*`，**绝不自动部署**。

可以额外配置使用独立 LLM 作为 Planner（任意 OpenAI 兼容服务）：`--planner-base-url` 和 `--planner-model`。LLM 只能从白名单层和预定义 `alpha` 里提交 **JSON 建议**，不能执行随意生成的 Python、shell 或任意参数更新脚本。默认不依赖 LLM，确定性算法可以独立运行。

## 4. 明确人工审批后导出新检查点

```bash
wspace validate --model /path/to/hf-fp16-model \
  --device auto --patch runs/candidate_patch \
  --cases examples/holdout.sample.jsonl \
  --out runs/manual-review

# 仅在你检查指标、抽样和回归全部完成后，手动执行
wspace export --model /path/to/hf-fp16-model \
  --device auto --patch runs/candidate_patch \
  --out /path/to/new-edited-hf-checkpoint
```

Patch 格式：`manifest.json` + `values.npz`，保存原/新列值、基础权重哈希、列值哈希、模型标识、编辑参数和发现样本 ID。应用前先对**所有目标张量校验**，使用上下文管理器确保评测后精确恢复；当哈希不匹配时拒绝修改。**注意** `export` 是人工触发的操作，不意味着门禁通过；该命令自身不会代替人工审批或检查诊断结论。

## 数据格式

**`cases.jsonl`**（任务评测）

```json
{"id":"target-01","prompt":"12+30 等于多少？只输出数字。","expected":"42","split":"target","tags":["math","short-answer"]}
```

**`contrasts.jsonl`**（同一问题的成对续写；必须是模型真实发生过的异常与可信正常续写，不推荐凭空杜撰）

```json
{"id":"contrast-01","prompt":"Compute 12+30.","good":"12 plus 30 is 42. Final answer: 42.","bad":"Let me reconsider. Let me reconsider. Let me reconsider. Let me reconsider."}
```

**`traces.jsonl`**（推理日志；已存在也能诊断）

```json
{"id":"target-01","prompt":"12+30 等于多少？","text":"42","finish_reason":"stop","output_tokens":1,"max_output_tokens":256,"elapsed_ms":123,"expected":"42","split":"target","tags":["math"]}
```

`expected_regex` 可代替 `expected` 做**格式或特定字符串**检查（只检查最终回答）；这不是通用事实性验证器。注意不要在 `target` 和 `guard` 或 discovery/tune/holdout 中复用同一问题；目前 Agent 检查 ID 不重叠，**语义相似或重述的数据泄漏仍需使用者负责**。

## 开发与测试

```bash
pip install -e '.[hf,dev]'
pytest -q
python -m compileall -q src
```

计划贡献方向：MLX 原生浮点适配器、更多架构适配器、严格数学答案验证器、编排式工具调用、HTML 可视化、显著性检验与多种随机种子，以及 Qwen3.5-0.8B 的复现实验。详见 [路线图](docs/ROADMAP.md)。

## 研究来源与引用

- Rikiya Takehi 等，2026-11-02，[Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)。
- 作者 [公开实验日志与 Qwen 实验记录](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp8_math)；特别是 `runs/p50c/work_notes/final_report.md`、`frozen_weights_manifest.json`、`exp8_math/README.md`。

本仓库是**独立开源工程原型**，未复制或包含作者的实验权重、私有 Harness 或原始数据。算法以研究公开日志为启发，保留可归因的原始论文引用；论文效果不代表本项目实测效果。Apache、MIT 或 CC BY 数据的使用应分别遵守各自具体文件声明。

许可证：MIT（见 `LICENSE`）。
