# 复盘原始 Experiment 8：Astra 如何选中 Qwen 第 15、19 层与 128 列

**资料范围**：2026-11-02 作者公开的 `lm_understand-logs/exp8_math`，尤其是 p50c 的 `final_report.md`、`experiment_history.md`、`notes_before_pass26.md`、`frozen_weights_manifest.json`。此文是按作者记录整理的实验回溯，**不是本项目已经运行过的 Qwen 实验**。

参考链接：

- [完整研究](https://rikiyat.github.io/blog/weight-space-programming/)
- [实验目录与评分规则](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/README.md)
- [p50c 实验历史](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/runs/p50c/work_notes/experiment_history.md)
- [p50c 最终报告](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/runs/p50c/work_notes/final_report.md)
- [最终选中索引](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/runs/p50c/work_notes/frozen_weights_manifest.json)

## 1. 先判断是不是能力缺失

首轮对 32 道 GSM8K 练习题和 64 道 MATH 练习题进行原始模型检测，仅 **11/96** 正确，**85/96** 回答触及 32768-token 上限；错误尾部包含大量重复短句。若主要问题是**无限重复**，能力可能被错误的行为模式掩盖，而不是数学知识本身完全缺失。

因此第一步没有盲目进行全量权重搜索，也没有直接提出“增强数学神经元”。先定义可测量的故障类别：**loop**。

## 2. 教师强制读取正常/异常的内层行为

早期使用匹配的正常和重复续写案例，通过 teacher-forced forward 读取隐藏状态、MLP 中间激活和部分注意力信息。实验中一个细节：最初的数据构造因 chat template 添加了空 thinking block，作者随后调整为原生模板格式。**这不是小问题**：如果两个续写的 token 对齐或 prefix 处理错误，你提取的可能是模板差异，而非真正的循环方向。

在第 16、20 层残差输出，得到与循环相关的差分向量。具体可采用：

\[
u_l = \mathrm{Normalize}\left(\mathbb{E}_i[h_l(\mathrm{bad}_i)-h_l(\mathrm{good}_i)]\right).
\]

该向量的含义是“在当前小样本集合下，与重复续写同向的残差方向”，**不是某种能在所有任务中唯一标识重复的神经电路**。

## 3. 逐步试探，不是一眼看出正确层

实验历史出现多轮候选和对照：

1. 第 19 层 MLP 神经元 #11 的局部消融，仅在小样本上带来微小、统计不够稳定的提升。
2. 第 19 层 MLP 的单层 rank-one direction removal，在小样本上仍只是有限改善。
3. 合并第 15、19 层的 MLP 输出投影方向抑制，开始出现更明显的重复下降信号；但部分早期试验复用了同一批 32 题，**不能当作独立证据**。
4. 考察注意力头 15.h7 / 19.h5 的替代假设，观察到某些与重复相关的指标，但因缺少更稳定的生成收益，被放弃。
5. 比较多档剂量、单独层、随机正交等范数扰动与新的独立数据，才冻结最终候选。

所以两个层号不是由静态权重分布的一个异常值直接“算出来”，而是**故障定义 → 层内激活对比 → 神经元 writer 选择 → 干预 → 对照 → 泛化检验**形成的结论。

## 4. 具体改了什么

冻结方案：

```text
layers: 15, 19
module suffix: mlp.down_proj.weight
shape per layer: [1024, 3584]
selected columns: 64 per layer
alpha: 6
method: selected-column projection suppression
```

数学表达：

\[
W'_{:,S}=W_{:,S}-6u(u^TW_{:,S}),\quad|S|=64.
\]

修改元素上限 2 × 1024 × 64 = **131,072**（相当于整个 0.8B 模型约 0.016%），根据作者记录，BF16 下真正改变的有 **130,188**。两层完整 down-proj 矩阵的相对 Frobenius 变化分别约 **10.64%**、**8.11%**，因此“只动 0.016% 参数”**绝不等于整个系统作用很小**。

## 5. 为什么认为这不是偶然

作者保留了以下证据：

- 正常 vs 循环续写的 teacher-forced 概率差：不同剂量下循环概率的抑制比正常续写更明显。
- 第 15 / 19 层分别干预后，观察到各自对目标行为的贡献。
- 正交随机等范数对照没有产生同等级别的提升。
- 冻结候选之后，对 **1,152** 条未用于选择编辑的练习问题进行成对评测：原始 **10.6%**，编辑后 **60.4%**，有 **579** 次错误恢复和 **5** 次正确退化。
- 正常/错误结束、强制截断、提前退出等指标也被分别统计，避免把“只会尽快 EOS”误认为改善数学能力。

这些是**原作者的数据**，不是 WeightSpace Doctor 测出来的数据。原论文还进行了独立 MATH-500/GSM8K 等基准测试，但要复现论文应使用相同抽样温度、token cap、prompt、分数解析器、模型版本和任务集合。

## 6. 由原始实验推导出的工程验收原则

| 论文启示 | 本工程对应能力 | 缺失或还需增强的部分 |
|---|---|---|
| 先找行为循环症状 | `scan`、`quick`、`repair` 早期 triage | 更全面的推理任务和采样配置控制 |
| 匹配两类续写提取方向 | `capture_layer_features` 和 `contrast_statistics` | 严格 token 对齐、跨任务稳定性估计 |
| 找到 MLP writer | `top_writers`，激活加权有符号贡献 | 遮蔽/消融 writer 的独立验证 |
| 少数列 rank-one 编辑 | `build_patch`，哈希和可逆写入 | 多种投影、稀疏/低秩策略对比 |
| 剂量实验 | `repair` 的 alpha sweep | 每个剂量的更多重复采样 |
| 单层隔离、随机对照 | `causal` + `build_random_control` | 多种随机方向、跨模型负对照 |
| 独立练习集与 Benchmark | `tune` / `holdout` / `guard` + sign test | math-verify、IFEval、完整 benchmarks、跨种子 |
| 权重可能有不可见副作用 | 禁止自动部署、需要 guard | 广泛长期/跨任务回归指标 |

**最终工程结论**：把“发现哪个神经元”改写成一个可自动验证的科学假设生成问题。无论 LLM 把候选解释得多漂亮，`repair` 必须允许“没有足够证据，不编辑”这一正常结果。
