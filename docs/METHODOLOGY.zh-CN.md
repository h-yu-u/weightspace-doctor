# 权重空间编辑：数学、定位证据与局限

## 1. 原作者第 8 组实验：算法到底是什么

公开参考：

- [原文章](https://rikiyat.github.io/blog/weight-space-programming/)
- [原实验说明 `exp8_math/README.md`](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/README.md)
- [50-pass 的实验总结](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/runs/p50c/work_notes/final_report.md)
- [最终 64+64 个 writer 列索引](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/runs/p50c/work_notes/frozen_weights_manifest.json)

注意：权重空间编辑论文是**早期实验**；公开日志展示了大量试错及控制实验，并不意味着一个通用自动定位器已经被严格证明。

### 1.1 先找“行为缺陷”，而不是先找“坏神经元”

原始 Qwen3.5-0.8B 在特定评测设置下出现严重的无限推理和截断。作者研究中先在 96 个练习问题上看到 85 个达到 32768-token 上限，诊断出重复表达/推理循环，再开始排查激活。

该观察很重要：如果模型只是不知道某种知识，直接压制“重复方向”没有理论上的必然收益。

### 1.2 配对 teacher-forced 续写，估计异常方向

原始实验通过同一个问题的**正常推进续写**和**反复绕圈续写**作配对前向，在第 16、20 层残差输出附近比较激活。我们把它抽象成：

\[
\delta_{l,i} = h_l(x_i,\text{bad}_i)-h_l(x_i,\text{good}_i)\,,
\quad
u_l = \frac{\frac1N\sum_i \delta_{l,i}}{\|\frac1N\sum_i \delta_{l,i}\|_2}.
\]

其中 `h` 是同一解码层输出尾部若干 token 的平均向量（实现上的近似）；`u` 是“异常减正常”方向。这里是**关联特征**而非证明唯一的机制。

**与作者做法的区别**：作者在精心设计的生成/教师强制采样反馈下最终形成方向，本实现采用了通用 `tail_tokens` 均值和更简单的分层得分。这是可运行的工程近似，不声称数学与作者日志逐操作完全同一。

### 1.3 在 MLP down_proj 中找“往异常方向写入”的列

如果候选 MLP 输出投影权重为 \(W_l\in\mathbb{R}^{d\times m}\)，第 \(j\) 个隐藏单元对残差的写入是矩阵第 \(j\) 列。它在异常方向 \(u\) 上的投影幅度：

\[
\operatorname{writerScore}(l,j)=\left|u_l^TW_{l,:,j}\right|.
\]

工程当前同时在 `down_proj` 上采集 MLP 输入激活（预 Hook），计算每个隐藏单元的成对平均激活变化：

\[
\Delta a_{l,j} = \mathbb{E}_i[a_{l,j}(\mathrm{bad}_i)-a_{l,j}(\mathrm{good}_i)].
\]

然后优先根据**正向写入异常方向的激活加权贡献**排名：

\[
\operatorname{activatedWriterScore}(l,j)
= \max\big(0,(u_l^TW_{l,:,j})\cdot\Delta a_{l,j}\big).
\]

当缺少激活数据或没有正向写入列时，才退化到 `|uᵀW_j|` 排名。每个层的残差差分还会用跨 pair 相干度（`coherence`）与正常激活幅度归一化（`score`）进行启发式排序。

**关键局限**：激活加权的平均贡献仍有相消、长度不对齐、token 位置和上下文混淆风险。原始论文的精细化模型反馈和因果对照并不能被这个统计打分完全替代。它只是**改善候选列排序**，还需要逐层隔离、剂量与独立控制实验。

这里的梯度为零：这是统计和前向插桩，不是反向传播训练。

### 1.4 稀疏秩一投影编辑

令 \(S_l\) 为上一步选择的 MLP 列，给定剂量 \(\alpha\)：

\[
\boxed{W'_{l,:,S_l} = W_{l,:,S_l} - \alpha u_l\big(u_l^TW_{l,:,S_l}\big)}
\]

其他列不动。改变的浮点权重条目上限为 \(d\times |S_l|\) 每层。与完全做一层 W' = W - α u(uᵀW) 相比，这是局部、稀疏的变体。

原作者最终选择：

- Qwen3.5-0.8B 第 **15**、**19** 层的 `mlp.down_proj.weight`；
- 每层 **64** 列，矩阵是 **1024×3584**；
- 最终剂量 **α=6**；
- 共 2×1024×64 = **131,072 个触及的浮点元素**；
- 在 BF16 量化后实际变化的元素数可能低于这个数（公开日志给出 **130,188**）。

原作者的独立 held-out practice 对照、随机正交控制、完整/隔离层实验和不同 α 剂量都是同样不可缺少的工程证据。**本工程绝不固定所有模型都选择 15、19 层或 α=6**。通过多层激活定位并在有限 α 集中选择。

### 1.5 不用梯度不等于没有实验成本

在作者 p50c 实验中，使用了**50 次** forward-pass 工具预算，并结合大规模练习/留出样本进行对比。按“先探索、后验证”拆分预算很重要。仅依靠一个推理模型看几列数字并不能说明所选编辑有效。

## 2. WeightSpace Doctor 的证据工具链

| 阶段 | 工具 | 是否需要运行模型 | 输出证据 |
|---|---|---:|---|
| 行为诊断 | `wspace scan` | ❌（已有日志） | Loop/Cut/Format/Accuracy 等 |
| 远程基线 | `wspace probe` | ✅（HTTP 请求） | JSONL baseline + Markdown |
| 结构检查 | `wspace inspect` | ❌ | 模型层数、量化类型 |
| 张量健康 | `wspace weight-audit` | ❌（只读权重） | 非有限值、列范数异常 |
| 激活定位 | `wspace localize` | ✅（teacher-forced forward） | 可疑层 + writer 列 + 方向 |
| 编辑 | `wspace propose` | ❌（只读权重+计算 patch） | 有校验和的 sparse patch |
| 因果干预 | `wspace causal` | ✅（配对对数概率） | delta good/bad, selectivity |
| 泛化验证 | `wspace validate` | ✅（生成） | accuracy + loop + cut + paired sign p |
| 自动流程 | `wspace repair` | ✅（多阶段） | 轨迹、证据、决策、人工审阅建议 |

### 2.1 selectivity 的含义

在相同提示下测量两个续写的平均 token 对数概率：

\[
\Delta_{good} = \log P_{edited}(good)-\log P_{base}(good)
\]

\[
\Delta_{bad} = \log P_{edited}(bad)-\log P_{base}(bad)
\]

\[
\mathrm{Selectivity}=\Delta_{good}-\Delta_{bad}.
\]

理想情况下，编辑减少 bad 概率更多（\(\Delta_{bad}<0\) 且 selectivity > 0）。但它还不能证明推理能力提高，仍要看实际任务正确率与保护任务退化。

**注意**：实现对 prompt prefix 与 continuation 分别 tokenization 再拼 token IDs，减少边界不确定性；此方法是一种固定比较协议，可能不同于对拼接后原始字符串一次 tokenization 的真实分布。

### 2.2 假阳性控制

- 层消融（`isolated_layers`）：两个层是否都确实贡献效果？
- 随机正交等范数对照（`negative_control_patch`）：仅制造相同强度扰动，会不会也提高分数？
- 剂量反应（`alpha` sweep）：0.5 / 1 / 2 / 可选 4 / 6 并不是越大越好。
- 数据隔离：发现样本不能出现在 tune / final holdout / causal holdout；Agent 会核对 ID。**同义改写的语义重合需要人工判定。**
- 答案正确率：未提供标准答案时不判定准确度；“减少输出长度”不是成功标准。
- 精确配对 sign test：对 discordant paired cases 的 rescue vs regression 运行单侧精确二项检验；默认 p<0.05 才把有改善的结果列为可以建议人工审核（仍非发布资格）。小样本不能取代多次独立复现。

## 3. 为什么 4-bit MLX 不能直接套这个算法

MLX 量化矩阵是离散 packed 值并伴随 group scales / bias。`W-u(uᵀW)` 产生新的浮点矩阵，不能直接写入一个 packed int4 的原位 bits，且会导致量化误差、group 统计改变。当前项目特意拒绝对 packed dtype 权重编辑。

建议：

```text
生产可用 MLX 4-bit
       │
       └──► OpenAI 兼容接口生成用例 → 快速报告 (不改模型)

需要修复、找到匹配循环证据后：
       │
       ▼
可获得的 HF BF16/FP16 源模型（最好单独 GPU/大内存设备）
       │
       ▼
activation localization + sparse editable patch
       │
       ▼
独立验证 + 显式导出 BF16/FP16 新目录
       │
       ▼
MLX 转换器重新转换、重新量化
       │
       ▼
在 MLX 服务中再次对齐评测 → 再决定是否发布
```

不能将 Qwen3.5-0.8B 的异常方向与层号直接迁移到你的 Qwen 27B 或其他具体模型。模型结构、训练语料和量化形式不同，都需重新定位并检验。
