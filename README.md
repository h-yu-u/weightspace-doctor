# WeightSpace Doctor

**A testable, conservative, CLI-first model diagnostics and gradient-free sparse-weight editing research prototype.** MIT · Python 3.10+.

**[中文文档](README.zh-CN.md)** · [Architecture](docs/ARCHITECTURE.zh-CN.md) · [Method](docs/METHODOLOGY.zh-CN.md) · [Security](docs/SECURITY.md) · [Roadmap](docs/ROADMAP.md)

Inspired by [Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) and the authors' [public Experiment 8 logs](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp8_math). Independently implements an evidence-gated process:

```text
                     ┌─────────────────────┐
model / logs / API ──►│ Quick behavior scan │  no weight access
                     └──────────┬──────────┘
                            loop signal?
                          ┌───────┴────────┐
                          No               Yes
                          ▼                 ▼
                       Abstain      Matched good/bad continuations
                                            │
                                  contrastive activations
                                            │
                                  rank suspect layers /
                                  top MLP writer columns
                                            │
                                 reversible sparse patches
                                            │
                                     α dose trials
                                            │
                               unseen target + guard cases
                                            │
                           layer-isolation / random controls
                                            │
                                  manual review / abstain
```

## Implemented

| Command | Purpose | Runtime |
|---|---|---|
| `wspace demo` | Synthetic offline smoke traces | NumPy only |
| `wspace scan` | Loop, cut-off, errors, format and labeled-quality assessment | NumPy only |
| `wspace probe` | OpenAI-compatible API probes (including a running MLX server) | NumPy only |
| `wspace inspect` | Check local checkpoint config and quantization metadata | Local config/index only |
| `wspace weight-audit` | Floating-projection numeric health | HF + PyTorch |
| `wspace localize` | Contrastive residual directions and activation-weighted writer ranking | HF + PyTorch |
| `wspace propose` | Sparse, BF16/FP16-safe, reversible rank-one patches | HF + PyTorch |
| `wspace causal` | Paired teacher-forced selectivity, isolated layers and random controls | HF + PyTorch |
| `wspace validate` | Target and guard performance before/after edits | HF + PyTorch |
| `wspace repair` | Bounded, abstaining repair agent (never auto-deploys) | HF + PyTorch |
| `wspace export` | Explicit manual export to a new directory | HF + PyTorch |

**Not implemented/validated:** direct edits of packed 4-bit MLX weights; universal architectures (only decoder layers exposing `mlp.down_proj` are currently supported); real Qwen3.5-0.8B or 27B end-to-end reproduction; robust factuality or backdoor detection. All example datasets are synthetic and not evidence of improvements to a real model. Weight-space edits are a hypothesis until independent intervention tests demonstrate utility.

## Quickstart (offline, no GPU / downloads)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
wspace demo --out runs/demo
wspace scan --traces runs/demo/traces.jsonl --out runs/demo/report
cat runs/demo/report/report.md
```

## Diagnose an already running MLX 4-bit model, without touching weights

**One-command smoke check** with a built-in `general`, `math`, or `code` probe suite:

```bash
wspace quick --profile general --base-url http://127.0.0.1:8080/v1 \
   --model '<ID-from-server>' --out runs/quick
```

This writes `cases.jsonl`, `traces.jsonl`, `diagnosis.json`, and `report.md`. Built-in fixtures are small **smoke tests, not capability benchmarks**.

```bash
mlx_lm.server --model /path/to/your-mlx-model --host 127.0.0.1 --port 8080
curl -s http://127.0.0.1:8080/v1/models
wspace probe --cases examples/cases.sample.jsonl \
    --base-url http://127.0.0.1:8080/v1 \
    --model '<real-ID-from-/v1/models>' \
    --max-new-tokens 512 --out runs/quick-check
```

Open `runs/quick-check/report.md` and `diagnosis.json`. If you use an internal OpenAI-compatible LLM, replace `--base-url` and supply an API key via `OPENAI_API_KEY`. The model ID must match the server. Server support for deterministic seeds varies.

## Localize and edit a high-precision HF checkpoint

```bash
pip install -e '.[hf]'

wspace inspect --model-dir /path/to/hf-bf16-model
wspace localize --model /path/to/hf-bf16-model \
   --device auto --pairs examples/contrasts.sample.jsonl \
   --layers all --topk 64 --out runs/localization
wspace propose --model /path/to/hf-bf16-model \
   --device auto --localization runs/localization \
   --alpha 1 --topk 64 --out runs/patch
wspace causal --model /path/to/hf-bf16-model \
   --device auto --patch runs/patch \
   --pairs examples/causal_holdout.sample.jsonl --out runs/causal.json
```

**This demonstrates command wiring, not meaningful real-model benchmarks**: replace the small synthetic contrast data with carefully constructed and independently verified good/bad continuations from your model.

For the full agent:

```bash
wspace repair --model /path/to/hf-bf16-model --device auto \
  --pairs examples/contrasts.sample.jsonl \
  --tune examples/tune.sample.jsonl \
  --holdout examples/holdout.sample.jsonl \
  --causal-holdout examples/causal_holdout.sample.jsonl \
  --alphas 0.5,1,2 --topk 64 --out runs/repair-001
```

**Guardrails** include: four or more contrast pairs, loop signal required for the automated workflow, disjoint split IDs, bounded candidate doses, relative-norm and edited-column limits, untouched validation set, random orthogonal controls, independently held-out causal pairs, and manual promotion. The default validation gate expects >=20 target cases and >=10 unrelated guard cases. Sparse patch files contain original and new values, checksums, model ID, and provenance for exact rollback. Export is manual only.

## Mathematical sketch

For a `down_proj` matrix \(W\in\mathbb{R}^{d\times m}\), a normalized unwanted residual activation direction \(u\in\mathbb{R}^{d}\), and selected writer columns \(S\subseteq\{1,\dots,m\}\):

\[
W'_{:,S} = W_{:,S} - \alpha u (u^T W_{:,S}),\quad W'_{:,\neg S}=W_{:,\neg S}.
\]

We derive \(u\) from the mean of bad-minus-good matched residual activations. Writer columns are *hypotheses* ranked by positive activation-weighted contributions \(\max(0,(u^TW_{:,j})\Delta a_j)\), falling back to pure projection if matched activation values are unavailable. Neither this ranking nor a visual weight analysis establishes causality; only controlled interventions supply evidence.

## Development

```bash
pip install -e '.[hf,dev]'
pytest -q
```

Please review [security and evidence restrictions](docs/SECURITY.md) before working with proprietary checkpoints or publishing edited models. No model weights are bundled with the project. Licensed MIT independently of the original authors' experiment data and code.
