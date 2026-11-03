# WeightSpace Doctor v0.1.0 — Initial Open-Source Preview

WeightSpace Doctor is an experimental, conservative CLI-first toolkit for behavior diagnosis, contrastive localization and reversible sparse-weight editing of language models.

## Highlights

- Analyze generation traces to flag repeated reasoning, truncation, format failures and task-specific quality issues.
- Run quick probe suites against OpenAI-compatible APIs, including locally served MLX models, **without editing model weights**.
- Inspect Hugging Face model metadata, analyze matched good/bad continuations and rank suspect MLP writer columns.
- Build reversible sparse high-precision patches with tensor checks, bounded update magnitude and explicit export.
- Evaluate edits using held-out target/guard cases, directional comparisons, dose-response tests and negative controls.
- Provide structured JSON and readable Markdown diagnosis reports, a modular CLI, bilingual documentation and CI.

## Important limitations

- The project has not reproduced the authors' Qwen3.5-0.8B benchmark results. Initial tests use synthetic traces and small CPU models.
- Editing quantized MLX 4-bit checkpoints directly is **not supported**; API-based diagnosis is supported.
- The weight-localization and editing backend currently requires suitable HF/PyTorch models exposing `mlp.down_proj`.
- Patches are **research hypotheses**, not guaranteed improvements. They must pass independent validation and manual review before production use.

## Install and quick smoke test

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
wspace demo --out runs/demo
wspace scan --traces runs/demo/traces.jsonl --out runs/demo/report
```

For high-precision HF model experiments, install `pip install -e '.[hf]'` and read the README and `docs/ARCHITECTURE.zh-CN.md`.

## References

- [Weight-Space Programming (original article)](https://rikiyat.github.io/blog/weight-space-programming/)
- [Public Experiment 8 research logs](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp8_math)

This independent open-source implementation is not affiliated with the original authors.
