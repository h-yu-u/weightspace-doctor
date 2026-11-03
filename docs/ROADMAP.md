# Roadmap and reproducibility milestones

## v0.1.0 — present research prototype

- [x] Offline trace scan and JSON/Markdown reporting.
- [x] OpenAI-compatible HTTP probing; run on an existing MLX inference server without loading/editing quantized weights.
- [x] HF preflight and float projection health audit.
- [x] Paired residual contrast localization, MLP input activation capture and activation-weighted writer column ranking.
- [x] Restricted sparse rank-one projector edits and exact rollback checks.
- [x] Alpha sweep, unseen validation sets, negative controls, independent causal continuation tests, bounded planner, no automatic deployment.
- [x] Unit tests on small synthetic models and CI template.
- [ ] Independent real-Qwen benchmark replication (not yet executed).

## v0.2.0 — evidence quality and realistic Qwen replication

- [ ] Reproduce author Experiment 8 on pinned Qwen3.5-0.8B with a publicly documented chat template, sampling protocol, input budgets and actual benchmark results.
- [ ] Validate activation-weighted writer contribution ranking against pure `|uᵀW_j|` and neuron ablation on a real model; calibrate mixed-activation and token-boundary effects.
- [ ] Introduce judge plugins (`math-verify`, program execution sandbox, component-library API tests) instead of strict string matching only.
- [ ] Layer-wise false-positive baselines and alternative contrast direction estimators; detailed confidence intervals and pre-registered holdouts.
- [ ] Test BF16/FP16 and quantized-base / requantized-candidate parity.
- [ ] Configurable chat templates and thinking boundaries per architecture, with fixtures drawn from publicly available model cards.
- [ ] HTML diagnostics for side-by-side error groups and per-case difference exploration.

## v0.3.0 — modular adapters and scale

- [ ] Adapter interfaces for Gated DeltaNet/attention hybrids, MoE experts and additional decoder families.
- [ ] Better activation collection with memory-friendly prefix caching, bounded batches and explicit token/window alignment.
- [ ] Run on GPU/distributed inference backends with pluggable scheduling and limit enforcement.
- [ ] Explicitly supported MLX float edit adapter; no promise of direct packed 4-bit editing.
- [ ] Optional conversion workflow: HF BF16 → edited BF16 → MLX quantization → post-quantization guard evaluation.
- [ ] CLI-generated reproducibility manifests, model revision pinning, multi-seed confidence reports and safe artifact retention.

## Not planned without strong evidence

- Automatically modifying any remote or closed-source model through API-only access.
- Claiming that unusual weight magnitudes imply a security backdoor.
- Editing a live model in service without transactional isolation and approval.
- Advertising guaranteed capability gains or a one-shot cure for knowledge gaps.

## Definition of done for Qwen3.5-0.8B replication

Publish a separate experiment run containing exact source model revision, prompt+tokenization protocol, honest baseline and edited benchmark scores, all generation settings, activation/direction provenance, selected layers/columns, patch hashes, negative controls, independent held-out pairs, multi-seed reports and safety regression checks. Until then, project tests remain engineering smoke tests only.
