# Engineering lessons (living record)

## 2026-11-03 — Why first detect a measurable failure

A baseline model that does not suffer a repetition loop should not be subjected to a blind repetition-direction edit. The automated agent gates early when no such symptom is observed. A small smoke suite is not evidence that the model is problem-free.

## 2026-11-03 — Activation direction is a hypothesis, not a circuit diagnosis

A contrastive mean vector can conflate behavior with length, wording and token-position differences. In particular, ranking MLP columns by `abs(u.T @ W)` ignores whether those neurons are active. The current implementation therefore captures `down_proj` inputs and ranks signed activation-weighted contributions; this remains associational and requires independent intervention tests.

## 2026-11-03 — Code can pass toy tests without reproducing an experimental result

A 0.8B Qwen paper result depends on architecture, chat template, sampling controls, token budget, evaluation grading and data splits. We do not have a real-model reproducibility run in this project, so unit tests only establish implementation behavior on synthetic fixtures.

## 2026-11-03 — Quantization is not just a dtype label

Packed bits and per-group scales change how weights are stored. Editing dense floats and then requantizing is a separate process from editing an MLX 4-bit model in place. Floating `.weight` objects in a custom quantization wrapper are not enough to authorize edits; check module type and quantization metadata as well.

## 2026-11-03 — Scoring a reasoning-model answer requires preserving delimiters

Qwen's `<|/think|>` marker can disappear when decoding with `skip_special_tokens=True`. Preserve a raw decoding copy when extracting only the final answer for task accuracy. Failure to do this can wrongly label correct outputs as failures.

## 2026-11-03 — A small observed gain is not a validation win

Use paired case IDs, count rescues and regressions, and apply a paired sign test (currently p<0.05) before recommending manual review. This is still insufficient for publication if the holdout is semantically leaked, too narrow, or already used for manual tuning.
