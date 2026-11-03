# Security, Privacy and Validation Boundaries

This repository is a **research prototype** for model behavior diagnosis and restricted floating-point weight modification, not a general model safety auditor or a production self-modifying service.

## Safeguards implemented

- No model code is generated/executed from LLM answers. The optional planner returns structured JSON, validated against allowed layer IDs and alpha values. Bad output fails closed.
- All sparse column edits are restricted by matrix dtype, shape, alpha, fraction of columns, and relative update norm. Packed 4-bit weights are not supported for editing.
- Patches are auditable: SHA256 for original tensors and selected columns, explicit before/after values, operation parameters, and discovery sample IDs. `.npz` is loaded with `allow_pickle=False`.
- All patch target tensors are checked before the first write. Partial write errors trigger attempted rollback; normal evaluation uses a context manager for rollback.
- `export` is manual and writes to a separate new/empty directory; it never replaces the source model.
- Agent failure branches return `abstained_*` or `not_approved`. No training, gradients, optimizer, or forced adaptation when a loop symptom is absent.
- Final model-task evaluation compares exact paired IDs on `target` and `guard` splits; by default 20 target and 10 guard cases are required. A one-sided exact paired sign test must support the target improvement. Independent causal contrasts and negative controls are required for a review recommendation.

## Known limitations and threats

1. **Untrusted models**: `transformers.from_pretrained(..., trust_remote_code=False)` is not a complete model supply-chain security solution. Use trusted model sources and check revisions/checksums/licenses.
2. **Untrusted patch input**: apply patches only from trusted collaborators or independently checked processes. Manifest/NPZ sizes are not fully bounded; hostile files may cause memory exhaustion. `np.load(..., allow_pickle=False)` prevents pickle loading, but this is not a hardened format sandbox.
3. **Untrusted network**: an HTTP `probe` sends user prompts to the configured API. Do not send customer secrets or proprietary prompts to an untrusted provider. Use localhost HTTPS/private transport as appropriate. The tool uses the `OPENAI_API_KEY` environment variable by default, never stores keys in reports.
4. **No safe concurrent hot-patching**: never call an editing method concurrently on a model handling other requests. An exclusive model instance is required. Whole-process crashes, out-of-memory signals, or GPU errors can interrupt rollback. Load the pristine checkpoint again after such an event.
5. **No mechanistic proof from ranking**: a writer score based on `|uᵀ W_j|` is an association. Even causal perturbations can have indirect and distributed effects.
6. **No full competency/safety coverage**: a few handpicked cases and simple format checks cannot detect hallucinations, hidden capabilities, subtle harms or all regressions. High-stakes model deployment requires a much broader evaluation suite.
7. **Statistical limits**: a paired sign test p<0.05 and negative controls do not correct for all selection bias, semantic leakage or multiple alpha trials. Repeat with independent data, multiple random seeds and larger samples.
8. **No arbitrary quality oracle**: expected answer uses strict normalized-text equality and boxed extraction. `expected_regex` checks final-answer syntax and is not an arithmetic/semantic verifier. Use domain-specific validators for your real problems.
9. **Context length / tokenization**: model-specific thinking delimiters, tokenizer behavior and template parameters can change diagnostics. Verify the actual request template and model generation configuration before causal experiments.
10. **Data retention**: `probe` writes full prompt/output JSONL in the requested output directory by default; sanitize or store securely. `localize` only writes aggregate directions and metadata, not full activation traces.
11. **Exposure risk**: do not bind local MLX servers to `0.0.0.0` unless required authentication and network isolation exist. Prefer loopback for experiments.
12. **No measured large-model validation yet**: passing synthetic unit tests only establishes implementation behavior on small CPU fixtures, not generalization to Qwen3.5-0.8B, Qwen 27B or any deployment model.

## Recommended release checks

1. Inspect `run.json`, all `report.md`, `holdout_metrics.json`, and `independent_causal.json`.
2. Check real examples: independent target labels, negative and positive controls, decoding settings, refusal and cut-off edge cases.
3. Run several seeds and larger unrelated guard tests; compare the patch against prompt/template/decoding fixes and ordinary SFT/DPO when relevant.
4. Inspect full model outputs, safety regressions, token throughput, and domain benchmarks.
5. Export manually to a new directory and repeat benchmarks on the exported checkpoint, including after any requantization.
6. Maintain the original checkpoint and patch provenance so a human can roll back deployments.

## Responsible disclosure

Open an issue with a minimal reproducible case that does not contain sensitive model weights or user data. For vulnerabilities involving secret leakage or unauthorized remote execution, contact the repository maintainer privately rather than posting exploit details publicly.
