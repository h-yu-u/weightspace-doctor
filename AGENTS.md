# Repository instructions for AI coding agents

## Mission

Build a trustworthy **evidence-gated** model failure diagnostic and reversible repair toolkit. Treat all LLM-generated layer plans and weights as **untrusted proposals**, never authoritative evidence.

## Invariants (do not relax)

1. No arbitrary LLM-generated Python/shell code execution in the editing path.
2. No gradient-based training or model fine-tuning hidden inside a "weight edit" command.
3. Do not mutate packed 4-bit/8-bit quantized tensors. Keep MLX 4-bit to inference diagnosis until a dedicated validated adapter is implemented.
4. Every edit must be scoped to explicit modules/columns, bounded in scale, integrity checked, reversible, and manually exported to a new checkpoint.
5. Do not promote a patch on shorter answers, lower repetition, or a model's explanation alone. Require unseen labeled target+guard outcomes, independent paired causal tests, and negative controls.
6. Do not compare a tuned candidate against the same examples it was trained/chosen on and call this held-out generalization.
7. Maintain reproducible artifacts; pin exact base revision for real-model claims and never claim original paper scores were reproduced without actually rerunning the relevant evaluation protocol.
8. No networking or large model dependency for `scan`, `demo`, `quick` suite generation, or JSONL parsing; weight/model-heavy imports must be lazy.
9. Error handling must fail closed for malformed patches. A partial write must trigger rollback attempts; no in-service hot patching.
10. Protect proprietary prompts and local model material: never upload/store inputs or weights to remote systems without explicit consent.

## Preferred style

- Small, cohesive functions and minimal mutable state; dependency injection for model and inference adapters.
- Dataclasses for immutable records, functions for calculations, separate modules for I/O and orchestration.
- Explicit checks and meaningful `ValueError` before touching memory.
- Prefer deterministic CPU fixtures over undocumented giant-model samples in CI.
- Keep package install lightweight; keep optional HF dependencies behind extras.
- Update docs/ROADMAP.md when changing the supported model/quantization scope.

## Standard checks

```bash
pytest -q
python -m compileall -q src examples
wspace demo --out /tmp/wsd-demo
wspace scan --traces /tmp/wsd-demo/traces.jsonl --out /tmp/wsd-scan
```

Read [methodology](docs/METHODOLOGY.zh-CN.md), [security](docs/SECURITY.md), and [contributing](CONTRIBUTING.md) before adding architectures or editing modes.
