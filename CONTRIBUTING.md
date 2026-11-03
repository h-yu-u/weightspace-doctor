# Contributing

Contributions should improve diagnostic accuracy, reproducibility and safety, rather than merely increase the number of editable architectures.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[hf,dev]'
pytest -q
python -m compileall -q src
```

## Design rules

- Prefer pure transformations in `metrics.py`, `diagnosis.py`, `schema.py`, and `validation.py`; isolate side effects in `backends.py`, `hf.py`, `editing.py`, and `io.py`.
- Keep inference, activation capture, planning, weight editing, evaluation, and persistence in separate modules. Do not combine them in one monolithic agent class.
- Never enable remote LLM-generated code execution or bypass patch preconditions.
- A new edit strategy must include a reversible operation, dtype/architecture checks, a norm budget, an independent negative control, and unit tests showing exact rollback.
- Differentiate findings from hypotheses, causal intervention evidence, and model-quality improvements. No unverified model-score claims in PR descriptions.
- Synthetic examples are useful for unit testing but are not empirical model benchmarks.
- Preserve source model immutability. Never commit model weights, proprietary prompts, API keys, raw customer generations, or private benchmark data.
- Add at least one regression test for a bug fix. Update CLI/help text and README when public behavior changes.

## Suggested PR structure

1. Summary and motivation.
2. Source model, architecture and environment if model-specific.
3. Mechanism or symptom being examined.
4. Test commands and results; distinguish synthetic tests and actual-model benchmarks.
5. Risk analysis, negative control, guard evaluations, and rollback implications.
6. Documentation updates and any compatibility break.

## Reporting security issues

Do not include private model weights or user data in public issues. See [Security](docs/SECURITY.md) for threat boundaries.
