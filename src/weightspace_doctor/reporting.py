"""Markdown reports are derived exclusively from measurements."""

from pathlib import Path
from typing import Any

from .diagnosis import triage
from .io import write_json
from .schema import Assessment


def render_scan(assessment: Assessment, model: str) -> str:
    rate = assessment.accuracy
    acc = "not measured" if rate is None else f"{rate:.1%} ({assessment.labeled} labeled)"
    lines = ["# WeightSpace Doctor · Diagnostic Report", "",
             f"- Model: `{model}`", f"- Samples: {assessment.count}",
             f"- Task accuracy: {acc}",
             f"- Repetition-loop heuristic: {assessment.loop_count}/{assessment.count} "
             f"({assessment.loop_rate:.1%})",
             f"- Token-limit hits: {assessment.budget_exhausted}/{assessment.count} "
             f"({assessment.cut_rate:.1%})",
             f"- API errors: {assessment.request_errors}",
             f"- Median output tokens / total request time: "
             f"{assessment.median_tokens_per_second if assessment.median_tokens_per_second is not None else 'n/a'} tokens/s",
             f"- p50/p95 latency: {assessment.p50_elapsed_ms}/{assessment.p95_elapsed_ms} ms",
             "", "## Symptom breakdown", ""]
    for key, count in assessment.anomaly_counts.items():
        lines.append(f"- `{key}`: {count}")
    lines.extend(["", "## Domain/tag breakdown", ""])
    for tag, group in assessment.by_tag.items():
        acc = "unlabeled" if group["accuracy"] is None else f"{group['accuracy']:.1%}"
        lines.append(f"- `{tag}`: n={group['n']}, accuracy={acc}, "
                     f"loops={group['loop_rate']:.1%}, cut={group['cut_rate']:.1%}")
    lines.extend(["", "## Recommended next investigations", ""])
    for act in triage(assessment):
        lines.append(f"- **{act['priority']} · {act['issue']}**: {act['action']}")
    lines.extend(["", "## Evidence per case", "",
                  "| Case | Split | Symptoms | Repetition ratio | Correct |",
                  "| --- | --- | --- | ---: | --- |"])
    for f in assessment.findings:
        flags = ", ".join(f["flags"]) or "-"
        valid = "?" if f["correctness"] is None else str(f["correctness"])
        lines.append(f"| {f['id']} | {f['split']} | {flags} | "
                     f"{f['repetition_ratio']:.2f} | {valid} |")
    lines.extend(["", "## Limitations", "",
                  "Repeated 8-grams are lexical heuristics, not model-token statistics. "
                  "Task correctness is only evaluated with explicit expected/regex labels. "
                  "Fast triage cannot identify causal weight directions without intervention. "
                  "Do not promote a patch solely because outputs became shorter."])
    return "\n".join(lines) + "\n"


def save_scan(folder: str | Path, assessment: Assessment, model: str) -> None:
    p = Path(folder)
    p.mkdir(parents=True, exist_ok=True)
    write_json(p / "diagnosis.json", {**assessment.to_dict(), "model": model,
                                     "triage": triage(assessment)})
    (p / "report.md").write_text(render_scan(assessment, model), encoding="utf-8")


def render_comparison(values: dict[str, Any]) -> str:
    lines = ["# Paired Evaluation", "",
             "| Split | Base accuracy | Edited accuracy | Base cut | Edited cut |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for split in ("target", "guard"):
        base = values[split]["base"]
        after = values[split]["edited"]
        fmt = lambda x: "n/a" if x is None else f"{x:.1%}"  # noqa: E731
        lines.append(f"| {split} | {fmt(base['accuracy'])} | {fmt(after['accuracy'])} | "
                     f"{fmt(base['cut_rate'])} | {fmt(after['cut_rate'])} |")
    lines.extend(["", f"**Decision**: `{values['gate']['status']}`",
                  f"**Reasons**: {', '.join(values['gate']['reasons'])}", "",
                  "Repeated trials and cross-domain checks are required before production."])
    return "\n".join(lines) + "\n"
