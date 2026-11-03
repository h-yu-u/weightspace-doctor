"""Paired, split-aware evaluation. No training or gradients."""

from typing import Any
import math

from .backends import TextGenerator, generate_cases
from .diagnosis import summarize
from .editing import temporary_patch
from .schema import Case, Trace


def _by_split(traces: list[Trace], split: str) -> list[Trace]:
    return [t for t in traces if t.split == split]


def paired_sign_test_pvalue(rescues: int, regressions: int) -> float:
    """Exact one-sided binomial sign-test for paired discordant outcomes.

    H0: a change is equally likely to rescue or regress on discordant pairs.
    Agreement cases are excluded, and zero discordances provide no evidence.
    """
    if rescues < 0 or regressions < 0:
        raise ValueError("Discordance counts must be non-negative")
    n = rescues + regressions
    if n == 0:
        return 1.0
    logs = [math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            - n * math.log(2) for k in range(rescues, n + 1)]
    pivot = max(logs)
    return min(1.0, math.exp(pivot) * math.fsum(math.exp(t - pivot) for t in logs))


def compare_traces(baseline: list[Trace], edited: list[Trace], *,
                   min_target: int = 20, min_guard: int = 10,
                   max_guard_drop: float = 0.03) -> dict[str, Any]:
    b = {t.id: t for t in baseline}
    e = {t.id: t for t in edited}
    if not b or set(b) != set(e):
        raise ValueError("Paired evaluation requires exactly matching case IDs")
    if any(b[key].split != e[key].split for key in b):
        raise ValueError("Case splits changed across paired runs")
    data: dict[str, Any] = {}
    for split in ("target", "guard"):
        base_items = _by_split(baseline, split)
        edit_items = _by_split(edited, split)
        data[split] = {"n": len(base_items),
                       "base": summarize(base_items).to_dict() if base_items else _empty(),
                       "edited": summarize(edit_items).to_dict() if edit_items else _empty()}
        rescues = regressions = 0
        for orig in base_items:
            new = e[orig.id]
            a = _truth(orig)
            after = _truth(new)
            if a is False and after is True:
                rescues += 1
            if a is True and after is False:
                regressions += 1
        data[split]["rescues"] = rescues
        data[split]["regressions"] = regressions
        data[split]["paired_sign_test_pvalue"] = paired_sign_test_pvalue(
            rescues, regressions)

    target, guard = data["target"], data["guard"]
    reasons: list[str] = []
    if target["n"] < min_target or guard["n"] < min_guard:
        reasons.append("insufficient_validation_cases")
    if target["base"]["accuracy"] is None or guard["base"]["accuracy"] is None:
        reasons.append("missing_ground_truth_labels")
    if sum(data[k][m]["request_errors"] for k in ("target", "guard")
           for m in ("base", "edited")):
        reasons.append("inference_request_errors")
    delta = _accuracy_gain(target)
    if delta is None or delta <= 0:
        reasons.append("target_accuracy_not_improved")
    elif target["paired_sign_test_pvalue"] >= 0.05:
        reasons.append("target_improvement_not_statistically_supported")
    guard_delta = _accuracy_gain(guard)
    if guard_delta is None or guard_delta < -max_guard_drop:
        reasons.append("guard_accuracy_regressed")
    if (target["edited"]["cut_rate"] > target["base"]["cut_rate"] + .05 or
            guard["edited"]["cut_rate"] > guard["base"]["cut_rate"] + .05):
        reasons.append("token_limit_regressed")
    if (target["edited"]["loop_rate"] > target["base"]["loop_rate"] + .05 or
            guard["edited"]["loop_rate"] > guard["base"]["loop_rate"] + .05):
        reasons.append("repetition_regressed")
    return {**data, "gate": {"status": "recommend_manual_review" if not reasons
                             else "not_approved", "reasons": reasons or ["all_measured_gates_passed"],
                             "min_target": min_target, "min_guard": min_guard},
            "target_accuracy_gain": delta, "guard_accuracy_gain": guard_delta}


def _accuracy_gain(part: dict[str, Any]) -> float | None:
    a, b = part["base"]["accuracy"], part["edited"]["accuracy"]
    return (b - a) if a is not None and b is not None else None


def _truth(item: Trace) -> bool | None:
    from .metrics import correctness
    return correctness(item)


def _empty() -> dict[str, Any]:
    return {"count": 0, "accuracy": None, "loop_rate": 0.0,
            "cut_rate": 0.0, "request_errors": 0}


def evaluate_patch(generator: TextGenerator, model: Any, patch: dict[str, Any],
                   cases: list[Case], *, baseline: list[Trace] | None = None,
                   max_new_tokens: int = 256, temperature: float = 0.0,
                   seed: int = 0, min_target: int = 20,
                   min_guard: int = 10) -> tuple[dict[str, Any], list[Trace], list[Trace]]:
    ids = {c.id for c in cases}
    if ids.intersection(patch.get("discovery_ids", [])):
        raise ValueError("Leakage: validation case ID was used to discover the direction")
    if baseline is None:
        baseline = generate_cases(generator, cases, max_new_tokens=max_new_tokens,
                                  temperature=temperature, seed=seed)
    with temporary_patch(model, patch):
        edited = generate_cases(generator, cases, max_new_tokens=max_new_tokens,
                                temperature=temperature, seed=seed)
    return compare_traces(baseline, edited, min_target=min_target,
                          min_guard=min_guard), baseline, edited
