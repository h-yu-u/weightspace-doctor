"""Bounded inspect→localize→edit→test→abstain workflow (never auto-deploy)."""

from pathlib import Path
from typing import Any

from .backends import generate_cases
from .causal import causal_evidence, isolate_patch
from .diagnosis import summarize, triage
from .editing import build_patch, build_random_control, save_patch
from .io import write_json, write_jsonl
from .localization import localize_model, save_localization
from .planner import llm_plan, recommend_layers
from .reporting import render_comparison, save_scan
from .schema import Case, ContrastPair
from .validation import evaluate_patch


def _candidate_score(metrics: dict[str, Any]) -> float:
    target, guard = metrics["target"], metrics["guard"]
    if target["base"]["accuracy"] is None:
        return -1e6  # No labeled quality -> do not optimize token counts blindly.
    gain = metrics["target_accuracy_gain"] or 0.0
    cut_improve = target["base"]["cut_rate"] - target["edited"]["cut_rate"]
    loop_improve = target["base"]["loop_rate"] - target["edited"]["loop_rate"]
    guard_drop = max(0.0, -(metrics["guard_accuracy_gain"] or 0.0))
    return 5.0 * gain + cut_improve + 0.5 * loop_improve - 7.0 * guard_drop


def _check_ids(pairs: list[ContrastPair], tune: list[Case], holdout: list[Case]):
    used = set(p.id for p in pairs)
    tuning = {c.id for c in tune}
    final = {c.id for c in holdout}
    if used & tuning or used & final or tuning & final:
        raise ValueError("Discovery/tuning/holdout case IDs must be disjoint")
    if not tune or not holdout:
        raise ValueError("Both tuning and unseen holdout cases are required")
    if not any(c.split == "target" and c.expected is not None for c in tune):
        raise ValueError("Tuning needs labeled target cases; blind length minimization is unsafe")


def repair(model: Any, tokenizer: Any, pairs: list[ContrastPair],
           tuning_cases: list[Case], holdout_cases: list[Case], *,
           model_id: str, out: str | Path, alphas: list[float] | None = None,
           max_new_tokens: int = 512, temperature: float = 0.0,
           seed: int = 2026, topk: int = 64, layers: list[int] | None = None,
           max_relative_change: float = 0.18,
           planner_url: str | None = None,
           planner_model: str | None = None,
           causal_holdout_pairs: list[ContrastPair] | None = None) -> dict[str, Any]:
    from .hf import HFBackend
    from .io import unique_ids

    unique_ids(pairs)
    unique_ids(tuning_cases)
    unique_ids(holdout_cases)
    _check_ids(pairs, tuning_cases, holdout_cases)
    if causal_holdout_pairs:
        unique_ids(causal_holdout_pairs)
        used = set(p.id for p in pairs) | {c.id for c in tuning_cases + holdout_cases}
        if used & {p.id for p in causal_holdout_pairs}:
            raise ValueError("Causal holdout must be disjoint from all other examples")
    alphas = alphas or [0.5, 1.0, 2.0]
    if len(alphas) > 8:
        raise ValueError("Maximum 8 candidate alphas per run")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    backend = HFBackend(model, tokenizer, thinking=True)
    trace = {"model": model_id, "states": ["triaging"], "status": "running",
             "decoding": {"max_new_tokens": max_new_tokens, "temperature": temperature,
                          "seed": seed},
             "validation_split": {"tune": len(tuning_cases), "holdout": len(holdout_cases),
                                  "contrasts": len(pairs),
                                  "independent_contrasts": len(causal_holdout_pairs or [])}}

    baseline_tune = generate_cases(backend, tuning_cases, max_new_tokens=max_new_tokens,
                                   temperature=temperature, seed=seed)
    write_jsonl(out / "tuning_baseline.jsonl", [t.to_dict() for t in baseline_tune])
    diag = summarize(baseline_tune)
    save_scan(out / "baseline_diagnosis", diag, model_id)
    trace["baseline"] = diag.to_dict()
    trace["triage"] = triage(diag)
    if diag.request_errors:
        trace.update(status="abstained_inference_errors", states=trace["states"] + ["abstained"])
        write_json(out / "run.json", trace)
        return trace
    if diag.loop_rate < .15:
        absence = ("abstained_token_limit_without_repetition"
                   if diag.cut_rate >= .15 else "abstained_no_loop_signal")
        trace.update(status=absence, states=trace["states"] + ["abstained"])
        write_json(out / "run.json", trace)
        return trace
    if len(pairs) < 4:
        trace.update(status="abstained_insufficient_contrasts",
                     states=trace["states"] + ["abstained"])
        write_json(out / "run.json", trace)
        return trace

    trace["states"].append("localizing")
    localized = localize_model(model, tokenizer, pairs, indices=layers, topk=topk)
    save_localization(out / "localization", localized)
    selected_layers = recommend_layers(localized)
    llm_response = None
    if planner_url and planner_model:
        # Untrusted LLM JSON is checked against discovered layers and the alpha whitelist.
        llm_response = llm_plan(localized, diag.to_dict(), base_url=planner_url,
                                model=planner_model)
        selected_layers = llm_response["layers"]
        alphas = list(dict.fromkeys([llm_response["alpha"], *alphas]))[:8]
    trace["planner"] = llm_response or {"origin": "deterministic_ranker",
                                         "layers": selected_layers}
    if not selected_layers:
        trace.update(status="abstained_no_editable_layers",
                     states=trace["states"] + ["abstained"])
        write_json(out / "run.json", trace)
        return trace

    trace["states"].append("tuning")
    best: tuple[float, dict[str, Any], dict[str, Any]] | None = None
    trials: list[dict[str, Any]] = []
    for alpha in alphas:
        try:
            patch = build_patch(model, localized, model_id=model_id, alpha=alpha,
                                chosen_layers=selected_layers, topk=topk,
                                max_relative_change=max_relative_change)
            result, _, edited = evaluate_patch(
                backend, model, patch, tuning_cases, baseline=baseline_tune,
                max_new_tokens=max_new_tokens, temperature=temperature, seed=seed)
            score = _candidate_score(result)
            trials.append({"alpha": alpha, "score": score,
                           "target_accuracy_gain": result["target_accuracy_gain"],
                           "target_cut_rate": result["target"]["edited"]["cut_rate"],
                           "guard_accuracy_gain": result["guard_accuracy_gain"]})
            write_jsonl(out / f"tuning_alpha_{alpha:g}.jsonl", [t.to_dict() for t in edited])
            if best is None or score > best[0]:
                best = score, patch, result
        except (ValueError, RuntimeError) as exc:
            trials.append({"alpha": alpha, "error": str(exc)})
    trace["trials"] = trials
    if best is None or best[0] <= 0:
        trace.update(status="abstained_no_tuning_improvement",
                     states=trace["states"] + ["abstained"])
        write_json(out / "run.json", trace)
        return trace

    trace["states"].append("holdout_validation")
    _, patch, tune_metrics = best
    save_patch(out / "candidate_patch", patch)
    write_json(out / "tuning_metrics.json", tune_metrics)
    metrics, baseline_hold, edited_hold = evaluate_patch(
        backend, model, patch, holdout_cases, max_new_tokens=max_new_tokens,
        temperature=temperature, seed=seed)
    write_jsonl(out / "holdout_baseline.jsonl", [t.to_dict() for t in baseline_hold])
    write_jsonl(out / "holdout_edited.jsonl", [t.to_dict() for t in edited_hold])
    write_json(out / "holdout_metrics.json", metrics)

    # Independent norm-matched orthogonal control: does a random perturbation
    # of the same size improve equally? It is a useful negative control,
    # not a substitute for diverse held-out benchmarks.
    control = build_random_control(model, patch, seed=seed)
    save_patch(out / "negative_control_patch", control)
    control_metrics, _, _ = evaluate_patch(
        backend, model, control, holdout_cases, baseline=baseline_hold,
        max_new_tokens=max_new_tokens, temperature=temperature, seed=seed)
    write_json(out / "negative_control_metrics.json", control_metrics)
    causal = None
    if causal_holdout_pairs and len(causal_holdout_pairs) >= 4:
        trace["states"].append("independent_causal_validation")
        causal = {
            "patch": causal_evidence(model, tokenizer, causal_holdout_pairs, patch),
            "random_control": causal_evidence(model, tokenizer, causal_holdout_pairs, control),
            "isolated_layers": {},
        }
        for entry in patch["entries"]:
            selected = isolate_patch(patch, [entry["layer"]])
            causal["isolated_layers"][str(entry["layer"])] = causal_evidence(
                model, tokenizer, causal_holdout_pairs, selected)
        write_json(out / "independent_causal.json", causal)
    if (metrics["target_accuracy_gain"] is not None and
            control_metrics["target_accuracy_gain"] is not None and
            metrics["target_accuracy_gain"] <= control_metrics["target_accuracy_gain"]):
        metrics["gate"]["reasons"].append("not_better_than_norm_matched_random_control")
        metrics["gate"]["status"] = "not_approved"
    if causal is None:
        metrics["gate"]["reasons"].append("independent_causal_contrast_holdout_required")
        metrics["gate"]["status"] = "not_approved"
    elif (not causal["patch"]["directionally_supported"] or
          causal["patch"]["selectivity"] <= causal["random_control"]["selectivity"]):
        metrics["gate"]["reasons"].append("selective_causal_evidence_not_established")
        metrics["gate"]["status"] = "not_approved"
    if patch["pairs"] < 4:
        metrics["gate"]["reasons"].append("insufficient_contrast_pairs")
        metrics["gate"]["status"] = "not_approved"
    write_json(out / "holdout_metrics.json", metrics)

    trace.update(status=metrics["gate"]["status"],
                 states=trace["states"] + ["finished"],
                 winner={"alpha": patch["alpha"],
                         "layers": [e["layer"] for e in patch["entries"]]},
                 gate=metrics["gate"],
                 note="Candidate only. No patch was deployed or merged into checkpoint.")
    write_json(out / "run.json", trace)
    (out / "report.md").write_text(
        "# Model Repair Agent · Final Report\n\n"
        + f"Model: `{model_id}`  \nMethod: sparse projector, no gradients  \n"
        + f"Candidate layers: {trace['winner']['layers']}; alpha={patch['alpha']}\n\n"
        + render_comparison(metrics) + "\n"
        + (f"Independent causal selectivity: {causal['patch']['selectivity']:.4f}; "
           f"random control: {causal['random_control']['selectivity']:.4f}\n\n"
           if causal is not None else "Independent causal contrasts: NOT PROVIDED\n\n")
        + f"Negative-control target accuracy gain: "
        f"{control_metrics['target_accuracy_gain']}\n\n"
        + "**No automatic deployment.** Inspect samples, rerun with multiple seeds, "
          "verify domain-specific guardrails, then explicitly export.\n",
        encoding="utf-8")
    return trace
