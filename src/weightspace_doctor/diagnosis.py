"""Pure summarization and next-action triage."""

from collections import Counter
from statistics import median

from .metrics import assess_trace
from .schema import Assessment, Trace


def summarize(traces: list[Trace], *, threshold: float = 0.38) -> Assessment:
    if not traces:
        raise ValueError("At least one trace is required")
    findings = [assess_trace(t, threshold=threshold) for t in traces]
    anomaly_counts = dict(sorted(Counter(flag for f in findings for flag in f.flags).items()))
    tag_groups: dict[str, list[int]] = {}
    for i, trace in enumerate(traces):
        for tag in trace.tags:
            tag_groups.setdefault(tag, []).append(i)
    tags = {}
    for tag, idxs in tag_groups.items():
        subset = [findings[i] for i in idxs]
        labeled = sum(f.correctness is not None for f in subset)
        tags[tag] = {"n": len(subset), "labeled": labeled,
                     "accuracy": (sum(f.correctness is True for f in subset) / labeled
                                  if labeled else None),
                     "loop_rate": sum("repetition_loop" in f.flags for f in subset) / len(subset),
                     "cut_rate": sum("budget_exhaustion" in f.flags for f in subset) / len(subset)}
    speeds = [t.output_tokens * 1000 / t.elapsed_ms for t in traces
              if t.elapsed_ms and t.elapsed_ms > 0 and t.output_tokens is not None]
    times = sorted(t.elapsed_ms for t in traces if t.elapsed_ms is not None)
    return Assessment(
        count=len(traces),
        labeled=sum(f.correctness is not None for f in findings),
        correct=sum(f.correctness is True for f in findings),
        loop_count=sum("repetition_loop" in f.flags for f in findings),
        budget_exhausted=sum("budget_exhaustion" in f.flags for f in findings),
        request_errors=sum("request_error" in f.flags for f in findings),
        incomplete=sum(not f.completed for f in findings),
        findings=[f.to_dict() for f in findings],
        anomaly_counts=anomaly_counts, by_tag=tags,
        median_tokens_per_second=round(median(speeds), 3) if speeds else None,
        p50_elapsed_ms=round(median(times)) if times else None,
        p95_elapsed_ms=(times[min(len(times)-1, int(.95*(len(times)-1)))]) if times else None,
    )


def triage(a: Assessment) -> list[dict[str, str]]:
    actions: list[dict[str, str]] = []
    if a.request_errors:
        actions.append({"priority": "P0", "issue": "inference_transport_error",
                        "action": "Check API endpoint, model name and timeout before inspecting weights"})
    if a.anomaly_counts.get("empty_output", 0):
        actions.append({"priority": "P1", "issue": "empty_generation",
                        "action": "Inspect EOS/chat template, stop strings, model server errors; "
                                  "do not attempt weight editing until the pipeline is correct"})
    if a.anomaly_counts.get("invalid_json", 0):
        actions.append({"priority": "P2", "issue": "json_output_invalid",
                        "action": "Check format instructions, constrained decoding and validators "
                                  "before considering weight changes"})
    if a.anomaly_counts.get("possible_refusal", 0):
        actions.append({"priority": "P2", "issue": "potential_benign_refusal",
                        "action": "Review examples manually; refusal heuristic is language-dependent "
                                  "and not a definitive policy or safety diagnosis"})
    if a.loop_rate >= 0.2 and a.count >= 5:
        actions.append({"priority": "P1", "issue": "possible_reasoning_loop",
                        "action": "Check sampling/chat template first; collect matched good/bad "
                                  "continuations, then run localize and controlled sparse edits"})
    if a.cut_rate >= 0.2:
        actions.append({"priority": "P1", "issue": "budget_exhaustion",
                        "action": "Investigate max_tokens, EOS and truncation; do not conflate "
                                  "budget exhaustion with lack of reasoning ability"})
    if a.labeled >= 5 and a.accuracy is not None and a.accuracy < 0.5:
        actions.append({"priority": "P2", "issue": "task_quality_low",
                        "action": "Collect errors by domain; knowledge gaps need data/RAG/SFT, "
                                  "not assumed weight-space edits"})
    if not actions:
        actions.append({"priority": "P3", "issue": "no_strong_signal",
                        "action": "Increase representative cases; these heuristics cannot prove safety "
                                  "or absence of hidden model failures"})
    return actions
