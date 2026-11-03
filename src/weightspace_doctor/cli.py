"""Stable, scriptable CLI. Heavy dependencies are imported only when needed."""

import argparse
import json
import shutil
import sys
from pathlib import Path

from .backends import OpenAIBackend, generate_cases
from .diagnosis import summarize
from .io import (read_cases, read_pairs, read_traces, write_json,
                 write_jsonl)
from .reporting import render_comparison, save_scan


def layers_arg(value: str | None) -> list[int] | None:
    if value is None or value == "all":
        return None
    try:
        layers = [int(v.strip()) for v in value.split(",")]
    except ValueError as exc:
        raise ValueError("--layers must be comma-separated integers or 'all'") from exc
    if not layers or min(layers) < 0 or len(set(layers)) != len(layers):
        raise ValueError("Invalid or duplicated layer indices")
    return layers


def _add_model(parser):
    parser.add_argument("--model", required=True, help="HF model ID or local BF16/FP16 directory")
    parser.add_argument("--device", default="cpu", help="cpu | auto | cuda | mps")


def _add_decoding(parser):
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)


def make_parser():
    parser = argparse.ArgumentParser(prog="wspace",
                                     description="Weight-Space Model Diagnostics and Repair")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Write synthetic trace examples for offline smoke test")
    demo.add_argument("--out", required=True)

    scan = sub.add_parser("scan", help="Diagnose existing JSONL inference traces (CPU-only)")
    scan.add_argument("--traces", required=True)
    scan.add_argument("--out", required=True)
    scan.add_argument("--model", default="external-or-unspecified")
    scan.add_argument("--loop-threshold", type=float, default=0.38)

    quick = sub.add_parser("quick", help="One-command built-in smoke suite + API diagnosis")
    quick.add_argument("--profile", choices=("general", "math", "code"), default="general")
    quick.add_argument("--base-url", required=True)
    quick.add_argument("--model", required=True)
    quick.add_argument("--out", required=True)
    quick.add_argument("--api-key-env", default="OPENAI_API_KEY")
    quick.add_argument("--timeout", type=int, default=180)
    _add_decoding(quick)

    probe = sub.add_parser("probe", help="Run quick probes through an OpenAI-compatible API")
    probe.add_argument("--cases", required=True)
    probe.add_argument("--base-url", required=True)
    probe.add_argument("--model", required=True)
    probe.add_argument("--out", required=True)
    probe.add_argument("--api-key-env", default="OPENAI_API_KEY")
    probe.add_argument("--timeout", type=int, default=180)
    _add_decoding(probe)

    preflight = sub.add_parser("inspect", help="Inspect local checkpoint config/metadata")
    preflight.add_argument("--model-dir", required=True)
    preflight.add_argument("--out", default=None)

    audit = sub.add_parser("weight-audit", help="Read-only numeric health audit of MLP projections")
    _add_model(audit)
    audit.add_argument("--layers", default="all")
    audit.add_argument("--out", required=True)

    localize = sub.add_parser("localize", help="Contrastive activation and writer localization")
    _add_model(localize)
    localize.add_argument("--pairs", required=True)
    localize.add_argument("--out", required=True)
    localize.add_argument("--layers", default="all")
    localize.add_argument("--topk", type=int, default=64)
    localize.add_argument("--tail-tokens", type=int, default=16)
    localize.add_argument("--no-thinking", action="store_true")

    propose = sub.add_parser("propose", help="Build a reversible floating-weight patch")
    _add_model(propose)
    propose.add_argument("--localization", required=True)
    propose.add_argument("--out", required=True)
    propose.add_argument("--alpha", type=float, default=1.0)
    propose.add_argument("--top-layers", type=int, default=2)
    propose.add_argument("--topk", type=int, default=64)
    propose.add_argument("--layers", default=None)
    propose.add_argument("--max-relative-change", type=float, default=0.18)

    validate = sub.add_parser("validate", help="Paired baseline/edited holdout test + rollback")
    _add_model(validate)
    validate.add_argument("--patch", required=True)
    validate.add_argument("--cases", required=True)
    validate.add_argument("--out", required=True)
    validate.add_argument("--min-target", type=int, default=20)
    validate.add_argument("--min-guard", type=int, default=10)
    _add_decoding(validate)

    repair = sub.add_parser("repair", help="Full bounded, no-gradient Repair Agent")
    _add_model(repair)
    repair.add_argument("--pairs", required=True)
    repair.add_argument("--tune", required=True)
    repair.add_argument("--holdout", required=True)
    repair.add_argument("--causal-holdout", default=None,
                        help="Independent paired continuations; required for review gate")
    repair.add_argument("--out", required=True)
    repair.add_argument("--layers", default="all")
    repair.add_argument("--alphas", default="0.5,1,2")
    repair.add_argument("--topk", type=int, default=64)
    repair.add_argument("--max-relative-change", type=float, default=0.18)
    repair.add_argument("--planner-base-url", default=None)
    repair.add_argument("--planner-model", default=None)
    _add_decoding(repair)
    repair.set_defaults(max_new_tokens=2048)

    causal = sub.add_parser("causal", help="Measure selective continuation suppression and layer isolation")
    _add_model(causal)
    causal.add_argument("--patch", required=True)
    causal.add_argument("--pairs", required=True)
    causal.add_argument("--out", required=True)
    causal.add_argument("--tail-tokens", type=int, default=16)

    export = sub.add_parser("export", help="Explicitly save a new HF checkpoint with a patch")
    _add_model(export)
    export.add_argument("--patch", required=True)
    export.add_argument("--out", required=True)
    return parser



def _api_probe(args: argparse.Namespace, cases) -> int:
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    backend = OpenAIBackend(args.base_url, args.model,
                            api_key_env=args.api_key_env, timeout=args.timeout)
    traces = generate_cases(backend, cases, max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature, seed=args.seed)
    out = Path(args.out)
    write_jsonl(out / "traces.jsonl", [t.to_dict() for t in traces])
    diag = summarize(traces)
    save_scan(out, diag, args.model)
    print(json.dumps({"accuracy": diag.accuracy, "loop_rate": diag.loop_rate,
                      "errors": diag.request_errors, "samples": diag.count,
                      "out": str(out)}))
    return 0 if not diag.request_errors else 1

def _demo(folder: str) -> dict[str, str]:
    from .schema import Trace
    p = Path(folder)
    p.mkdir(parents=True, exist_ok=True)
    traces = [
        Trace(id=f"bad-{i}", prompt="Calculate 12+30", text=("Let me check again. " * 110),
              finish_reason="length", output_tokens=256, max_output_tokens=256,
              expected="42", split="target") for i in range(3)]
    traces += [Trace(id=f"good-{i}", prompt="Calculate 12+30", text="\\boxed{42}",
                     finish_reason="stop", output_tokens=12, max_output_tokens=256,
                     expected="42", split="target") for i in range(3)]
    traces += [Trace(id="guard-1", prompt="Respond OK", text="OK", finish_reason="stop",
                     output_tokens=1, max_output_tokens=256, expected="OK", split="guard")]
    write_jsonl(p / "traces.jsonl", [t.to_dict() for t in traces])
    return {"demo_traces": str(p / "traces.jsonl")}


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        return _dispatch(args)
    except (ValueError, RuntimeError, OSError, KeyError, ImportError) as exc:
        print(f"wspace: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "demo":
        print(json.dumps(_demo(args.out), ensure_ascii=False, indent=2))
        return 0
    if args.command == "scan":
        report = summarize(read_traces(args.traces), threshold=args.loop_threshold)
        save_scan(args.out, report, args.model)
        print(json.dumps({"count": report.count, "loop_rate": report.loop_rate,
                          "cut_rate": report.cut_rate, "accuracy": report.accuracy,
                          "report": str(Path(args.out) / "report.md")}, indent=2))
        return 0
    if args.command in ("probe", "quick"):
        if args.command == "quick":
            from .suites import built_in_cases
            cases = built_in_cases(args.profile)
            write_jsonl(Path(args.out) / "cases.jsonl", [case.__dict__ for case in cases])
        else:
            cases = read_cases(args.cases)
        return _api_probe(args, cases)
    if args.command == "inspect":
        from .preflight import inspect_model_dir
        data = inspect_model_dir(args.model_dir)
        print(json.dumps(data, ensure_ascii=False, indent=2))
        if args.out:
            write_json(args.out, data)
        return 0

    # Load HF only for commands that actually need direct weight access.
    from .hf import HFBackend, load_hf_model
    model, tokenizer = load_hf_model(args.model, device=args.device)
    if args.command == "weight-audit":
        from .weight_audit import audit_weights
        report = audit_weights(model, indices=layers_arg(args.layers))
        write_json(args.out, report)
        print(json.dumps({"out": args.out,
                          "problem_layers": [e["layer"] for e in report["layers"]
                                             if e["status"] != "ok"]}))
        return 0
    if args.command == "localize":
        from .localization import localize_model, save_localization
        pairs = read_pairs(args.pairs)
        result = localize_model(model, tokenizer, pairs, indices=layers_arg(args.layers),
                                topk=args.topk, tail_tokens=args.tail_tokens,
                                thinking=not args.no_thinking)
        save_localization(args.out, result)
        print(json.dumps({"out": args.out, "pairs": len(pairs),
                          "ranked_layers": result["layers"][:5]}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "propose":
        from .editing import build_patch, save_patch
        from .localization import read_localization
        patch = build_patch(model, read_localization(args.localization), model_id=args.model,
                            alpha=args.alpha, top_layers=args.top_layers,
                            topk=args.topk, chosen_layers=layers_arg(args.layers),
                            max_relative_change=args.max_relative_change)
        save_patch(args.out, patch)
        print(json.dumps({"out": args.out, "alpha": patch["alpha"],
                          "layers": [p["layer"] for p in patch["entries"]]}, indent=2))
        return 0
    if args.command == "validate":
        from .editing import load_patch
        from .validation import evaluate_patch
        patch = load_patch(args.patch)
        if patch["model_id"] != args.model:
            raise ValueError("Patch model ID differs from --model: refusing")
        result, base, edited = evaluate_patch(
            HFBackend(model, tokenizer), model, patch, read_cases(args.cases),
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            seed=args.seed, min_target=args.min_target, min_guard=args.min_guard)
        out = Path(args.out)
        write_jsonl(out / "baseline.jsonl", [t.to_dict() for t in base])
        write_jsonl(out / "edited.jsonl", [t.to_dict() for t in edited])
        write_json(out / "metrics.json", result)
        (out / "report.md").write_text(render_comparison(result), encoding="utf-8")
        print(json.dumps({"gate": result["gate"],
                          "target_accuracy_gain": result["target_accuracy_gain"]}, indent=2))
        return 0
    if args.command == "repair":
        from .agent import repair
        result = repair(model, tokenizer, read_pairs(args.pairs), read_cases(args.tune),
                        read_cases(args.holdout), model_id=args.model, out=args.out,
                        alphas=[float(x) for x in args.alphas.split(",")],
                        layers=layers_arg(args.layers), topk=args.topk,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature, seed=args.seed,
                        max_relative_change=args.max_relative_change,
                        planner_url=args.planner_base_url,
                        planner_model=args.planner_model,
                        causal_holdout_pairs=(read_pairs(args.causal_holdout)
                            if args.causal_holdout else None))
        print(json.dumps({"status": result["status"], "out": args.out}, indent=2))
        return 0
    if args.command == "causal":
        from .causal import causal_evidence, isolate_patch
        from .editing import build_random_control, load_patch
        patch = load_patch(args.patch)
        if patch["model_id"] != args.model:
            raise ValueError("Patch checkpoint mismatch")
        pairs = read_pairs(args.pairs)
        measured = {"patch": causal_evidence(model, tokenizer, pairs, patch,
                                            tail_tokens=args.tail_tokens)}
        control = build_random_control(model, patch)
        measured["control"] = causal_evidence(model, tokenizer, pairs, control,
                                               tail_tokens=args.tail_tokens)
        measured["isolated_layers"] = {}
        for entry in patch["entries"]:
            one = isolate_patch(patch, [entry["layer"]])
            measured["isolated_layers"][str(entry["layer"])] = causal_evidence(
                model, tokenizer, pairs, one, tail_tokens=args.tail_tokens)
        write_json(args.out, measured)
        print(json.dumps({"out": args.out,
                          "selectivity": measured["patch"]["selectivity"],
                          "control_selectivity": measured["control"]["selectivity"]}))
        return 0
    if args.command == "export":
        from .editing import load_patch, temporary_patch
        patch = load_patch(args.patch)
        if patch["model_id"] != args.model:
            raise ValueError("Patch checkpoint mismatch")
        out = Path(args.out).resolve()
        if out.exists() and any(out.iterdir()):
            raise ValueError("Export destination is nonempty; only new directories are allowed")
        if Path(args.model).expanduser().resolve() == out:
            raise ValueError("Cannot overwrite source checkpoint")
        out.mkdir(parents=True, exist_ok=True)
        with temporary_patch(model, patch):
            model.save_pretrained(str(out), safe_serialization=True, max_shard_size="2GB")
            tokenizer.save_pretrained(str(out))
        shutil.copy2(Path(args.patch) / "manifest.json", out / "weightspace_patch.json")
        print(json.dumps({"exported": str(out),
                          "notice": "Manual verification required before deployment"}))
        return 0
    raise ValueError("Unknown command")
