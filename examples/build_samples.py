"""Generate synthetic, illustrative JSONL fixtures. NOT real-model evidence."""

import json
from pathlib import Path


OUT = Path(__file__).parent


def write(name, rows):
    with (OUT / name).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    traces = [
        {"id": "trace-loop", "prompt": "Compute 11+31", "text": "Wait, reconsider the steps. " * 90,
         "finish_reason": "length", "output_tokens": 256, "max_output_tokens": 256,
         "elapsed_ms": 2500, "expected": "42", "split": "target", "tags": ["math"]},
        {"id": "trace-answer", "prompt": "Compute 11+31", "text": "\\boxed{42}",
         "finish_reason": "stop", "output_tokens": 8, "max_output_tokens": 256,
         "elapsed_ms": 200, "expected": "42", "split": "target", "tags": ["math"]},
        {"id": "trace-json", "prompt": "Return JSON", "text": "{valid: no}",
         "finish_reason": "stop", "output_tokens": 9, "max_output_tokens": 256,
         "elapsed_ms": 180, "expected_regex": "^\\{", "split": "guard", "tags": ["json"]},
        {"id": "trace-guard", "prompt": "Reply OK", "text": "OK",
         "finish_reason": "stop", "output_tokens": 1, "max_output_tokens": 256,
         "elapsed_ms": 110, "expected": "OK", "split": "guard", "tags": ["instruction"]},
        {"id": "trace-api", "prompt": "test API", "text": "", "finish_reason": "error",
         "error": "HTTP 503 synthetic", "split": "target", "tags": ["infrastructure"]},
    ]
    write("traces.sample.jsonl", traces)

    cases = [
        {"id": "sm-math-1", "prompt": "What is 12 + 30? Respond with digits only.",
         "expected": "42", "split": "target", "tags": ["math", "short"]},
        {"id": "sm-math-2", "prompt": "What is 9 * 9? Respond with digits only.",
         "expected": "81", "split": "target", "tags": ["math", "short"]},
        {"id": "sm-math-3", "prompt": "What is 100 - 23? Respond with digits only.",
         "expected": "77", "split": "target", "tags": ["math", "short"]},
        {"id": "sm-math-4", "prompt": "What is 144 / 12? Respond with digits only.",
         "expected": "12", "split": "target", "tags": ["math", "short"]},
        {"id": "sm-format-json", "prompt": "Return exactly valid JSON with key result and integer value 3.",
         "expected_regex": "^\\s*\\{[^}]+\\}\\s*$", "split": "guard", "tags": ["json"]},
        {"id": "sm-guard-1", "prompt": "Reply with exactly one word: OK", "expected": "OK",
         "split": "guard", "tags": ["instruction"]},
    ]
    write("cases.sample.jsonl", cases)

    questions = [
        ("Compute 14+28.", "14 plus 28 is 42. Final answer: 42."),
        ("Compute 9*8.", "9 times 8 is 72. Final answer: 72."),
        ("Compute 100-37.", "100 minus 37 is 63. Final answer: 63."),
        ("Compute 81/9.", "81 divided by 9 is 9. Final answer: 9."),
        ("Compute 15+26.", "15 plus 26 is 41. Final answer: 41."),
        ("Compute 11*11.", "11 times 11 is 121. Final answer: 121."),
    ]
    pairs = []
    for i, (prompt, correct) in enumerate(questions):
        loop = "But maybe I should check again. " * 14
        pairs.append({"id": f"discovery-{i+1:02d}", "prompt": prompt,
                      "good": correct, "bad": f"Let me think. {loop}"})
    write("contrasts.sample.jsonl", pairs)

    holdout_pairs = [
        ("Compute 17+29.", "17 plus 29 equals 46."),
        ("Compute 7*13.", "7 times 13 equals 91."),
        ("Compute 88-19.", "88 minus 19 equals 69."),
        ("Compute 96/8.", "96 divided by 8 equals 12."),
        ("Compute 45+37.", "45 plus 37 equals 82."),
        ("Compute 16*4.", "16 times 4 equals 64."),
    ]
    write("causal_holdout.sample.jsonl", [
        {"id": f"causal-{i+1:02d}", "prompt": prompt,
         "good": correct, "bad": "Wait, maybe redo the entire calculation. " * 14}
        for i, (prompt, correct) in enumerate(holdout_pairs)])

    # These are deliberately simplistic examples for the IO contract and CLI, not
    # real task-quality evidence.  They should NOT be used to approve patches.
    for kind, offset, n in (("tune", 200, 24), ("holdout", 400, 30)):
        rows = []
        for i in range(n):
            a, b = offset + i, 5 + (i % 31)
            rows.append({"id": f"{kind}-target-{i+1:03d}",
                         "prompt": f"Compute {a}+{b}. Return only the integer.",
                         "expected": str(a + b), "split": "target", "tags": ["math"]})
        for i in range(12):
            rows.append({"id": f"{kind}-guard-{i+1:03d}",
                         "prompt": f"Write the uppercase form of the ASCII word 'sun' (example {i+1}). "
                                   "Return exactly one word.",
                         "expected": "SUN", "split": "guard", "tags": ["instruction"]})
        write(f"{kind}.sample.jsonl", rows)


if __name__ == "__main__":
    main()
