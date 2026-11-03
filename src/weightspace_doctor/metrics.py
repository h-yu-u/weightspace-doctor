"""Conservative, explainable heuristics. These are symptoms, not mechanistic proofs."""

import json
import re

from .schema import Finding, Trace


_TOKEN = re.compile(r"[a-zA-Z0-9_]+|[\u3400-\u9fff]|[^\s]", re.UNICODE)
_BOX_START = "\\boxed{"


def last_boxed(text: str) -> str | None:
    """Parse nested braces in the last latex boxed answer without evaluating expressions."""
    start = text.rfind(_BOX_START)
    if start < 0:
        return None
    depth = 1
    result: list[str] = []
    for ch in text[start + len(_BOX_START):]:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(result).strip()
        result.append(ch)
    return None


def normalized_answer(text: str) -> str:
    return " ".join(text.casefold().strip().split())


def correctness(trace: Trace) -> bool | None:
    if trace.error is not None:
        return None
    if trace.expected_regex is not None:
        final = trace.final_text if trace.final_text is not None else (
            trace.text.split("<|/think|>")[-1].split("</think>")[-1])
        try:
            return re.search(trace.expected_regex, final, re.MULTILINE) is not None
        except re.error as exc:
            raise ValueError(f"Invalid expected_regex for case {trace.id}: {exc}") from exc
    if trace.expected is None:
        return None
    final = trace.final_text if trace.final_text is not None else (
        trace.text.split("<|/think|>")[-1].split("</think>")[-1])
    boxed = last_boxed(final)
    candidate = boxed if boxed is not None else final
    return normalized_answer(candidate) == normalized_answer(trace.expected)


def duplicate_ngram_ratio(text: str, *, n: int = 8, tail: int = 256) -> float:
    """Fraction of tail n-grams already seen in the same tail. Not a tokenizer metric."""
    if n < 2:
        raise ValueError("n must be >= 2")
    tokens = _TOKEN.findall(text)[-tail:]
    if len(tokens) < max(48, n * 3):
        return 0.0
    seen: set[tuple[str, ...]] = set()
    duplicates = 0
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i:i + n])
        if gram in seen:
            duplicates += 1
        seen.add(gram)
    return round(duplicates / (len(tokens) - n + 1), 4)


def whitespace_tail(text: str, *, min_len: int = 128) -> bool:
    return len(text) - len(text.rstrip()) >= min_len


def is_cut(trace: Trace) -> bool:
    # Authoritative finish reason if available. Some servers label token exhaustion
    # as "stop"; do not override an explicit stop reason with an approximate count.
    if trace.finish_reason in {"length", "max_tokens", "token_limit"}:
        return True
    if trace.finish_reason in {"stop", "eos", "end_turn", "error"}:
        return False
    return (trace.output_tokens is not None and trace.max_output_tokens is not None
            and trace.output_tokens >= trace.max_output_tokens)


def assess_trace(trace: Trace, *, threshold: float = 0.38) -> Finding:
    if trace.error is not None:
        return Finding(trace.id, trace.split, ("request_error",), 0.0, None, False,
                       (trace.error[:300],))
    ratio = duplicate_ngram_ratio(trace.text)
    flags: list[str] = []
    evidence: list[str] = []
    if not trace.text.strip():
        flags.append("empty_output")
        evidence.append("Response body is empty; investigate EOS, endpoint and generation config")
    if "json" in trace.tags and trace.text.strip():
        final = trace.final_text if trace.final_text is not None else (
        trace.text.split("<|/think|>")[-1].split("</think>")[-1])
        try:
            json.loads(final)
        except ValueError:
            flags.append("invalid_json")
            evidence.append("JSON-tagged response is not syntactically valid JSON")
    if "benign" in trace.tags and re.search(
            r"\b(?:I cannot|I can't|I am unable|sorry,? I cannot)\b|无法(?:提供|回答|帮助)",
            trace.text, flags=re.IGNORECASE):
        flags.append("possible_refusal")
        evidence.append("Refusal-like text in a case explicitly tagged benign")
    if trace.finish_reason == "content_filter":
        flags.append("provider_content_filter")
        evidence.append("Request ended under provider content filtering")
    if whitespace_tail(trace.text):
        flags.append("whitespace_tail")
        evidence.append("Output ends with >127 consecutive whitespace chars")
    if ratio >= threshold:
        flags.append("repetition_loop")
        evidence.append(f"Last-256 lexical-token repeated 8-gram ratio={ratio:.3f}")
    if is_cut(trace):
        flags.append("budget_exhaustion")
        evidence.append("Generation reached its declared token limit")
    correct = correctness(trace)
    if correct is False:
        flags.append("incorrect_or_format_failure")
    completed = not is_cut(trace) and not whitespace_tail(trace.text)
    return Finding(trace.id, trace.split, tuple(flags), ratio, correct,
                   completed, tuple(evidence))
