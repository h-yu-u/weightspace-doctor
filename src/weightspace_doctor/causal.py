"""Teacher-forced continuation likelihood under reversible causal interventions.

This is an intervention test, not proof of a unique circuit or a learned ability.
No gradients, training loops, or learned probe classifiers are used.
"""

from statistics import mean
from typing import Any

from .editing import temporary_patch
from .hf import format_prefix, require_torch
from .schema import ContrastPair


def score_continuation(model: Any, tokenizer: Any, prompt_prefix: str,
                       continuation: str, *, tail_tokens: int = 16,
                       max_context_tokens: int = 2048) -> float:
    """Mean log P of the final N continuation tokens using teacher forcing.

    Prefix and continuation are tokenized separately; this explicitly fixes
    the token boundary for comparable scoring in base/edited models.
    """
    if not 1 <= tail_tokens <= 128 or max_context_tokens < tail_tokens + 1:
        raise ValueError("Invalid tail_tokens or max_context_tokens")
    torch = require_torch()
    p = tokenizer(prompt_prefix, add_special_tokens=False)["input_ids"]
    s = tokenizer(continuation, add_special_tokens=False)["input_ids"]
    if torch.is_tensor(p):
        p = p.squeeze().tolist()
    if torch.is_tensor(s):
        s = s.squeeze().tolist()
    if p and isinstance(p[0], list):
        p = p[0]
    if s and isinstance(s[0], list):
        s = s[0]
    if not p or not s:
        raise ValueError("Prefix and continuation must each have at least one token")
    if len(p) + len(s) > max_context_tokens:
        raise ValueError("Context too long for bounded teacher-forced probe")
    tail = min(tail_tokens, len(s))
    device = model.get_input_embeddings().weight.device
    ids = torch.tensor([p + s], device=device, dtype=torch.long)
    with torch.inference_mode():
        try:
            out = model(input_ids=ids, use_cache=False, logits_to_keep=tail + 1)
            logits = out.logits
            if logits.shape[1] > tail + 1:
                logits = logits[:, -(tail + 1):, :]
        except TypeError as exc:
            if "logits_to_keep" not in str(exc):
                raise
            out = model(input_ids=ids, use_cache=False)
            logits = out.logits[:, -(tail + 1):, :]
        # The final input token cannot predict itself; the logits just before
        # each target token correspond to positions -tail-1 .. -2.
        if logits.shape[1] < tail + 1:
            raise ValueError("Model returned insufficient logits for continuation scoring")
        target = ids[0, -tail:]
        predictions = logits[0, -(tail + 1):-1, :].float()
        logprobs = torch.log_softmax(predictions, dim=-1)
        selected = logprobs.gather(1, target[:, None]).squeeze(-1)
        return float(selected.mean().item())


def score_pairs(model: Any, tokenizer: Any, pairs: list[ContrastPair], *,
                thinking: bool = True, tail_tokens: int = 16) -> list[dict[str, Any]]:
    result = []
    for pair in pairs:
        prefix = format_prefix(tokenizer, pair.prompt, thinking=thinking)
        good = score_continuation(model, tokenizer, prefix, pair.good,
                                  tail_tokens=tail_tokens)
        bad = score_continuation(model, tokenizer, prefix, pair.bad,
                                 tail_tokens=tail_tokens)
        result.append({"id": pair.id, "good_logp": good, "bad_logp": bad})
    return result


def causal_evidence(model: Any, tokenizer: Any, pairs: list[ContrastPair],
                    patch: dict[str, Any], *, tail_tokens: int = 16,
                    thinking: bool = True) -> dict[str, Any]:
    if not pairs:
        raise ValueError("Requires matched productive/repetitive contrasts")
    base = score_pairs(model, tokenizer, pairs, tail_tokens=tail_tokens, thinking=thinking)
    with temporary_patch(model, patch):
        after = score_pairs(model, tokenizer, pairs, tail_tokens=tail_tokens,
                            thinking=thinking)
    measured = []
    for b, e in zip(base, after):
        measured.append({"id": b["id"],
                         "good_delta": e["good_logp"] - b["good_logp"],
                         "bad_delta": e["bad_logp"] - b["bad_logp"]})
    good = mean(x["good_delta"] for x in measured)
    bad = mean(x["bad_delta"] for x in measured)
    return {"n": len(pairs), "tail_tokens": tail_tokens,
            "productive_logp_delta": good, "repetitive_logp_delta": bad,
            "selectivity": good - bad,
            "directionally_supported": bad < 0 and good > bad,
            "per_pair": measured,
            "note": "Positive selectivity means repetition continuations were suppressed "
                    "more than productive continuations; not proof of a unique mechanism."}


def isolate_patch(patch: dict[str, Any], layers: list[int]) -> dict[str, Any]:
    selected = [entry for entry in patch["entries"] if entry["layer"] in layers]
    if not selected or len(selected) != len(layers):
        raise ValueError("Isolation layer ids must match patch entries")
    return {**patch, "entries": selected, "method": "isolated_" + patch["method"]}
