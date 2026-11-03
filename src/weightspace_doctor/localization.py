"""Gradient-free paired activation localization and MLP writer ranking.

The writer ranking is an association/hypothesis, not a proof of a causal circuit.
Only forward hooks; never backward passes, learned probes or optimizer steps.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .hf import (canonical_prefix, editable_float_linear, format_prefix,
                 get_layer_container, is_quantized_model, require_torch)
from .io import write_json
from .schema import ContrastPair


@dataclass(frozen=True)
class LayerEvidence:
    layer: int
    module: str
    score: float
    coherence: float
    separation: float
    writer_columns: tuple[int, ...]
    writer_energy_share: float
    writer_method: str
    editable: bool

    def to_dict(self) -> dict[str, Any]:
        return {"layer": self.layer, "module": self.module, "score": self.score,
                "coherence": self.coherence, "separation": self.separation,
                "writer_columns": list(self.writer_columns),
                "writer_energy_share": self.writer_energy_share,
                "writer_method": self.writer_method, "editable": self.editable}


def unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector.astype(np.float64))
    if not np.isfinite(norm) or norm < 1e-10:
        raise ValueError("Activation contrast direction is nonfinite or numerically zero")
    return (vector / norm).astype(np.float32)


def contrast_statistics(good: np.ndarray, bad: np.ndarray):
    """[pairs, hidden] -> normalized bad-minus-good direction and robust rank features."""
    if good.shape != bad.shape or good.ndim != 2 or good.shape[0] == 0:
        raise ValueError("Expected two aligned [number_of_pairs, hidden_dim] matrices")
    diff = bad.astype(np.float32) - good.astype(np.float32)
    mean = diff.mean(axis=0)
    direction = unit(mean)
    norms = np.linalg.norm(diff, axis=1)
    pair_cos = (diff @ direction) / np.maximum(norms, 1e-10)
    coherence = float(np.mean(pair_cos))
    separation = float(np.linalg.norm(mean) / max(float(np.mean(norms)), 1e-10))
    base_norm = max(float(np.mean(np.linalg.norm(good, axis=1))), 1e-10)
    score = float(np.linalg.norm(mean) / base_norm * max(coherence, 0))
    return direction, {"score": round(score, 6),
                       "coherence": round(coherence, 6),
                       "separation": round(separation, 6)}


def top_writers(weight: Any, direction: np.ndarray, *, k: int = 64,
                activation_delta: np.ndarray | None = None,
                chunk_size: int = 1024) -> tuple[list[int], float, str]:
    """Rank writer columns with activation-weighted signed contributions.

    Contribution score = (u^T W[:,j]) * mean(bad_activation_j-good_activation_j).
    Positive scores are hypothesized to write the unwanted direction more on
    bad continuations. Remaining columns are ranked by |u^T W[:,j]|, not randomly.

    If activations are unavailable, revert to the weaker weight-only heuristic.
    """
    torch = require_torch()
    if len(weight.shape) != 2 or weight.shape[0] != direction.size:
        raise ValueError("Direction and output projection have incompatible dimensions")
    if not weight.dtype.is_floating_point:
        raise ValueError("Quantized/packed projection weights are not editable")
    if not 1 <= k <= weight.shape[1]:
        raise ValueError("top-k must fit the projection's intermediate dimension")
    projected: list[np.ndarray] = []
    with torch.inference_mode():
        d = torch.as_tensor(direction, dtype=torch.float32, device=weight.device)
        for offset in range(0, weight.shape[1], chunk_size):
            block = weight[:, offset:offset + chunk_size].float()
            proj = d @ block
            projected.append(proj.detach().cpu().numpy())
    projection = np.concatenate(projected)
    if not np.all(np.isfinite(projection)):
        raise ValueError("Nonfinite projection weights; run weight-audit first")

    if activation_delta is None:
        method = "projection_only_no_activations"
        importance = np.abs(projection)
        order = np.argsort(-importance, kind="stable")
        score_for_energy = importance
    else:
        if activation_delta.shape != projection.shape or not np.all(np.isfinite(activation_delta)):
            raise ValueError("MLP activation delta is incompatible or nonfinite")
        signed = projection * activation_delta.astype(np.float32)
        positives = np.maximum(signed, 0)
        if np.any(positives > 0):
            method = "activation_weighted_positive_writer"
            # For the tail of k beyond the positive writers, use magnitude
            # to break ties rather than choosing arbitrary zero-score columns.
            order = np.lexsort((np.arange(len(projection)), -np.abs(projection), -positives))
            score_for_energy = np.abs(signed)
        else:
            method = "projection_only_fallback_no_positive_writer"
            score_for_energy = np.abs(projection)
            order = np.argsort(-score_for_energy, kind="stable")
    indices = order[:k]
    denominator = float(np.dot(score_for_energy, score_for_energy))
    energy = float(np.dot(score_for_energy[indices], score_for_energy[indices]) /
                   max(denominator, 1e-20))
    return [int(v) for v in indices], round(energy, 6), method


def capture_layer_features(model: Any, tokenizer: Any, text: str,
                           layer_indices: list[int], *, tail_tokens: int = 16,
                           mlp_inputs: bool = True) -> tuple[dict[int, np.ndarray],
                                                              dict[int, np.ndarray]]:
    """Collect only tail-mean residuals and down_proj inputs, with safe hook cleanup."""
    torch = require_torch()
    layers = get_layer_container(model)
    if (not layer_indices or len(set(layer_indices)) != len(layer_indices) or
            min(layer_indices) < 0 or max(layer_indices) >= len(layers) or
            tail_tokens < 1):
        raise ValueError("Invalid layer list or tail_tokens")
    residuals: dict[int, np.ndarray] = {}
    activations: dict[int, np.ndarray] = {}
    hooks = []

    def tail_mean(hidden: Any, index: int) -> np.ndarray:
        if not torch.is_tensor(hidden) or hidden.ndim != 3:
            raise ValueError(f"Layer {index} activation is not [batch,seq,hidden]")
        # Do not keep an entire attention map or full activation tensor in memory.
        return (hidden[0, -tail_tokens:, :].detach().float()
                .mean(dim=0).cpu().numpy())

    def residual_hook(index: int):
        def hook(_module: Any, _inputs: Any, output: Any):
            hidden = output[0] if isinstance(output, tuple) else output
            residuals[index] = tail_mean(hidden, index)
        return hook

    def pre_hook(index: int):
        def hook(_module: Any, inputs: Any):
            activations[index] = tail_mean(inputs[0], index)
        return hook

    try:
        for i in layer_indices:
            hooks.append(layers[i].register_forward_hook(residual_hook(i)))
            down = getattr(getattr(layers[i], "mlp", None), "down_proj", None)
            if mlp_inputs and down is not None:
                hooks.append(down.register_forward_pre_hook(pre_hook(i)))
        tensors = tokenizer(text, return_tensors="pt")
        device = model.get_input_embeddings().weight.device
        tensors = {name: data.to(device) for name, data in tensors.items()}
        with torch.inference_mode():
            model(**tensors, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    if len(residuals) != len(layer_indices):
        raise RuntimeError("Did not capture every requested decoder layer")
    return residuals, activations


def capture_residuals(model: Any, tokenizer: Any, text: str,
                      layer_indices: list[int], *, tail_tokens: int = 16) -> dict[int, np.ndarray]:
    residuals, _ = capture_layer_features(model, tokenizer, text, layer_indices,
                                         tail_tokens=tail_tokens, mlp_inputs=False)
    return residuals


def localize_model(model: Any, tokenizer: Any, pairs: list[ContrastPair], *,
                   indices: list[int] | None = None, topk: int = 64,
                   tail_tokens: int = 16, thinking: bool = True) -> dict[str, Any]:
    """Contrastive forward-based localization; cannot alone prove a causal circuit."""
    if not pairs:
        raise ValueError("At least one contrast pair required")
    layers = get_layer_container(model)
    indices = indices if indices is not None else list(range(len(layers)))
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("Layer list cannot be empty or contain duplicates")
    residual_good: dict[int, list[np.ndarray]] = {i: [] for i in indices}
    residual_bad: dict[int, list[np.ndarray]] = {i: [] for i in indices}
    act_good: dict[int, list[np.ndarray]] = {i: [] for i in indices}
    act_bad: dict[int, list[np.ndarray]] = {i: [] for i in indices}
    for pair in pairs:
        prefix = format_prefix(tokenizer, pair.prompt, thinking=thinking)
        good, good_act = capture_layer_features(model, tokenizer, prefix + pair.good,
                                                indices, tail_tokens=tail_tokens)
        bad, bad_act = capture_layer_features(model, tokenizer, prefix + pair.bad,
                                              indices, tail_tokens=tail_tokens)
        for i in indices:
            residual_good[i].append(good[i])
            residual_bad[i].append(bad[i])
            if i in good_act and i in bad_act:
                act_good[i].append(good_act[i])
                act_bad[i].append(bad_act[i])

    evidence: list[LayerEvidence] = []
    dirs: dict[str, np.ndarray] = {}
    prefix = canonical_prefix(model)
    for i in indices:
        direction, stats = contrast_statistics(np.stack(residual_good[i]),
                                               np.stack(residual_bad[i]))
        dirs[f"layer_{i}"] = direction
        mlp = getattr(layers[i], "mlp", None)
        down = getattr(mlp, "down_proj", None)
        weight = getattr(down, "weight", None)
        editable = bool(not is_quantized_model(model) and editable_float_linear(down) and
                        weight.shape[0] == direction.size)
        columns: list[int] = []
        energy = 0.0
        method = "unsupported_projection"
        if editable:
            activation_delta = None
            if len(act_good[i]) == len(pairs) and len(act_bad[i]) == len(pairs):
                activation_delta = (np.mean(act_bad[i], axis=0) -
                                    np.mean(act_good[i], axis=0))
            columns, energy, method = top_writers(
                weight, direction, k=min(topk, weight.shape[1]),
                activation_delta=activation_delta)
        evidence.append(LayerEvidence(i, f"{prefix}.{i}.mlp.down_proj",
                                      stats["score"], stats["coherence"],
                                      stats["separation"], tuple(columns), energy,
                                      method, editable))
    ranked = sorted(evidence, key=lambda item: item.score, reverse=True)
    return {"pairs": len(pairs), "pair_ids": [p.id for p in pairs],
            "method": "residual_contrast_and_activation_weighted_mlp_writer_contribution",
            "evidence_level": "hypothesis_only_no_causal_proof",
            "layers": [entry.to_dict() for entry in ranked], "directions": dirs}


def save_localization(path: str | Path, result: dict[str, Any]) -> None:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p / "directions.npz", **result["directions"])
    write_json(p / "localization.json", {k: v for k, v in result.items()
                                         if k != "directions"})


def read_localization(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    import json
    result = json.loads((p / "localization.json").read_text(encoding="utf-8"))
    with np.load(p / "directions.npz", allow_pickle=False) as arr:
        result["directions"] = {key: arr[key].copy() for key in arr.files}
    return result
