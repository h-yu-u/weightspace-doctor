"""Explicit sparse projector patches; reversible, hash-checked and quantization aware."""

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .hf import (editable_float_linear, get_layer_container, is_quantized_model,
                 require_torch)
from .io import write_json
from .localization import unit


def matrix_sha256(weight: Any, *, chunk_cols: int = 512) -> str:
    """Hash native dtype bytes in stable column chunks, without full FP32 copies."""
    torch = require_torch()
    h = hashlib.sha256()
    with torch.inference_mode():
        for pos in range(0, weight.shape[1], chunk_cols):
            block = weight[:, pos:pos + chunk_cols].detach().contiguous().cpu()
            h.update(block.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def values_sha256(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr.astype(np.float32)).tobytes()).hexdigest()


def relative_update_norm(weight: Any, delta: np.ndarray) -> float:
    torch = require_torch()
    total = 0.0
    with torch.inference_mode():
        for offset in range(0, weight.shape[1], 512):
            block = weight[:, offset:offset + 512].float()
            total += float(torch.square(block).sum().item())
    return float(np.linalg.norm(delta.astype(np.float64)) / max(total ** .5, 1e-12))


def _compute_columns(weight: Any, columns: list[int], direction: np.ndarray,
                     alpha: float, *, target_delta_norm: float | None = None):
    torch = require_torch()
    # Floating-point full precision only: int/quantized/packed weights must refuse.
    if not weight.dtype.is_floating_point:
        raise ValueError("Quantized model: edit the BF16/FP16 source, then requantize")
    with torch.inference_mode():
        before = weight[:, columns].detach().float().cpu().numpy().copy()
        u = unit(direction)
        delta = -alpha * np.outer(u, u @ before)
        if target_delta_norm is not None:
            norm = float(np.linalg.norm(delta))
            if norm < 1e-9:
                raise ValueError("Orthogonal negative-control projection is zero")
            delta *= target_delta_norm / norm
        casted = torch.as_tensor(before + delta, dtype=weight.dtype)
        after = casted.float().numpy().copy()
    return before, after


def build_patch(model: Any, localization: dict[str, Any], *, model_id: str,
                alpha: float = 1.0, top_layers: int = 2, topk: int = 64,
                chosen_layers: list[int] | None = None,
                max_changed_fraction: float = 0.05,
                max_relative_change: float = 0.18) -> dict[str, Any]:
    if not 0 < alpha <= 10:
        raise ValueError("alpha must be in (0,10]")
    if not 1 <= top_layers <= 8 or topk <= 0:
        raise ValueError("top_layers must be 1..8, and topk must be positive")
    if is_quantized_model(model):
        raise ValueError("Direct editing of quantized checkpoints is forbidden")
    layers = get_layer_container(model)
    candidates = [d for d in localization["layers"] if d["editable"] and
                  d["coherence"] > 0 and d["writer_columns"]]
    if chosen_layers is not None:
        available = {d["layer"]: d for d in candidates}
        if not chosen_layers or not set(chosen_layers).issubset(available):
            raise ValueError("Chosen layers must be editable and present in localization")
        chosen = [available[i] for i in chosen_layers]
    else:
        chosen = candidates[:top_layers]
    if not chosen:
        raise ValueError("No editable coherent layer candidates. Collect more contrast pairs")
    entries: list[dict[str, Any]] = []
    for data in chosen:
        i = data["layer"]
        if i < 0 or i >= len(layers):
            raise ValueError(f"Invalid layer index {i}")
        down = getattr(getattr(layers[i], "mlp", None), "down_proj", None)
        weight = getattr(down, "weight", None)
        if not editable_float_linear(down):
            raise ValueError(f"Layer {i} projection is unsupported/quantized")
        columns = sorted(data["writer_columns"][:topk])
        if not columns or len(set(columns)) != len(columns) or max(columns) >= weight.shape[1]:
            raise ValueError("Invalid sparse column selection")
        changed_fraction = len(columns) / weight.shape[1]
        if changed_fraction > max_changed_fraction:
            raise ValueError(f"Layer {i}: {changed_fraction:.1%} columns exceeds edit budget")
        u = localization["directions"][f"layer_{i}"]
        if weight.shape[0] != len(u):
            raise ValueError("Projection hidden size differs from activation direction")
        before, after = _compute_columns(weight, columns, u, alpha)
        if np.array_equal(before, after):
            raise ValueError(f"Layer {i}: edit rounded to zero at {weight.dtype}")
        relative = relative_update_norm(weight, after - before)
        if relative > max_relative_change:
            raise ValueError(f"Layer {i} update norm={relative:.2%} exceeds "
                             f"budget {max_relative_change:.2%}; lower alpha")
        entries.append({"layer": i, "module": data["module"],
                        "shape": list(weight.shape), "dtype": str(weight.dtype),
                        "columns": columns, "before": before, "after": after,
                        "original_full_sha256": matrix_sha256(weight),
                        "before_sha256": values_sha256(before),
                        "after_sha256": values_sha256(after),
                        "relative_update_norm": relative,
                        "column_fraction": changed_fraction,
                        "score": data["score"]})
    return {"model_id": model_id, "method": "sparse_rank_one_projection_v1",
            "alpha": alpha, "pairs": localization["pairs"],
            "discovery_ids": localization.get("pair_ids", []),
            "control": False, "entries": entries,
            "evidence_level": "candidate_not_validated"}


def build_random_control(model: Any, patch: dict[str, Any], *, seed: int = 2026,
                         max_relative_change: float = 0.18) -> dict[str, Any]:
    """Same columns and exact per-layer update L2; random orthogonal direction."""
    layers = get_layer_container(model)
    rng = np.random.default_rng(seed)
    entries = []
    for entry in patch["entries"]:
        original = entry["before"]
        shift = entry["after"] - original
        # Dominant edited direction from truncated SVD, rather than requiring
        # localization to be present when a patch is reloaded.
        left, _, _ = np.linalg.svd(shift.astype(np.float64), full_matrices=False)
        u = left[:, 0]
        random_u = rng.normal(size=u.shape)
        random_u -= (random_u @ u) * u
        random_u = unit(random_u)
        weight = layers[entry["layer"]].mlp.down_proj.weight
        before, after = _compute_columns(weight, entry["columns"], random_u, 1.0,
                                         target_delta_norm=float(np.linalg.norm(shift)))
        rounded_delta_norm = float(np.linalg.norm((after - before).astype(np.float64)))
        target_norm = float(np.linalg.norm(shift.astype(np.float64)))
        if (abs(rounded_delta_norm - target_norm) / max(target_norm, 1e-10) > 0.10):
            raise ValueError("Negative control dtype rounding prevents fair norm matching")
        relative = relative_update_norm(weight, after - before)
        if relative > max_relative_change:
            raise ValueError("Negative control update unexpectedly exceeds norm budget")
        entries.append({**entry, "before": before, "after": after,
                        "before_sha256": values_sha256(before),
                        "after_sha256": values_sha256(after),
                        "relative_update_norm": relative})
    return {**patch, "method": "orthogonal_random_norm_matched_control_v1",
            "control": True, "entries": entries, "evidence_level": "negative_control"}


def save_patch(folder: str | Path, patch: dict[str, Any]) -> None:
    p = Path(folder)
    p.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    manifests = []
    for i, e in enumerate(patch["entries"]):
        arrays[f"p{i}_before"] = e["before"].astype(np.float32)
        arrays[f"p{i}_after"] = e["after"].astype(np.float32)
        manifests.append({k: v for k, v in e.items() if k not in ("before", "after")})
    np.savez_compressed(p / "values.npz", **arrays)
    write_json(p / "manifest.json", {**patch, "entries": manifests})


def load_patch(folder: str | Path) -> dict[str, Any]:
    p = Path(folder)
    manifest = json.loads((p / "manifest.json").read_text(encoding="utf-8"))
    entries = []
    with np.load(p / "values.npz", allow_pickle=False) as data:
        for i, e in enumerate(manifest["entries"]):
            before, after = data[f"p{i}_before"].copy(), data[f"p{i}_after"].copy()
            if values_sha256(before) != e["before_sha256"] or (
                    values_sha256(after) != e["after_sha256"]):
                raise ValueError("Patch file hash mismatch, refuse to load")
            entries.append({**e, "before": before, "after": after})
    return {**manifest, "entries": entries}


def _validated_weights(model: Any, patch: dict[str, Any], *, revert: bool):
    """Check ALL preconditions before touching ANY tensor."""
    if is_quantized_model(model):
        raise ValueError("Direct editing of quantized checkpoints is forbidden")
    layers = get_layer_container(model)
    checked = []
    seen_layers: set[int] = set()
    for e in patch["entries"]:
        i, cols = e["layer"], e["columns"]
        if i in seen_layers:
            raise ValueError("Patch contains duplicate target layers")
        seen_layers.add(i)
        if not (isinstance(i, int) and 0 <= i < len(layers)):
            raise ValueError("Layer index not in model")
        if len(set(cols)) != len(cols) or sorted(cols) != cols:
            raise ValueError("Patch column indices must be unique and sorted")
        down = getattr(getattr(layers[i], "mlp", None), "down_proj", None)
        weight = getattr(down, "weight", None)
        if (not editable_float_linear(down) or
                list(weight.shape) != e["shape"] or str(weight.dtype) != e["dtype"] or
                not cols or cols[0] < 0 or cols[-1] >= weight.shape[1]):
            raise ValueError(f"Layer {i} incompatible, quantized or unsafe")
        expected_shape = (weight.shape[0], len(cols))
        if (e["before"].shape != expected_shape or
                e["after"].shape != expected_shape):
            raise ValueError(f"Layer {i}: patch columns have invalid matrix shape")
        with require_torch().inference_mode():
            observed = weight[:, cols].detach().float().cpu().numpy()
        expected = e["after_sha256"] if revert else e["before_sha256"]
        if values_sha256(observed) != expected:
            raise ValueError(f"Layer {i}: weight checksum mismatch; refusing edit")
        if not revert and matrix_sha256(weight) != e["original_full_sha256"]:
            raise ValueError(f"Layer {i}: base tensor changed outside patch columns")
        checked.append((weight, e))
    return checked


def apply_patch(model: Any, patch: dict[str, Any], *, revert: bool = False) -> None:
    torch = require_torch()
    checked = _validated_weights(model, patch, revert=revert)
    completed = []
    try:
        with torch.no_grad():
            for weight, e in checked:
                arr = e["before"] if revert else e["after"]
                value = torch.as_tensor(arr, dtype=weight.dtype, device=weight.device)
                weight[:, e["columns"]] = value
                completed.append((weight, e))
    except Exception:
        # Roll back all previously changed tensors on failure.
        with torch.no_grad():
            for weight, e in reversed(completed):
                arr = e["after"] if revert else e["before"]
                weight[:, e["columns"]] = torch.as_tensor(arr, dtype=weight.dtype,
                                                           device=weight.device)
        raise


@contextmanager
def temporary_patch(model: Any, patch: dict[str, Any]) -> Iterator[None]:
    apply_patch(model, patch)
    try:
        yield
    finally:
        apply_patch(model, patch, revert=True)
