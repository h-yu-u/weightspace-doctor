"""Read-only projection health checks. These indicators are not behavioral diagnoses."""

from typing import Any

import numpy as np

from .hf import (editable_float_linear, get_layer_container, is_quantized_model,
                 require_torch)


def audit_weights(model: Any, indices: list[int] | None = None, *,
                  chunk_cols: int = 512) -> dict[str, Any]:
    torch = require_torch()
    if is_quantized_model(model):
        raise ValueError("Floating-projection audit requires an unquantized model")
    layers = get_layer_container(model)
    indices = list(range(len(layers))) if indices is None else indices
    if not indices or min(indices) < 0 or max(indices) >= len(layers):
        raise ValueError("Audit layer indices out of range")
    rows = []
    for i in indices:
        projection = getattr(getattr(layers[i], "mlp", None), "down_proj", None)
        weight = getattr(projection, "weight", None)
        if weight is None:
            rows.append({"layer": i, "status": "no_down_proj"})
            continue
        if not editable_float_linear(projection):
            rows.append({"layer": i, "status": "packed_or_unsupported",
                         "note": "Cannot audit packed quantized values as floating-point weights"})
            continue
        count = invalid = zero = 0
        sumsq = 0.0
        max_abs = 0.0
        column_norms = []
        with torch.inference_mode():
            for offset in range(0, weight.shape[1], chunk_cols):
                x = weight[:, offset:offset + chunk_cols].detach().float()
                finite = torch.isfinite(x)
                count += x.numel()
                invalid += int((~finite).sum().item())
                zero += int((x == 0).sum().item())
                clean = torch.where(finite, x, torch.zeros_like(x))
                sumsq += float(torch.square(clean).sum().item())
                max_abs = max(max_abs, float(clean.abs().max().item()))
                column_norms.extend(torch.linalg.vector_norm(clean, dim=0).cpu().tolist())
        col = np.array(column_norms, dtype=np.float64)
        median = float(np.median(col))
        mad = float(np.median(np.abs(col - median)))
        outlier_cols = int(np.sum(col > median + 10 * max(mad, 1e-12)))
        health = "nonfinite_weights" if invalid else ("zero_matrix" if sumsq == 0 else "ok")
        rows.append({"layer": i, "status": health, "shape": list(weight.shape),
                     "dtype": str(weight.dtype), "nonfinite_count": invalid,
                     "zero_share": round(zero / count, 6),
                     "frobenius_norm": round(sumsq ** .5, 6),
                     "max_abs": round(max_abs, 6),
                     "column_norm_median": round(median, 6),
                     "column_norm_mad": round(mad, 6),
                     "extreme_column_norm_outliers": outlier_cols})
    return {"layers": rows,
            "warning": "No finite-weight anomaly does not imply model correctness. "
                       "Outlier norms are heuristic, not evidence of a bad circuit."}
