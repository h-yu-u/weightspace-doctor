"""Fast local checkpoint metadata audit: config/index only, no model load."""

import json
from pathlib import Path
from typing import Any


def inspect_model_dir(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_dir():
        raise ValueError("inspect requires a local model directory")
    cfg = p / "config.json"
    if not cfg.exists():
        raise FileNotFoundError(cfg)
    config = json.loads(cfg.read_text(encoding="utf-8"))
    text_cfg = config.get("text_config", config)
    quant = (config.get("quantization_config") or text_cfg.get("quantization_config")
             or config.get("quantization") or text_cfg.get("quantization"))
    file_index = next(iter(sorted(p.glob("*.safetensors.index.json"))), None)
    weights = {}
    if file_index:
        index = json.loads(file_index.read_text(encoding="utf-8"))
        weights = index.get("weight_map", {})
    else:
        # safetensors metadata is read without materializing tensor values
        for file in sorted(p.glob("*.safetensors")):
            try:
                from safetensors import safe_open
                with safe_open(str(file), framework="pt", device="cpu") as handle:
                    weights.update({key: file.name for key in handle.keys()})
            except ImportError:
                weights[file.name] = "safetensors_extra_required_for_tensor_keys"
    approx_files = sorted(p.glob("*.safetensors"))
    total_bytes = sum(file.stat().st_size for file in approx_files)
    return {"model_dir": str(p.resolve()), "model_type": text_cfg.get("model_type"),
            "layers": text_cfg.get("num_hidden_layers"),
            "hidden_size": text_cfg.get("hidden_size"),
            "intermediate_size": text_cfg.get("intermediate_size"),
            "quantization": quant, "safetensors_files": len(approx_files),
            "size_gb": round(total_bytes / (1024 ** 3), 3),
            "weight_tensors": len(weights),
            "down_proj_tensors": sorted(k for k in weights if k.endswith("mlp.down_proj.weight")),
            "edit_support": "NO direct packed-4bit writes" if quant else
                            "HF floating-point weights: after shape/dtype checks",
            "note": "Metadata inspection cannot diagnose model behavior or check hidden backdoors"}
