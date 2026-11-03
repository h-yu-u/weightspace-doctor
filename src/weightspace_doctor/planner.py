"""Optional structured LLM hypotheses. Agent suggestions NEVER execute arbitrary code."""

import json
import os
from typing import Any

from .backends import openai_chat


def recommend_layers(localization: dict[str, Any], *, top_layers: int = 2) -> list[int]:
    candidates = [d for d in localization["layers"] if d.get("editable") and
                  d.get("coherence", 0) > 0 and d.get("writer_columns")]
    return [int(row["layer"]) for row in candidates[:top_layers]]


def llm_plan(localization: dict[str, Any], diagnostics: dict[str, Any], *,
             base_url: str, model: str, api_key_env: str = "OPENAI_API_KEY",
             timeout: int = 60) -> dict[str, Any]:
    """Whitelist-only edits: alpha+known layer IDs; no Python, shell or file operations."""
    valid_layers = recommend_layers(localization, top_layers=8)
    if not valid_layers:
        raise ValueError("No editable candidates")
    evidence = [{k: v for k, v in row.items() if k != "writer_columns"}
                for row in localization["layers"]]
    task = {"instruction": "Select <=2 suspicious layers and alpha for a reversible "
                           "sparse projector patch. Return JSON only with keys 'layers', "
                           "'alpha', 'reason'. This is only a hypothesis, not a diagnosis.",
            "allowed_layers": valid_layers, "allowed_alphas": [0.5, 1.0, 2.0, 4.0, 6.0],
            "layers_evidence": evidence, "observed_diagnostics": diagnostics}
    result = openai_chat(base_url, model,
                         [{"role": "user", "content": json.dumps(task)}],
                         api_key=os.environ.get(api_key_env, ""), timeout=timeout,
                         params={"temperature": 0, "max_tokens": 400})
    raw = result["choices"][0]["message"].get("content", "")
    try:
        plan = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Planner returned invalid JSON (no action was taken)") from exc
    layers = plan.get("layers")
    if (not isinstance(layers, list) or not 1 <= len(layers) <= 2 or
            len(set(map(str, layers))) != len(layers) or
            any(type(i) is not int or i not in valid_layers for i in layers) or
            type(plan.get("alpha")) not in (int, float) or
            float(plan["alpha"]) not in (0.5, 1.0, 2.0, 4.0, 6.0)):
        raise ValueError("Planner output violates weight-edit whitelist")
    return {"layers": layers, "alpha": float(plan["alpha"]),
            "reason": str(plan.get("reason", ""))[:800], "origin": "llm_hypothesis"}
