"""Lazily loaded HF model adapter; no imports required for fast log scanning."""

import time
import re
from collections.abc import Sequence
from typing import Any

from .schema import Case, Trace


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install `pip install -e '.[hf]'` for weight operations") from exc
    return torch


def load_hf_model(model_id: str, *, device: str = "cpu"):
    try:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install `pip install -e '.[hf]'` (transformers >= 5.11)") from exc
    kwargs: dict[str, Any] = {"dtype": "auto", "trust_remote_code": False}
    if device == "auto":
        kwargs["device_map"] = "auto"
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
    # Qwen3.5 official checkpoints can be multimodal. Loading the dedicated
    # text-only class avoids vision weights and preserves the decoder modules
    # edited in the authors' experiment (HF's documented text path).
    if getattr(config, "model_type", None) == "qwen3_5":
        try:
            from transformers import Qwen3_5ForCausalLM
        except ImportError as exc:
            raise RuntimeError("This Qwen3.5 checkpoint needs recent Transformers") from exc
        cls = Qwen3_5ForCausalLM
    else:
        cls = AutoModelForCausalLM
    model = cls.from_pretrained(model_id, **kwargs)
    if device not in ("cpu", "auto"):
        model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    model.eval()
    return model, tokenizer


def is_quantized_model(model: Any) -> bool:
    if getattr(model, "is_quantized", False):
        return True
    if getattr(model, "quantization_method", None) is not None:
        return True
    config = getattr(model, "config", None)
    if config is None:
        return False
    text = getattr(config, "text_config", None)
    return any(getattr(cfg, "quantization_config", None) is not None
               for cfg in (config, text) if cfg is not None)


def editable_float_linear(module: Any) -> bool:
    """Only ordinary unquantized torch.Linear with supported native float weights.

    Some quantized wrappers expose deceptively floating `.weight` tensors while
    relying on out-of-band quant_state/scales.  Do not patch such wrappers.
    """
    torch = require_torch()
    if not isinstance(module, torch.nn.Linear):
        return False
    weight = getattr(module, "weight", None)
    if weight is None or weight.ndim != 2:
        return False
    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        return False
    return not any(getattr(obj, name, None) is not None for obj in (module, weight)
                   for name in ("quant_state", "qweight", "quant_scales", "quantization_config"))


def get_layer_container(model: Any) -> Sequence[Any]:
    """Covers HF Qwen3/3.5 text-only and multimodal, Llama and Gemma families."""
    paths = ("model.language_model.layers", "model.layers", "language_model.layers",
             "language_model.model.layers", "model.model.layers", "transformer.h")
    for path in paths:
        found = resolve_path(model, path)
        if found is not None and hasattr(found, "__len__") and len(found):
            return found
    raise ValueError("Cannot locate transformer layers; architecture unsupported for edits")


def resolve_path(base: Any, path: str) -> Any | None:
    current = base
    for key in path.split("."):
        if not hasattr(current, key):
            return None
        current = getattr(current, key)
    return current


def canonical_prefix(model: Any) -> str:
    for path in ("model.language_model.layers", "model.layers", "language_model.layers",
                 "language_model.model.layers", "model.model.layers", "transformer.h"):
        if resolve_path(model, path) is not None:
            return path
    raise ValueError("No decoder layer container found")


def format_prefix(tokenizer: Any, prompt: str, *, thinking: bool) -> str:
    if not getattr(tokenizer, "chat_template", None):
        return prompt
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=thinking)
    except TypeError:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True)


def extract_final_answer(raw: str, clean: str) -> str:
    """Extract a response, not chain-of-thought, for scored final answers.

    Qwen uses <|/think|>, while some servers use </think>.  If there is no
    reliable boundary, do not claim to separate reasoning from the answer.
    """
    for marker in ("<|/think|>", "</think>"):
        if marker in raw:
            tail = raw.rsplit(marker, 1)[-1]
            tail = re.sub(r"<\|(?:im_end|endoftext|eot_id|fim_suffix)\|>", "", tail)
            return tail.strip()
    return clean


class HFBackend:
    def __init__(self, model: Any, tokenizer: Any, *, thinking: bool = True):
        self.model = model
        self.tokenizer = tokenizer
        self.thinking = thinking

    def generate(self, case: Case, *, max_new_tokens: int = 256,
                 temperature: float = 0.0, seed: int = 0) -> Trace:
        torch = require_torch()
        started = time.perf_counter()
        try:
            prompt = format_prefix(self.tokenizer, case.prompt, thinking=self.thinking)
            inputs = self.tokenizer(prompt, return_tensors="pt")
            device = self.model.get_input_embeddings().weight.device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            kwargs = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0}
            if temperature > 0:
                kwargs["temperature"] = temperature
            with torch.inference_mode():
                generated = self.model.generate(**inputs, **kwargs)
            ids = generated[0, inputs["input_ids"].shape[-1]:].tolist()
            # Keep a raw decoding copy: skip_special_tokens=True erases Qwen's
            # <|/think|> boundary, making reasoning+final answers indistinguishable.
            raw = self.tokenizer.decode(ids, skip_special_tokens=False)
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            final = extract_final_answer(raw, text)
            eos = self.model.generation_config.eos_token_id
            eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
            if getattr(self.tokenizer, "eos_token_id", None) is not None:
                eos_ids.add(self.tokenizer.eos_token_id)
            reason = "eos" if ids and ids[-1] in eos_ids else (
                "length" if len(ids) >= max_new_tokens else "stop")
            return Trace(case.id, case.prompt, text, reason, len(ids), max_new_tokens,
                         int((time.perf_counter() - started) * 1000), case.expected,
                         case.expected_regex, case.split, case.tags,
                         final_text=final)
        except Exception as exc:
            return Trace(id=case.id, prompt=case.prompt, text="", finish_reason="error",
                         error=f"{type(exc).__name__}: {exc}",
                         max_output_tokens=max_new_tokens,
                         elapsed_ms=int((time.perf_counter() - started) * 1000),
                         expected=case.expected, expected_regex=case.expected_regex,
                         split=case.split, tags=case.tags)
