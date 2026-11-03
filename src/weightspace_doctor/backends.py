"""LLM inference adapters. The HTTP client sends prompts to the configured endpoint."""

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Protocol

from .schema import Case, Trace


class TextGenerator(Protocol):
    def generate(self, case: Case, *, max_new_tokens: int, temperature: float,
                 seed: int) -> Trace: ...


def openai_chat(base_url: str, model: str, messages: list[dict[str, str]], *,
                api_key: str = "", timeout: int = 120,
                params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Strictly bounded request; no token caching or arbitrary code execution."""
    url = base_url.rstrip("/") + "/chat/completions"
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Expected an absolute http(s) base URL, ending in /v1")
    payload = {"model": model, "messages": messages, **(params or {})}
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url,
                                     data=json.dumps(payload).encode("utf-8"),
                                     headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content = response.read(32 * 1024 * 1024)
        result = json.loads(content)
        if "choices" not in result:
            raise ValueError("Chat completion missing choices")
        return result
    except urllib.error.HTTPError as exc:
        body = exc.read(1200).decode("utf-8", errors="replace")
        raise RuntimeError(f"Chat completion HTTP {exc.code}: {body}") from exc


class OpenAIBackend:
    def __init__(self, base_url: str, model: str, *, api_key_env: str = "OPENAI_API_KEY",
                 timeout: int = 180):
        self.base_url = base_url
        self.model = model
        self.api_key = os.environ.get(api_key_env, "")
        self.timeout = timeout

    def generate(self, case: Case, *, max_new_tokens: int = 256,
                 temperature: float = 0.0, seed: int = 0) -> Trace:
        start = time.perf_counter()
        try:
            # Some OpenAI-compatible runtimes reject 'seed' or 'top_k'.
            # Pass only broadly supported Chat Completions parameters.
            params = {"max_tokens": max_new_tokens, "temperature": temperature, "seed": seed}
            messages = [{"role": "user", "content": case.prompt}]
            try:
                result = openai_chat(self.base_url, self.model, messages,
                                     api_key=self.api_key, timeout=self.timeout,
                                     params=params)
            except RuntimeError as exc:
                if "seed" not in str(exc).lower() or "HTTP 400" not in str(exc):
                    raise
                # Lightweight runtimes such as mlx_lm.server may omit seed.
                # This retry works, but API pairs cannot claim seeded sampling.
                result = openai_chat(self.base_url, self.model, messages,
                                     api_key=self.api_key, timeout=self.timeout,
                                     params={k: v for k, v in params.items() if k != "seed"})
            choice = result["choices"][0]
            msg = choice.get("message") or {}
            final_text = str(msg.get("content") or "")
            text = final_text
            # Thinking-mode endpoints sometimes expose reasoning_content separately.
            # Keep it for loop diagnosis; never assume it contains a scored final answer.
            reasoning = msg.get("reasoning_content") or ""
            if reasoning:
                text = str(reasoning) + "\n" + str(text)
            usage = result.get("usage") or {}
            return Trace(id=case.id, prompt=case.prompt, text=str(text),
                         finish_reason=str(choice.get("finish_reason") or "unknown"),
                         output_tokens=usage.get("completion_tokens"),
                         max_output_tokens=max_new_tokens,
                         elapsed_ms=int((time.perf_counter() - start) * 1000),
                         expected=case.expected, expected_regex=case.expected_regex,
                         split=case.split, tags=case.tags, final_text=final_text)
        except (RuntimeError, ValueError, KeyError, IndexError, urllib.error.URLError,
                TimeoutError) as exc:
            return Trace(id=case.id, prompt=case.prompt, text="", error=str(exc),
                         finish_reason="error", max_output_tokens=max_new_tokens,
                         elapsed_ms=int((time.perf_counter() - start) * 1000),
                         expected=case.expected, expected_regex=case.expected_regex,
                         split=case.split, tags=case.tags)


def generate_cases(generator: TextGenerator, cases: list[Case], *,
                   max_new_tokens: int = 256, temperature: float = 0.0,
                   seed: int = 0) -> list[Trace]:
    return [generator.generate(case, max_new_tokens=max_new_tokens,
                               temperature=temperature, seed=seed)
            for case in cases]
