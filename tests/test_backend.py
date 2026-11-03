"""Offline mock server integration: no third-party API or internet needed."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from weightspace_doctor.backends import OpenAIBackend, openai_chat
from weightspace_doctor.cli import main
from weightspace_doctor.hf import extract_final_answer
from weightspace_doctor.metrics import correctness
from weightspace_doctor.schema import Case, Trace


class MockModelAPI(BaseHTTPRequestHandler):
    requests_seen = []

    def do_POST(self):
        assert self.path == "/v1/chat/completions"
        n = int(self.headers.get("content-length", "0"))
        item = json.loads(self.rfile.read(n))
        type(self).requests_seen.append(item)
        if "seed" in item:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error": "unsupported seed"}')
            return
        payload = {"choices": [{"message": {"content": "42",
                                            "reasoning_content": "Let me add 12 and 30."},
                                "finish_reason": "stop"}],
                   "usage": {"completion_tokens": 12}}
        content = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, fmt, *args):
        pass


@pytest.fixture
def local_api():
    MockModelAPI.requests_seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockModelAPI)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)


def test_http_probe_seed_fallback_and_reasoning(local_api):
    backend = OpenAIBackend(local_api, "test-model")
    result = backend.generate(Case("t", "12+30", expected="42"), seed=12)
    assert result.error is None
    assert result.text.startswith("Let me add")
    assert result.final_text == "42"
    assert correctness(result)
    assert len(MockModelAPI.requests_seen) == 2
    assert "seed" in MockModelAPI.requests_seen[0]
    assert "seed" not in MockModelAPI.requests_seen[1]


def test_cli_probe_integration(local_api, tmp_path):
    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps({"id": "a", "prompt": "12+30", "expected": "42"}) + "\n")
    output = tmp_path / "probe"
    assert main(["probe", "--cases", str(cases), "--base-url", local_api,
                 "--model", "test-model", "--out", str(output)]) == 0
    data = json.loads((output / "diagnosis.json").read_text())
    assert data["accuracy"] == 1.0
    assert data["count"] == 1


def test_http_rejects_bad_url():
    with pytest.raises(ValueError):
        openai_chat("file:///tmp", "test-model", [{"role": "user", "content": "hi"}])


def test_qwen_thinking_boundaries_are_not_scored_as_final():
    assert extract_final_answer("<|im_start|>assistant<|channel=analysis|>Step 1."
                                "<|/think|>\\boxed{42}<|im_end|>",
                                "Step 1.\\boxed{42}") == "\\boxed{42}"
    assert extract_final_answer("Thinking.</think>DONE", "Thinking.DONE") == "DONE"
    assert correctness(Trace("t", "p", "Thinking <|/think|>\\boxed{42}",
                             expected="42")) is True
    assert correctness(Trace("t", "p", "42 <|/think|>incorrect",
                             expected_regex=r"^42$")) is False


def test_builtin_quick_suite_produces_complete_report(local_api, tmp_path):
    output = tmp_path / "quick"
    assert main(["quick", "--profile", "code", "--base-url", local_api,
                 "--model", "test-model", "--out", str(output)]) == 0
    assert (output / "cases.jsonl").is_file()
    results = json.loads((output / "diagnosis.json").read_text())
    assert results["count"] == 13
    assert results["labeled"] == 13
    assert (output / "report.md").exists()


def test_builtin_suites_distinct_ids_and_splits():
    from weightspace_doctor.suites import built_in_cases
    for p in ("general", "math", "code"):
        cases = built_in_cases(p)
        assert len({x.id for x in cases}) == len(cases)
        assert sum(x.split == "target" for x in cases) >= 5
        assert sum(x.split == "guard" for x in cases) >= 5
        assert all(x.expected is not None or x.expected_regex is not None for x in cases)


def test_qwen35_uses_text_only_hf_class(monkeypatch):
    import sys
    import types
    from weightspace_doctor.hf import load_hf_model

    mod = types.ModuleType("transformers")
    selected = []

    class Config:
        @staticmethod
        def from_pretrained(model_id, trust_remote_code):
            return types.SimpleNamespace(model_type="qwen3_5")

    class TextModel:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            selected.append(("text", model_id, kwargs))
            return types.SimpleNamespace(eval=lambda: None)

    class GenericModel:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            raise AssertionError("Should not load the multimodal generic CausalLM path")

    class Tokenizer:
        @staticmethod
        def from_pretrained(model_id, trust_remote_code):
            return "tokenizer"

    mod.AutoConfig = Config
    mod.Qwen3_5ForCausalLM = TextModel
    mod.AutoModelForCausalLM = GenericModel
    mod.AutoTokenizer = Tokenizer
    monkeypatch.setitem(sys.modules, "transformers", mod)
    model, tokenizer = load_hf_model("Qwen/Qwen3.5-0.8B", device="cpu")
    assert model is not None and tokenizer == "tokenizer"
    assert selected[0][0] == "text"
    assert selected[0][2]["trust_remote_code"] is False
