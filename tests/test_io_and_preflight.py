import json

import pytest

from weightspace_doctor.cli import main
from weightspace_doctor.io import jsonl_read, read_cases, write_jsonl
from weightspace_doctor.preflight import inspect_model_dir


def test_jsonl_and_duplicate_case_ids(tmp_path):
    path = tmp_path / "cases.jsonl"
    write_jsonl(path, [{"id": "a", "prompt": "hi"}, {"id": "a", "prompt": "yo"}])
    with pytest.raises(ValueError, match="Duplicate"):
        read_cases(path)
    path.write_text("{ not json }\n", encoding="utf-8")
    with pytest.raises(ValueError):
        jsonl_read(path)


def test_cli_offline_demo_scan(tmp_path):
    assert main(["demo", "--out", str(tmp_path)]) == 0
    assert main(["scan", "--traces", str(tmp_path / "traces.jsonl"),
                 "--out", str(tmp_path / "diagnosis")]) == 0
    results = json.loads((tmp_path / "diagnosis/diagnosis.json").read_text())
    assert results["loop_count"] == 3
    assert (tmp_path / "diagnosis/report.md").exists()


def test_inspect_config_quant_and_without_weights(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5", "num_hidden_layers": 24,
        "quantization_config": {"bits": 4, "group_size": 64}}))
    metadata = inspect_model_dir(tmp_path)
    assert metadata["layers"] == 24
    assert "NO direct packed-4bit" in metadata["edit_support"]
