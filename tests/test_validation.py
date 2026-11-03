import pytest

from weightspace_doctor.schema import Case, Trace
from weightspace_doctor.validation import (compare_traces, evaluate_patch,
                                            paired_sign_test_pvalue)


def tr(i, text, *, split="target", expected="42", reason="stop"):
    return Trace(id=i, prompt="math", text=text, split=split,
                 expected=expected, finish_reason=reason, output_tokens=16,
                 max_output_tokens=100)


def test_gate_rescues_without_regressing_guard():
    base = [tr(f"t{i}", "bad") for i in range(20)] + [tr(f"g{i}", "OK", split="guard", expected="OK") for i in range(10)]
    edited = [tr(f"t{i}", "42") for i in range(20)] + [tr(f"g{i}", "OK", split="guard", expected="OK") for i in range(10)]
    result = compare_traces(base, edited)
    assert result["gate"]["status"] == "recommend_manual_review"
    assert result["target"]["rescues"] == 20
    assert result["target_accuracy_gain"] == 1


def test_no_labels_or_insufficient_cases_block():
    base = [tr("a", "bad"), tr("g", "OK", split="guard", expected=None)]
    changed = [tr("a", "42"), tr("g", "OK", split="guard", expected=None)]
    result = compare_traces(base, changed)
    assert result["gate"]["status"] == "not_approved"
    assert "insufficient_validation_cases" in result["gate"]["reasons"]
    assert "missing_ground_truth_labels" in result["gate"]["reasons"]


def test_mismatched_paired_ids_rejected():
    with pytest.raises(ValueError):
        compare_traces([tr("a", "42")], [tr("b", "42")])


def test_refuse_discovery_leakage():
    class Dummy:
        pass
    with pytest.raises(ValueError, match="Leakage"):
        evaluate_patch(Dummy(), Dummy(), {"discovery_ids": ["a"]},
                       [Case(id="a", prompt="p")])


def test_exact_discordant_sign_test():
    assert paired_sign_test_pvalue(20, 0) == pytest.approx(2 ** -20)
    assert paired_sign_test_pvalue(1, 0) == pytest.approx(.5)
    assert paired_sign_test_pvalue(0, 0) == 1
    assert paired_sign_test_pvalue(2, 2) == pytest.approx(11 / 16)


def test_small_positive_gain_not_enough_to_approve():
    base = [tr(f"t{i}", "bad") for i in range(20)] + [
        tr(f"g{i}", "OK", split="guard", expected="OK") for i in range(10)]
    edited = [tr("t0", "42")] + [tr(f"t{i}", "bad") for i in range(1, 20)] + [
        tr(f"g{i}", "OK", split="guard", expected="OK") for i in range(10)]
    result = compare_traces(base, edited)
    assert result["gate"]["status"] == "not_approved"
    assert "target_improvement_not_statistically_supported" in result["gate"]["reasons"]
