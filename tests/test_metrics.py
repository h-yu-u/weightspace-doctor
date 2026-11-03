import pytest

from weightspace_doctor.diagnosis import summarize
from weightspace_doctor.metrics import (assess_trace, correctness, duplicate_ngram_ratio,
                                        is_cut, last_boxed, whitespace_tail)
from weightspace_doctor.schema import Trace


def test_loop_and_token_budget():
    trace = Trace("loop", "question", "A B C D E F G H " * 60, "length", 240, 240)
    finding = assess_trace(trace)
    assert "repetition_loop" in finding.flags
    assert "budget_exhaustion" in finding.flags
    assert finding.repetition_ratio > 0.9
    assert not finding.completed


def test_no_loop_for_short_or_genuine_text():
    assert duplicate_ngram_ratio("The answer is straightforward.") == 0
    assert not whitespace_tail("a\n\n")
    assert whitespace_tail("a" + " " * 129)
    normal = Trace("n", "p", "This task was completed.", "stop", 10, 10)
    assert not is_cut(normal), "Explicit stop beats a full-length count"


def test_nested_box_and_reasoning_content():
    assert last_boxed("first \\boxed{1} then \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert last_boxed("\\boxed{incomplete") is None
    trace = Trace("id", "p", "repeating thoughts </think> \\boxed{42}",
                  expected="42", final_text="\\boxed{42}")
    assert correctness(trace) is True


def test_request_error_never_counted_as_incorrect():
    a = summarize([Trace(id="e", prompt="p", text="", error="HTTP 401")])
    assert a.labeled == 0 and a.request_errors == 1 and a.accuracy is None


def test_regex_and_invalid_regex():
    good = Trace("id", "p", "{\"success\": true}", expected_regex=r"success.*true")
    assert correctness(good) is True
    bad = Trace("bad", "p", "okay", expected_regex="[")
    with pytest.raises(ValueError):
        correctness(bad)
