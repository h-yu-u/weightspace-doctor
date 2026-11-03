"""Agent can *refuse* irrelevant interventions before touching any weight."""

from weightspace_doctor.agent import repair
from weightspace_doctor.schema import Case, ContrastPair, Trace


def test_agent_abstains_without_behavioral_loop(monkeypatch, tiny_model, tmp_path):
    class FakeHFBackend:
        def __init__(self, model, tokenizer, thinking=True):
            pass

        def generate(self, case, *, max_new_tokens, temperature, seed):
            return Trace(id=case.id, prompt=case.prompt, text=case.expected or "OK",
                         expected=case.expected, split=case.split,
                         output_tokens=4, max_output_tokens=max_new_tokens,
                         finish_reason="eos")

    monkeypatch.setattr("weightspace_doctor.hf.HFBackend", FakeHFBackend)
    pairs = [ContrastPair(f"discover{i}", f"question{i}", "good", "bad")
             for i in range(4)]
    tune = [Case("tune1", "what?", expected="42")]
    holdout = [Case("holdout1", "what?", expected="42")]
    model, tokenizer = tiny_model
    state = repair(model, tokenizer, pairs, tune, holdout, model_id="toy",
                   out=tmp_path)
    assert state["status"] == "abstained_no_loop_signal"
    assert "localizing" not in state["states"]
    assert (tmp_path / "run.json").exists()
