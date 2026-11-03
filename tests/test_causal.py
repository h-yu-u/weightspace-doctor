from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from weightspace_doctor.causal import causal_evidence, score_continuation, isolate_patch
from weightspace_doctor.editing import build_patch
from weightspace_doctor.localization import localize_model
from weightspace_doctor.schema import ContrastPair


class CharacterTokenizer:
    chat_template = None

    def __call__(self, text, *, add_special_tokens=False, return_tensors=None):
        ids = [ord(c) % 64 for c in text]
        return {"input_ids": [ids] if return_tensors is None else torch.tensor([ids])}


class LogitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(64, 4)
        self.model = nn.Module()
        from conftest import TinyBlock
        self.model.layers = nn.ModuleList([TinyBlock()])
        self.head = nn.Linear(4, 64, bias=False)

    def get_input_embeddings(self):
        return self.emb

    def forward(self, input_ids, use_cache=False, logits_to_keep=0):
        hidden = self.model.layers[0](self.emb(input_ids))
        logits = self.head(hidden)
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:, :]
        return SimpleNamespace(logits=logits)


def test_teacher_forced_logit_scaling():
    torch.manual_seed(22)
    model, tokenizer = LogitModel(), CharacterTokenizer()
    actual = score_continuation(model, tokenizer, "test", "abc", tail_tokens=2)
    ids = tokenizer("testabc", return_tensors="pt")["input_ids"]
    with torch.inference_mode():
        logits = model(ids).logits
        expected = torch.log_softmax(logits[0, -3:-1, :], dim=-1)
        expected = expected.gather(1, ids[0, -2:, None]).mean().item()
    assert actual == pytest.approx(expected)


def test_causal_patch_restores_base(tiny_model):
    # The activation probe uses a toy forward returning residuals; causal scorer
    # needs vocabulary logits, so add a head to the very same weight-bearing model.
    model, tokenizer = tiny_model
    pairs = [ContrastPair(str(i), f"problem {i}", " good ending", " bad repeating")
             for i in range(4)]
    localized = localize_model(model, tokenizer, pairs, indices=[0], topk=2)
    patch = build_patch(model, localized, model_id="tiny", alpha=.3, topk=1,
                        max_relative_change=1, max_changed_fraction=.8)
    patch = isolate_patch(patch, [0])
    assert len(patch["entries"]) == 1
    with pytest.raises(ValueError):
        isolate_patch(patch, [9])
    from weightspace_doctor.editing import temporary_patch
    original = model.model.layers[0].mlp.down_proj.weight.detach().clone()
    with temporary_patch(model, patch):
        assert not torch.equal(original, model.model.layers[0].mlp.down_proj.weight)
    assert torch.equal(original, model.model.layers[0].mlp.down_proj.weight)


def test_measured_causal_deltas_on_toy_logits():
    # This exercises actual forward-logit scoring, not just weight rollback.
    from weightspace_doctor.editing import build_random_control
    torch.manual_seed(29)
    model, tokenizer = LogitModel(), CharacterTokenizer()
    pairs = [ContrastPair(f"p{i}", f"Compute {i+3}+7.",
                          " I can conclude the sum.",
                          " wait I should check wait I should check")
             for i in range(4)]
    located = localize_model(model, tokenizer, pairs, indices=[0], topk=1)
    patch = build_patch(model, located, model_id="toy-logits", alpha=.3, topk=1,
                        max_changed_fraction=.9, max_relative_change=1)
    control = build_random_control(model, patch, max_relative_change=1)
    original = model.model.layers[0].mlp.down_proj.weight.detach().clone()
    evidence = causal_evidence(model, tokenizer, pairs, patch, tail_tokens=4)
    negative = causal_evidence(model, tokenizer, pairs, control, tail_tokens=4)
    assert evidence["n"] == 4 and negative["n"] == 4
    assert all("bad_delta" in row for row in evidence["per_pair"])
    assert np.isfinite(evidence["selectivity"])
    assert torch.equal(model.model.layers[0].mlp.down_proj.weight, original)
