from pathlib import Path

import numpy as np
import pytest
import torch

from weightspace_doctor.editing import (apply_patch, build_patch, build_random_control,
                                        load_patch, save_patch, temporary_patch,
                                        values_sha256)
from weightspace_doctor.localization import (capture_residuals, contrast_statistics,
                                             localize_model, read_localization,
                                             save_localization)
from weightspace_doctor.schema import ContrastPair


def pairs():
    return [ContrastPair(id=f"pair-{i}", prompt=f"How much is {i}+2?",
                         good="The answer is 5 and I am done.",
                         bad="Wait wait wait wait wait again again again again.")
            for i in range(4)]


def test_contrast_and_writer_localization(tiny_model, tmp_path):
    model, tokenizer = tiny_model
    result = localize_model(model, tokenizer, pairs(), indices=[0, 1],
                            topk=2, tail_tokens=4)
    assert result["pairs"] == 4
    assert len(result["layers"]) == 2
    assert all(e["editable"] for e in result["layers"])
    assert all(len(e["writer_columns"]) == 2 for e in result["layers"])
    assert all(e["score"] > 0 for e in result["layers"])
    save_localization(tmp_path, result)
    loaded = read_localization(tmp_path)
    assert loaded["layers"] == result["layers"]
    for i in [0, 1]:
        np.testing.assert_array_equal(loaded["directions"][f"layer_{i}"],
                                      result["directions"][f"layer_{i}"])


def test_apply_and_revert_exact_float32(tiny_model, tmp_path):
    model, tokenizer = tiny_model
    localization = localize_model(model, tokenizer, pairs(), indices=[0, 1], topk=2)
    patch = build_patch(model, localization, model_id="tiny-model", alpha=0.5,
                        top_layers=1, topk=1,
                        max_changed_fraction=.8, max_relative_change=1.0)
    original = [x.mlp.down_proj.weight.detach().clone() for x in model.model.layers]
    save_patch(tmp_path, patch)
    loaded = load_patch(tmp_path)
    with temporary_patch(model, loaded):
        assert any(not torch.equal(x.mlp.down_proj.weight, old)
                   for x, old in zip(model.model.layers, original))
    assert all(torch.equal(x.mlp.down_proj.weight, old)
               for x, old in zip(model.model.layers, original))
    apply_patch(model, patch)
    apply_patch(model, patch, revert=True)
    assert all(torch.equal(x.mlp.down_proj.weight, old)
               for x, old in zip(model.model.layers, original))


def test_refuse_tampered_or_wrong_base(tiny_model, tmp_path):
    model, tokenizer = tiny_model
    result = localize_model(model, tokenizer, pairs(), indices=[0], topk=2)
    patch = build_patch(model, result, model_id="tiny-model", alpha=.5,
                        topk=1, max_changed_fraction=.8, max_relative_change=1)
    weight = model.model.layers[0].mlp.down_proj.weight
    with torch.no_grad():
        weight[0, 0] += 0.5
    with pytest.raises(ValueError, match="checksum|base tensor"):
        apply_patch(model, patch)
    save_patch(tmp_path, patch)
    manifest = tmp_path / "manifest.json"
    assert Path(manifest).exists()
    with np.load(tmp_path / "values.npz", allow_pickle=False) as orig:
        arr = {k: orig[k].copy() for k in orig.files}
    arr["p0_before"][0, 0] += 1
    np.savez_compressed(tmp_path / "values.npz", **arr)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_patch(tmp_path)


def test_norm_matched_negative_control(tiny_model):
    model, tokenizer = tiny_model
    result = localize_model(model, tokenizer, pairs(), indices=[0], topk=2)
    patch = build_patch(model, result, model_id="tiny-model", alpha=.4,
                        topk=1, max_changed_fraction=.8, max_relative_change=1)
    control = build_random_control(model, patch, seed=1, max_relative_change=1)
    assert control["control"]
    assert control["entries"][0]["columns"] == patch["entries"][0]["columns"]
    e1 = patch["entries"][0]
    e2 = control["entries"][0]
    assert np.isclose(np.linalg.norm(e1["after"] - e1["before"]),
                      np.linalg.norm(e2["after"] - e2["before"]), rtol=1e-4)


def test_captured_hooks_always_removed(tiny_model):
    model, tokenizer = tiny_model
    with pytest.raises(ValueError):
        capture_residuals(model, tokenizer, "abc", [99])
    assert not model.model.layers[0]._forward_hooks


def test_contrast_rejects_degenerate():
    with pytest.raises(ValueError):
        contrast_statistics(np.ones((2, 4)), np.ones((2, 4)))


def test_all_weight_preconditions_checked_before_first_write(tiny_model):
    model, tokenizer = tiny_model
    localization = localize_model(model, tokenizer, pairs(), indices=[0, 1], topk=2)
    patch = build_patch(model, localization, model_id="toy", chosen_layers=[0, 1],
                        alpha=.5, topk=1, max_changed_fraction=.8,
                        max_relative_change=1)
    original_first = model.model.layers[0].mlp.down_proj.weight.detach().clone()
    # Artificial competing edit to the second layer invalidates its fingerprint.
    with torch.no_grad():
        model.model.layers[1].mlp.down_proj.weight[0, 0] += 1
    with pytest.raises(ValueError):
        apply_patch(model, patch)
    assert torch.equal(model.model.layers[0].mlp.down_proj.weight, original_first)


def test_exception_inside_temporary_patch_restores_values(tiny_model):
    model, tokenizer = tiny_model
    loc = localize_model(model, tokenizer, pairs(), indices=[0], topk=2)
    patch = build_patch(model, loc, model_id="toy", alpha=.5, topk=1,
                        max_changed_fraction=.8, max_relative_change=1)
    original = model.model.layers[0].mlp.down_proj.weight.detach().clone()
    with pytest.raises(RuntimeError, match="simulated"):
        with temporary_patch(model, patch):
            raise RuntimeError("simulated")
    assert torch.equal(model.model.layers[0].mlp.down_proj.weight, original)


def test_refuse_quantized_wrapper_with_float_weight(tiny_model):
    from weightspace_doctor.hf import editable_float_linear
    model, tokenizer = tiny_model
    projection = model.model.layers[0].mlp.down_proj
    # A wrapper may look float32 but have quantized per-block metadata.
    projection.quant_state = {"scale": 1.0}
    assert not editable_float_linear(projection)
    result = localize_model(model, tokenizer, pairs(), indices=[0], topk=2)
    assert result["layers"][0]["editable"] is False
    with pytest.raises(ValueError, match="No editable"):
        build_patch(model, result, model_id="fake-quant")


def test_reject_duplicate_target_layers(tiny_model):
    model, tokenizer = tiny_model
    result = localize_model(model, tokenizer, pairs(), indices=[0], topk=2)
    patch = build_patch(model, result, model_id="toy", alpha=.5, topk=1,
                        max_changed_fraction=.8, max_relative_change=1)
    duplicate = {**patch, "entries": patch["entries"] * 2}
    original = model.model.layers[0].mlp.down_proj.weight.detach().clone()
    with pytest.raises(ValueError, match="duplicate target"):
        apply_patch(model, duplicate)
    assert torch.equal(original, model.model.layers[0].mlp.down_proj.weight)


def test_refuse_model_level_quantization_metadata(tiny_model):
    model, tokenizer = tiny_model
    loc = localize_model(model, tokenizer, pairs(), indices=[0], topk=1)
    model.config = type("Config", (), {"quantization_config": {"bits": 4}})()
    with pytest.raises(ValueError, match="quantized"):
        build_patch(model, loc, model_id="toy")
    check = localize_model(model, tokenizer, pairs(), indices=[0], topk=1)
    assert not check["layers"][0]["editable"]


def test_activation_weighted_writer_ranking_prefers_bad_writers():
    from weightspace_doctor.localization import top_writers
    w = torch.tensor([[2., 2., 2., 2.], [0., 0., 0., 0.], [0., 0., 0., 0.]])
    u = np.array([1., 0., 0.], dtype=np.float32)
    # Column 1 has a strong weight but is *less* active in the bad case;
    # it should not be the first target of a bad-direction suppression.
    delta = np.array([1., -10., .5, 0.], dtype=np.float32)
    selected, energy, method = top_writers(w, u, k=2, activation_delta=delta)
    assert selected == [0, 2]
    assert method == "activation_weighted_positive_writer"
    assert 0 < energy < 1
    fallback, _, fallback_method = top_writers(w, u, k=2)
    assert fallback == [0, 1]
    assert fallback_method == "projection_only_no_activations"
