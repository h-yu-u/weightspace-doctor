import torch

from weightspace_doctor.weight_audit import audit_weights


def test_audit_health_and_nonfinite(tiny_model):
    model, _ = tiny_model
    health = audit_weights(model, [0])
    assert health["layers"][0]["status"] == "ok"
    weight = model.model.layers[0].mlp.down_proj.weight
    with torch.no_grad():
        weight[0, 0] = float("nan")
    bad = audit_weights(model, [0])
    assert bad["layers"][0]["status"] == "nonfinite_weights"
    assert bad["layers"][0]["nonfinite_count"] == 1
