from qiskit.circuit import QuantumCircuit

from pc import api_server


def _fake_android_infer(**kwargs):
    _ = kwargs
    return 0


def test_optimize_returns_real_circuit_payload(monkeypatch):
    monkeypatch.setattr(api_server, "infer_action_with_android", _fake_android_infer)
    req = api_server.OptimizeRequest(
        circuit_id="half_adder",
        profile_id="balanced",
        max_steps=5,
        pad_level=1,
    )
    result = api_server.optimize(req)
    assert result["before"]["gate_count"] >= 1
    assert result["after"]["gate_count"] >= 1
    assert isinstance(result["before_qasm"], str) and result["before_qasm"].strip()
    assert isinstance(result["after_qasm"], str) and result["after_qasm"].strip()
    assert result["meta"]["policy_source"] == "android_inference_handoff"
    assert result["meta"]["steps_executed"] >= 1


def test_optimize_accepts_legacy_streamlit_circuit_ids(monkeypatch):
    monkeypatch.setattr(api_server, "infer_action_with_android", _fake_android_infer)
    req = api_server.OptimizeRequest(
        circuit_id="majority_vote",
        profile_id="high_fidelity",
        max_steps=4,
        pad_level=1,
    )
    result = api_server.optimize(req)
    assert result["resolved_circuit_id"] == "majority"
    assert result["profile_id"] == "high_fidelity"


def test_optimize_circuit_object_returns_quantum_circuit_instances(monkeypatch):
    monkeypatch.setattr(api_server, "infer_action_with_android", _fake_android_infer)
    req = api_server.OptimizeRequest(
        circuit_id="linear_dataflow_pipeline",
        profile_id="balanced",
        max_steps=3,
        pad_level=1,
    )
    artifacts = api_server.optimize_circuit_object(req)
    assert isinstance(artifacts.before_circuit, QuantumCircuit)
    assert isinstance(artifacts.after_circuit, QuantumCircuit)
    assert artifacts.resolved_circuit_id == "line"
