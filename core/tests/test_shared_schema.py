from core.shared_schema import (
    ObservationPayload,
    PriorityProfilePayload,
    observation_vector_from_payload,
    priority_payload_from_json,
)


def test_priority_payload_roundtrip():
    payload = PriorityProfilePayload.from_json(
        {
            "profile_id": "high_fidelity",
            "weights": {
                "two_qubit_gates": 0.5,
                "depth": 0.3,
                "total_gates": 0.1,
                "swap_gates": 0.1,
            },
            "budgets": {"max_depth": 250, "max_latency_ms": 1500, "max_shots": 2000},
            "context": {"queue_level": "high", "noise_level": "high", "backend": "qpu_a"},
        }
    )
    data = payload.to_json()
    assert data["profile_id"] == "high_fidelity"
    assert data["budgets"]["max_depth"] == 250


def test_observation_vector_uses_priority_profile_id():
    payload = ObservationPayload(
        gate_count=10,
        depth=5,
        num_cnot=3,
        num_rz=2,
        constraint_profile="balanced",
        priority_profile_id="high_fidelity",
        last_action_id=4,
    )
    obs = observation_vector_from_payload(payload)
    assert len(obs) == 6
    assert obs[4] == 4.0
    assert obs[5] == 2.0


def test_priority_payload_from_json_default():
    payload = priority_payload_from_json(None)
    assert payload.profile_id == "balanced"
