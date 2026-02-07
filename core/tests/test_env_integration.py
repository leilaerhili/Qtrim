import numpy as np
from qiskit.circuit import QuantumRegister, QuantumCircuit

from core.env_quantum_opt import QuantumOptEnv, EnvConfig
from core.metrics import compute_cost, weights_for_profile
from core.rewrites import (
    ACTION_CANCEL_DOUBLE_CX,
    ACTION_CANCEL_INVERSE_RZ,
    ACTION_ROUTE_CX_WITH_SWAP,
)

def builder_with_known_easy_improvement(pad_level: int) -> QuantumCircuit:
    qr = QuantumRegister(3, "q")
    qc = QuantumCircuit(qr)
    # Guaranteed improvements:
    qc.cx(qr[0], qr[1])
    qc.cx(qr[0], qr[1])        # cancellable
    qc.rz(0.4, qr[2])
    qc.rz(-0.4, qr[2])         # cancellable
    # Add a bit of padding if requested
    for _ in range(max(0, pad_level - 1)):
        qc.rz(0.1, qr[2])
        qc.rz(0.2, qr[2])
    return qc

def builder_with_nonlocal_cx(_: int) -> QuantumCircuit:
    qr = QuantumRegister(3, "q")
    qc = QuantumCircuit(qr)
    # With default line coupling 0-1-2, this CX requires routing.
    qc.cx(qr[0], qr[2])
    return qc

def test_env_reset_and_obs_shape():
    env = QuantumOptEnv(
        circuit_builder=builder_with_known_easy_improvement,
        pad_level=2,
        config=EnvConfig(max_steps=10),
    )
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (6,)
    assert env.observation_space.contains(obs)

def test_env_step_reward_positive_on_improvement():
    env = QuantumOptEnv(
        circuit_builder=builder_with_known_easy_improvement,
        pad_level=1,
        config=EnvConfig(max_steps=10, stall_patience=10, reward_noop=-0.05),
    )
    env.reset()
    # Apply known good action: cancel_double_cx
    obs, reward, terminated, truncated, info = env.step(ACTION_CANCEL_DOUBLE_CX)
    assert reward > 0  # cost should decrease
    assert "metrics" in info
    assert not truncated

def test_env_cost_decreases_after_two_known_actions():
    env = QuantumOptEnv(
        circuit_builder=builder_with_known_easy_improvement,
        pad_level=1,
        config=EnvConfig(max_steps=10, stall_patience=10),
    )
    env.reset()
    w = weights_for_profile("balanced")
    baseline_cost = compute_cost(env.get_circuit(), weights=w)

    env.step(ACTION_CANCEL_DOUBLE_CX)
    env.step(ACTION_CANCEL_INVERSE_RZ)

    final_cost = compute_cost(env.get_circuit(), weights=w)
    assert final_cost < baseline_cost

def test_env_supports_few_cx_profile_alias():
    env = QuantumOptEnv(
        circuit_builder=builder_with_known_easy_improvement,
        pad_level=1,
        config=EnvConfig(constraint_profile="few_cx"),
    )
    obs, _ = env.reset()
    assert obs[5] == 3.0

def test_env_default_coupling_enables_routing_action():
    env = QuantumOptEnv(
        circuit_builder=builder_with_nonlocal_cx,
        pad_level=1,
        config=EnvConfig(max_steps=5),
    )
    env.reset()
    _, _, _, _, info = env.step(ACTION_ROUTE_CX_WITH_SWAP)
    assert info["last_action"]["changed"] is True


def test_env_action_mask_marks_applicable_actions():
    env = QuantumOptEnv(
        circuit_builder=builder_with_known_easy_improvement,
        pad_level=1,
        config=EnvConfig(max_steps=6),
    )
    _, info = env.reset()
    mask = info["action_mask"]
    assert mask[ACTION_CANCEL_DOUBLE_CX] == 1
    assert mask[ACTION_CANCEL_INVERSE_RZ] == 1


def test_env_repeat_noop_penalty_applies():
    env = QuantumOptEnv(
        circuit_builder=builder_with_known_easy_improvement,
        pad_level=1,
        config=EnvConfig(
            max_steps=6,
            reward_noop=-0.1,
            reward_inapplicable=-0.2,
            reward_repeat_action=-0.03,
            reward_repeat_noop=-0.1,
        ),
    )
    env.reset()
    # First call should improve and remove the adjacent CX pair.
    _, first_reward, _, _, _ = env.step(ACTION_CANCEL_DOUBLE_CX)
    # Second repeated call becomes inapplicable and should incur strong penalty.
    _, second_reward, _, _, _ = env.step(ACTION_CANCEL_DOUBLE_CX)

    assert first_reward > 0.0
    assert second_reward <= -0.4


def test_env_reports_priority_profile_details():
    env = QuantumOptEnv(
        circuit_builder=builder_with_known_easy_improvement,
        pad_level=1,
        config=EnvConfig(
            priority_profile_id="high_fidelity",
            context_queue_level="high",
            context_noise_level="high",
        ),
    )
    _, info = env.reset()
    assert info["priority_profile_id"] == "high_fidelity"
    assert "priority_weights" in info
    assert info["context"]["queue_level"] == "high"
    assert info["context"]["noise_level"] == "high"


def test_env_applies_budget_penalty_when_depth_budget_exceeded():
    env = QuantumOptEnv(
        circuit_builder=builder_with_known_easy_improvement,
        pad_level=1,
        config=EnvConfig(
            max_steps=6,
            max_depth_budget=1,
            budget_penalty_scale=2.0,
        ),
    )
    env.reset()
    _, _, _, _, info = env.step(ACTION_CANCEL_DOUBLE_CX)
    assert info["budget_penalty"]["active"] is True
    assert info["budget_penalty"]["total"] > 0.0
