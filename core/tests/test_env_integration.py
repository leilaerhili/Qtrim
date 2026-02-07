import numpy as np
from qiskit.circuit import QuantumRegister, QuantumCircuit

from core.env_quantum_opt import QuantumOptEnv, EnvConfig
from core.metrics import compute_cost, weights_for_profile
from core.rewrites import (
    ACTION_CANCEL_DOUBLE_CX,
    ACTION_CANCEL_INVERSE_RZ,
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
