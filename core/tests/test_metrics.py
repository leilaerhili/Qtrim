import math
from qiskit.circuit import QuantumRegister, QuantumCircuit

from core.metrics import (
    gate_count, depth, gate_histogram, cx_count,
    compute_cost, weights_for_profile, observation_vector
)

def make_simple_circuit():
    qr = QuantumRegister(3, "q")
    qc = QuantumCircuit(qr)
    qc.h(qr[0])
    qc.cx(qr[0], qr[1])
    qc.rz(0.3, qr[2])
    qc.cx(qr[1], qr[2])
    return qc

def test_basic_counts_and_histogram():
    qc = make_simple_circuit()
    assert gate_count(qc) == len(qc.data)
    assert depth(qc) >= 1
    hist = gate_histogram(qc)
    assert hist.get("cx", 0) == 2
    assert cx_count(qc) == 2

def test_cost_profiles_change_cost():
    qc = make_simple_circuit()
    c_bal = compute_cost(qc, weights=weights_for_profile("balanced"))
    c_noise = compute_cost(qc, weights=weights_for_profile("low_noise"))
    assert c_bal > 0
    assert c_noise > 0
    # Not guaranteed strictly different for every circuit, but usually should be.
    assert c_bal != c_noise

def test_observation_vector_shape_and_types():
    qc = make_simple_circuit()
    obs = observation_vector(qc, last_action_id=2, constraint_id=1)
    assert len(obs) == 6
    assert all(isinstance(x, float) for x in obs)
    assert obs[4] == 2.0
    assert obs[5] == 1.0
