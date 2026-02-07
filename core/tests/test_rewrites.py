from qiskit.circuit import QuantumRegister, QuantumCircuit
from qiskit.quantum_info import Operator

from core.rewrites import (
    apply_action,
    ACTION_CANCEL_DOUBLE_CX,
    ACTION_CANCEL_INVERSE_RZ,
    ACTION_MERGE_ADJACENT_RZ,
    ACTION_REMOVE_IDENTITY_RZ,
    ACTION_RESYNTH_2Q_WINDOW_7,
)

def unitary_equal(a: QuantumCircuit, b: QuantumCircuit, tol=1e-8) -> bool:
    # Only safe for small circuits.
    Ua = Operator(a).data
    Ub = Operator(b).data
    import numpy as np
    return float(np.linalg.norm(Ua - Ub, ord="fro")) <= tol


def unitary_equal_up_to_global_phase(a: QuantumCircuit, b: QuantumCircuit, tol=1e-8) -> bool:
    Ua = Operator(a).data
    Ub = Operator(b).data
    import numpy as np

    idx = None
    for i in range(Ua.size):
        if abs(Ua.flat[i]) > tol or abs(Ub.flat[i]) > tol:
            idx = i
            break
    if idx is None:
        return True
    if abs(Ub.flat[idx]) <= tol:
        return False
    phase = Ua.flat[idx] / Ub.flat[idx]
    return bool(np.allclose(Ua, phase * Ub, atol=tol))

def test_cancel_double_cx():
    qr = QuantumRegister(2, "q")
    qc = QuantumCircuit(qr)
    qc.cx(qr[0], qr[1])
    qc.cx(qr[0], qr[1])  # cancels
    new_qc, res = apply_action(qc, ACTION_CANCEL_DOUBLE_CX)
    assert res.changed is True
    assert len(new_qc.data) == 0
    assert unitary_equal(qc, new_qc)

def test_cancel_inverse_rz():
    qr = QuantumRegister(1, "q")
    qc = QuantumCircuit(qr)
    qc.rz(0.7, qr[0])
    qc.rz(-0.7, qr[0])  # cancels
    new_qc, res = apply_action(qc, ACTION_CANCEL_INVERSE_RZ)
    assert res.changed is True
    assert len(new_qc.data) == 0
    assert unitary_equal(qc, new_qc)

def test_merge_adjacent_rz_preserves_unitary():
    qr = QuantumRegister(1, "q")
    qc = QuantumCircuit(qr)
    qc.rz(0.2, qr[0])
    qc.rz(0.3, qr[0])  # merge -> 0.5
    new_qc, res = apply_action(qc, ACTION_MERGE_ADJACENT_RZ)
    assert res.changed is True
    assert len(new_qc.data) == 1
    assert unitary_equal(qc, new_qc)

def test_remove_identity_rz():
    qr = QuantumRegister(1, "q")
    qc = QuantumCircuit(qr)
    qc.rz(0.0, qr[0])
    new_qc, res = apply_action(qc, ACTION_REMOVE_IDENTITY_RZ)
    assert res.changed is True
    assert len(new_qc.data) == 0
    assert unitary_equal(qc, new_qc)

def test_resynth_2q_window_preserves_multi_gate_replacement():
    qr = QuantumRegister(2, "q")
    qc = QuantumCircuit(qr)
    # This 3-gate window has a known 2-gate resynthesis candidate.
    qc.cx(qr[0], qr[1])
    qc.swap(qr[0], qr[1])
    qc.iswap(qr[0], qr[1])

    new_qc, res = apply_action(qc, ACTION_RESYNTH_2Q_WINDOW_7)
    assert res.changed is True
    assert len(new_qc.data) == 2
    assert unitary_equal_up_to_global_phase(qc, new_qc)
