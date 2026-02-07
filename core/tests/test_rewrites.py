from qiskit.circuit import QuantumRegister, QuantumCircuit
from qiskit.quantum_info import Operator

from core.rewrites import (
    apply_action,
    ACTION_CANCEL_DOUBLE_CX,
    ACTION_CANCEL_INVERSE_RZ,
    ACTION_MERGE_ADJACENT_RZ,
    ACTION_REMOVE_IDENTITY_RZ,
)

def unitary_equal(a: QuantumCircuit, b: QuantumCircuit, tol=1e-8) -> bool:
    # Only safe for small circuits.
    Ua = Operator(a).data
    Ub = Operator(b).data
    import numpy as np
    return float(np.linalg.norm(Ua - Ub, ord="fro")) <= tol

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
