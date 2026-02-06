"""
Predefined baseline circuits + padding utilities.

These are intentionally small, deterministic circuits that contain
rewrite opportunities for the RL environment to discover quickly.
"""

from __future__ import annotations

from typing import Callable, Dict

from qiskit.circuit import QuantumCircuit, QuantumRegister


def _pad_rz_chain(qc: QuantumCircuit, qubit, pad_level: int) -> None:
    """
    Add predictable RZ padding to create merge/cancel opportunities.
    """
    for _ in range(max(0, pad_level - 1)):
        qc.rz(0.1, qubit)
        qc.rz(0.2, qubit)


def build_toy_circuit(pad_level: int = 1) -> QuantumCircuit:
    """
    3-qubit toy circuit with obvious local rewrites.
    """
    qr = QuantumRegister(3, "q")
    qc = QuantumCircuit(qr, name="toy")
    qc.cx(qr[0], qr[1])
    qc.cx(qr[0], qr[1])  # cancellable
    qc.rz(0.7, qr[2])
    qc.rz(-0.7, qr[2])   # cancellable
    _pad_rz_chain(qc, qr[2], pad_level)
    return qc


def build_ghz_circuit(pad_level: int = 1) -> QuantumCircuit:
    """
    GHZ-like circuit with extra local noise for rewrites.
    """
    qr = QuantumRegister(3, "q")
    qc = QuantumCircuit(qr, name="ghz")
    qc.h(qr[0])
    qc.cx(qr[0], qr[1])
    qc.cx(qr[1], qr[2])
    qc.rz(0.25, qr[1])
    qc.rz(-0.25, qr[1])  # cancellable
    _pad_rz_chain(qc, qr[1], pad_level)
    return qc


def build_line_circuit(pad_level: int = 1) -> QuantumCircuit:
    """
    Linear entangling circuit with mergeable rotations.
    """
    qr = QuantumRegister(4, "q")
    qc = QuantumCircuit(qr, name="line")
    qc.cx(qr[0], qr[1])
    qc.cx(qr[1], qr[2])
    qc.cx(qr[2], qr[3])
    qc.rz(0.2, qr[2])
    qc.rz(0.3, qr[2])  # mergeable
    _pad_rz_chain(qc, qr[2], pad_level)
    return qc


BASELINE_BUILDERS: Dict[str, Callable[[int], QuantumCircuit]] = {
    "toy": build_toy_circuit,
    "ghz": build_ghz_circuit,
    "line": build_line_circuit,
}


def get_builder(name: str) -> Callable[[int], QuantumCircuit]:
    """
    Get a baseline circuit builder by name.
    """
    key = name.strip().lower()
    if key not in BASELINE_BUILDERS:
        raise KeyError(f"Unknown baseline circuit '{name}'. Options: {sorted(BASELINE_BUILDERS)}")
    return BASELINE_BUILDERS[key]
