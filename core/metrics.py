"""
core/metrics.py

Circuit metrics and lightweight analysis utilities for the RL-based optimizer.

This module intentionally focuses on fast, explainable metrics suitable for a
hackathon demo and for reinforcement learning rewards.

Primary metrics
- gate_count: total number of quantum operations in the circuit
- depth: circuit depth as reported by Qiskit
- gate_histogram: counts of gate names
- cost: a weighted scalar objective used for optimization / RL reward

Optional (small-qubit only) correctness proxy
- unitary_distance: compare unitaries between two circuits (2–3 qubits recommended)
  This is expensive; avoid using it inside RL training loops.

Compatibility
- Uses Qiskit's CircuitInstruction API (ci.operation, ci.qubits, ci.clbits),
  avoiding deprecated tuple-unpacking of circ.data.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import math

from qiskit.circuit import QuantumCircuit


@dataclass(frozen=True)
class CostWeights:
    """
    Weights used to compute a scalar cost.

    Default emphasizes total operations but still accounts for depth.
    You can increase w_cx when you want to penalize two-qubit gates.
    """
    w_gate_count: float = 1.0
    w_depth: float = 0.5
    w_cx: float = 0.0


@dataclass(frozen=True)
class CircuitMetrics:
    """Bundle of commonly used circuit metrics."""
    gate_count: int
    depth: int
    gate_histogram: Dict[str, int]
    cx_count: int
    cost: float


def gate_count(circ: QuantumCircuit) -> int:
    """Return total number of operations in the circuit."""
    return len(circ.data)


def depth(circ: QuantumCircuit) -> int:
    """
    Return circuit depth.

    Qiskit's depth computes the longest dependency chain assuming gates on
    disjoint qubits can be parallelized.
    """
    try:
        return int(circ.depth())
    except Exception:
        # Keep robustness if a circuit is malformed.
        return 0


def gate_histogram(circ: QuantumCircuit) -> Dict[str, int]:
    """
    Count operations by instruction name.

    Notes:
    - Includes all operations in circ.data (measurements included if present).
    """
    hist: Dict[str, int] = {}
    for ci in circ.data:
        name = ci.operation.name
        hist[name] = hist.get(name, 0) + 1
    return hist


def cx_count(circ: QuantumCircuit) -> int:
    """Return number of CX (CNOT) gates in the circuit."""
    cnt = 0
    for ci in circ.data:
        if ci.operation.name == "cx":
            cnt += 1
    return cnt


def compute_cost(
    circ: QuantumCircuit,
    weights: CostWeights = CostWeights(),
) -> float:
    """
    Compute a scalar cost for optimization.

    Cost is linear by default:
        cost = w_gate_count*gate_count + w_depth*depth + w_cx*cx_count

    You can vary weights for different constraint profiles:
    - low_noise: increase w_depth and/or w_cx
    - low_latency: increase w_depth
    - low_gate_budget: increase w_gate_count
    """
    gc = gate_count(circ)
    d = depth(circ)
    cx = cx_count(circ)

    cost_val = (
        weights.w_gate_count * float(gc)
        + weights.w_depth * float(d)
        + weights.w_cx * float(cx)
    )
    return float(cost_val)


def compute_metrics(
    circ: QuantumCircuit,
    weights: CostWeights = CostWeights(),
) -> CircuitMetrics:
    """Compute a standard metrics bundle."""
    hist = gate_histogram(circ)
    gc = len(circ.data)
    d = depth(circ)
    cx = hist.get("cx", 0)
    c = compute_cost(circ, weights=weights)

    return CircuitMetrics(
        gate_count=gc,
        depth=d,
        gate_histogram=hist,
        cx_count=cx,
        cost=c,
    )


# -----------------------------
# Observation vector utilities
# -----------------------------

def observation_vector(
    circ: QuantumCircuit,
    last_action_id: int = 0,
    constraint_id: int = 0,
) -> Tuple[float, ...]:
    """
    Produce a compact numeric observation vector for RL.

    Returns:
        (gate_count, depth, cx_count, rz_count, last_action_id, constraint_id)

    Notes:
    - Keep ids as floats to match common RL frameworks expecting float arrays.
    """
    hist = gate_histogram(circ)
    gc = float(len(circ.data))
    d = float(depth(circ))
    cx = float(hist.get("cx", 0))
    rz = float(hist.get("rz", 0))
    return (gc, d, cx, rz, float(last_action_id), float(constraint_id))


# -----------------------------
# Optional correctness proxy
# -----------------------------

def unitary_distance(
    a: QuantumCircuit,
    b: QuantumCircuit,
    max_qubits: int = 3,
) -> Optional[float]:
    """
    Compute a simple unitary distance between two circuits.

    Practical only for very small circuits (<= max_qubits). Use as a post-hoc
    demo check, not during RL training.

    Returns:
        A non-negative float distance, or None if not computed.

    Distance:
        ||U_a - U_b||_F / ||U_a||_F
    """
    if a.num_qubits != b.num_qubits:
        raise ValueError(f"Qubit mismatch: {a.num_qubits} vs {b.num_qubits}")

    if a.num_qubits > max_qubits:
        return None

    from qiskit.quantum_info import Operator
    import numpy as np

    Ua = Operator(a).data
    Ub = Operator(b).data

    diff = Ua - Ub
    num = float(np.linalg.norm(diff, ord="fro"))
    den = float(np.linalg.norm(Ua, ord="fro"))
    if den == 0.0:
        return None
    return num / den


# -----------------------------
# Constraint profiles (optional)
# -----------------------------

def weights_for_profile(profile: str) -> CostWeights:
    """
    Map a human-readable constraint profile to cost weights.

    Example profiles:
    - balanced
    - low_noise
    - low_latency
    - min_cx
    """
    p = profile.strip().lower()
    if p in ("balanced", "default"):
        return CostWeights(w_gate_count=1.0, w_depth=0.5, w_cx=0.0)
    if p in ("low_latency", "depth"):
        return CostWeights(w_gate_count=0.8, w_depth=1.2, w_cx=0.0)
    if p in ("low_noise", "noise"):
        return CostWeights(w_gate_count=0.6, w_depth=1.0, w_cx=0.7)
    if p in ("min_cx", "few_cx", "cx"):
        return CostWeights(w_gate_count=0.7, w_depth=0.4, w_cx=1.2)

    return CostWeights(w_gate_count=1.0, w_depth=0.5, w_cx=0.0)


# -----------------------------
# Self-test / quick demo
# -----------------------------

def _quick_demo() -> None:
    """
    Quick manual sanity test:
        python -m core.metrics
    """
    from qiskit.circuit import QuantumRegister

    qr = QuantumRegister(3, "q")
    qc = QuantumCircuit(qr)
    qc.h(qr[0])
    qc.cx(qr[0], qr[1])
    qc.rz(0.3, qr[2])
    qc.cx(qr[1], qr[2])

    m = compute_metrics(qc, weights=weights_for_profile("low_noise"))
    print("Gate count:", m.gate_count)
    print("Depth:", m.depth)
    print("CX count:", m.cx_count)
    print("Histogram:", m.gate_histogram)
    print("Cost:", m.cost)
    print("Obs:", observation_vector(qc, last_action_id=2, constraint_id=1))


if __name__ == "__main__":
    _quick_demo()
