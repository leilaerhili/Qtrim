"""
core/metrics.py

Circuit metrics and lightweight analysis utilities for the RL-based optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

from qiskit.circuit import QuantumCircuit


PRIORITY_KEYS = ("two_qubit_gates", "depth", "total_gates", "swap_gates")
_PRIORITY_KEY_ALIASES = {
    "two_qubit_gates": "two_qubit_gates",
    "two_qubit": "two_qubit_gates",
    "cx": "two_qubit_gates",
    "cnot": "two_qubit_gates",
    "num_cnot": "two_qubit_gates",
    "cx_count": "two_qubit_gates",
    "w_cx": "two_qubit_gates",
    "depth": "depth",
    "w_depth": "depth",
    "total_gates": "total_gates",
    "gate_count": "total_gates",
    "gates": "total_gates",
    "w_gate_count": "total_gates",
    "swap_gates": "swap_gates",
    "swap": "swap_gates",
    "swaps": "swap_gates",
    "swap_count": "swap_gates",
    "w_swap": "swap_gates",
}

_PROFILE_WEIGHT_PRESETS: Dict[str, Dict[str, float]] = {
    "balanced": {
        "two_qubit_gates": 0.10,
        "depth": 0.30,
        "total_gates": 0.55,
        "swap_gates": 0.05,
    },
    "high_fidelity": {
        "two_qubit_gates": 0.50,
        "depth": 0.20,
        "total_gates": 0.10,
        "swap_gates": 0.20,
    },
    "low_latency": {
        "two_qubit_gates": 0.15,
        "depth": 0.50,
        "total_gates": 0.25,
        "swap_gates": 0.10,
    },
    "low_cost": {
        "two_qubit_gates": 0.10,
        "depth": 0.30,
        "total_gates": 0.50,
        "swap_gates": 0.10,
    },
}

_PROFILE_ALIASES = {
    "default": "balanced",
    "low_noise": "high_fidelity",
    "noise": "high_fidelity",
    "min_cx": "high_fidelity",
    "few_cx": "high_fidelity",
    "cx": "high_fidelity",
    "depth": "low_latency",
}


@dataclass(frozen=True)
class CostWeights:
    """
    Weights used to compute a scalar circuit score.

    The names keep backward compatibility with the old training stack.
    """

    w_gate_count: float = 1.0
    w_depth: float = 0.5
    w_cx: float = 0.0
    w_swap: float = 0.0


@dataclass(frozen=True)
class CircuitMetrics:
    gate_count: int
    depth: int
    gate_histogram: Dict[str, int]
    cx_count: int
    swap_count: int
    cost: float


def canonical_profile_id(profile: str) -> str:
    key = str(profile).strip().lower()
    key = _PROFILE_ALIASES.get(key, key)
    if key in _PROFILE_WEIGHT_PRESETS:
        return key
    return "balanced"


def _coerce_priority_weights(weights: Mapping[str, float]) -> Dict[str, float]:
    parsed = {k: 0.0 for k in PRIORITY_KEYS}
    for raw_key, raw_val in dict(weights).items():
        key = _PRIORITY_KEY_ALIASES.get(str(raw_key).strip().lower())
        if key is None:
            continue
        try:
            val = float(raw_val)
        except Exception:
            continue
        if val < 0.0:
            val = 0.0
        parsed[key] = val
    return parsed


def normalize_priority_weights(weights: Mapping[str, float]) -> Dict[str, float]:
    parsed = _coerce_priority_weights(weights)
    total = float(sum(parsed.values()))
    if total <= 0.0:
        raise ValueError("Priority weights must sum to a positive value.")
    return {k: float(parsed[k] / total) for k in PRIORITY_KEYS}


def resolve_priority_weights(
    profile_id: str,
    override_weights: Optional[Mapping[str, float]] = None,
) -> CostWeights:
    profile = canonical_profile_id(profile_id)
    merged = dict(_PROFILE_WEIGHT_PRESETS[profile])
    if override_weights is not None:
        parsed_override = _coerce_priority_weights(override_weights)
        for k in PRIORITY_KEYS:
            if k in parsed_override and float(parsed_override[k]) > 0.0:
                merged[k] = float(parsed_override[k])
    normalized = normalize_priority_weights(merged)
    return CostWeights(
        w_gate_count=float(normalized["total_gates"]),
        w_depth=float(normalized["depth"]),
        w_cx=float(normalized["two_qubit_gates"]),
        w_swap=float(normalized["swap_gates"]),
    )


def cost_weights_to_priority_dict(weights: CostWeights) -> Dict[str, float]:
    return normalize_priority_weights(
        {
            "total_gates": float(weights.w_gate_count),
            "depth": float(weights.w_depth),
            "two_qubit_gates": float(weights.w_cx),
            "swap_gates": float(weights.w_swap),
        }
    )


def gate_count(circ: QuantumCircuit) -> int:
    return len(circ.data)


def depth(circ: QuantumCircuit) -> int:
    try:
        return int(circ.depth())
    except Exception:
        return 0


def gate_histogram(circ: QuantumCircuit) -> Dict[str, int]:
    hist: Dict[str, int] = {}
    for ci in circ.data:
        name = ci.operation.name
        hist[name] = hist.get(name, 0) + 1
    return hist


def cx_count(circ: QuantumCircuit) -> int:
    return int(gate_histogram(circ).get("cx", 0))


def swap_count(circ: QuantumCircuit) -> int:
    return int(gate_histogram(circ).get("swap", 0))


def compute_cost(
    circ: QuantumCircuit,
    weights: CostWeights = CostWeights(),
) -> float:
    gc = gate_count(circ)
    d = depth(circ)
    cx = cx_count(circ)
    swaps = swap_count(circ)
    return float(
        weights.w_gate_count * float(gc)
        + weights.w_depth * float(d)
        + weights.w_cx * float(cx)
        + weights.w_swap * float(swaps)
    )


def compute_metrics(
    circ: QuantumCircuit,
    weights: CostWeights = CostWeights(),
) -> CircuitMetrics:
    hist = gate_histogram(circ)
    gc = len(circ.data)
    d = depth(circ)
    cx = int(hist.get("cx", 0))
    swaps = int(hist.get("swap", 0))
    c = compute_cost(circ, weights=weights)
    return CircuitMetrics(
        gate_count=gc,
        depth=d,
        gate_histogram=hist,
        cx_count=cx,
        swap_count=swaps,
        cost=c,
    )


def observation_vector(
    circ: QuantumCircuit,
    last_action_id: int = 0,
    constraint_id: int = 0,
) -> Tuple[float, ...]:
    hist = gate_histogram(circ)
    gc = float(len(circ.data))
    d = float(depth(circ))
    cx = float(hist.get("cx", 0))
    rz = float(hist.get("rz", 0))
    return (gc, d, cx, rz, float(last_action_id), float(constraint_id))


def unitary_distance(
    a: QuantumCircuit,
    b: QuantumCircuit,
    max_qubits: int = 3,
) -> Optional[float]:
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


def weights_for_profile(profile: str) -> CostWeights:
    return resolve_priority_weights(profile_id=profile, override_weights=None)


def _quick_demo() -> None:
    from qiskit.circuit import QuantumRegister

    qr = QuantumRegister(3, "q")
    qc = QuantumCircuit(qr)
    qc.h(qr[0])
    qc.cx(qr[0], qr[1])
    qc.swap(qr[1], qr[2])
    qc.rz(0.3, qr[2])
    qc.cx(qr[1], qr[2])

    m = compute_metrics(qc, weights=weights_for_profile("high_fidelity"))
    print("Gate count:", m.gate_count)
    print("Depth:", m.depth)
    print("CX count:", m.cx_count)
    print("Swap count:", m.swap_count)
    print("Histogram:", m.gate_histogram)
    print("Cost:", m.cost)
    print("Obs:", observation_vector(qc, last_action_id=2, constraint_id=1))


if __name__ == "__main__":
    _quick_demo()
