"""
Predefined baseline circuits + padding utilities.

These are intentionally small, deterministic circuits that contain
rewrite opportunities for the RL environment to discover quickly.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Dict

from qiskit.circuit import QuantumCircuit, QuantumRegister


def _pad_rz_chain(qc: QuantumCircuit, qubit, pad_level: int) -> None:
    """
    Add predictable RZ padding to create merge/cancel opportunities.
    """
    for _ in range(max(0, pad_level - 1)):
        qc.rz(0.1, qubit)
        qc.rz(0.2, qubit)


def _pad_mixed_single_qubit(qc: QuantumCircuit, qubit, pad_level: int) -> None:
    """
    Add diverse 1Q padding so training is not overfit to RZ-only patterns.
    """
    repeats = max(0, int(pad_level) - 1)
    for i in range(repeats):
        mode = i % 3
        if mode == 0:
            qc.rz(0.1, qubit)
            qc.rz(0.2, qubit)
        elif mode == 1:
            qc.rx(0.25, qubit)
            qc.rx(-0.25, qubit)
        else:
            qc.ry(0.35, qubit)
            qc.ry(-0.35, qubit)
    if repeats >= 2:
        qc.s(qubit)
        qc.sdg(qubit)
    if repeats >= 3:
        qc.t(qubit)
        qc.tdg(qubit)


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
    _pad_mixed_single_qubit(qc, qr[2], pad_level)
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
    _pad_mixed_single_qubit(qc, qr[1], pad_level)
    return qc

def build_parity_circuit(pad_level: int = 1) -> QuantumCircuit:
    """
    Classical narrative: parity / XOR reduction.

    Given input bits a, b, c, we compute parity into a target wire using CX chains.
    We intentionally include redundant CX pairs and RZ padding to simulate a naive translation
    that benefits from local rewrites.
    """
    qr = QuantumRegister(4, "q")  # q0,q1,q2 are "inputs", q3 is "parity target"
    qc = QuantumCircuit(qr, name="parity")

    # Parity: target ^= a ^ b ^ c
    qc.cx(qr[0], qr[3])
    qc.cx(qr[1], qr[3])
    qc.cx(qr[2], qr[3])

    # Naive/redundant artifacts for rewrites
    qc.cx(qr[1], qr[3])
    qc.cx(qr[1], qr[3])  # cancellable adjacent pair

    qc.rz(0.15, qr[3])
    qc.rz(0.25, qr[3])  # mergeable

    _pad_mixed_single_qubit(qc, qr[3], pad_level)
    return qc


def build_half_adder_circuit(pad_level: int = 1) -> QuantumCircuit:
    """
    Classical narrative: half adder.

    Inputs: a, b
    Outputs: sum = a XOR b, carry = a AND b

    We model a standard reversible construction using:
    - sum on wire s via CX
    - carry on wire c via CCX (Toffoli)

    We intentionally add redundant CX / RZ patterns so the optimizer has visible wins.
    """
    qr = QuantumRegister(4, "q")  # q0=a, q1=b, q2=sum, q3=carry (ancillas start at |0>)
    qc = QuantumCircuit(qr, name="half_adder")

    a, b, s, c = qr[0], qr[1], qr[2], qr[3]

    # sum = a XOR b into s
    qc.cx(a, s)
    qc.cx(b, s)

    # carry = a AND b into c
    qc.ccx(a, b, c)

    # Naive artifacts (optimizer should clean these)
    qc.cx(b, s)
    qc.cx(b, s)  # cancellable pair

    qc.rz(0.4, s)
    qc.rz(-0.4, s)  # cancellable

    _pad_mixed_single_qubit(qc, s, pad_level)
    _pad_mixed_single_qubit(qc, c, pad_level)
    return qc


def build_majority_circuit(pad_level: int = 1) -> QuantumCircuit:
    """
    Classical narrative: majority vote (3-bit majority).

    Inputs: a, b, c
    Output: m = 1 if at least two of (a,b,c) are 1.

    Reversible construction uses Toffolis to compute pairwise ANDs into ancillas
    and combines them into an output wire. Small, deterministic, and still allows
    local rewrite wins via deliberate CX/RZ artifacts.
    """
    qr = QuantumRegister(6, "q")  # a,b,c plus ancillas t0,t1 and output m
    qc = QuantumCircuit(qr, name="majority")

    a, b, c, t0, t1, m = qr[0], qr[1], qr[2], qr[3], qr[4], qr[5]

    # t0 = a & b
    qc.ccx(a, b, t0)
    # t1 = a & c
    qc.ccx(a, c, t1)

    # m = t0 OR t1 OR (b & c)  (simple reversible-ish accumulation)
    qc.cx(t0, m)
    qc.cx(t1, m)
    qc.ccx(b, c, m)

    # Naive artifacts for local rewrites
    qc.rz(0.2, m)
    qc.rz(0.3, m)  # mergeable

    qc.cx(t0, m)
    qc.cx(t0, m)  # cancellable pair

    _pad_mixed_single_qubit(qc, m, pad_level)
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
    _pad_mixed_single_qubit(qc, qr[2], pad_level)
    return qc


def build_conjugation_circuit(pad_level: int = 1) -> QuantumCircuit:
    """
    Patterns for conjugation/toggling rewrites (H-G-H, S-G-Sdg, CX/CZ toggles).
    """
    qr = QuantumRegister(4, "q")
    qc = QuantumCircuit(qr, name="conjugation")
    q0, q1, q2, q3 = qr[0], qr[1], qr[2], qr[3]

    qc.h(q0)
    qc.x(q0)
    qc.h(q0)

    qc.h(q1)
    qc.z(q1)
    qc.h(q1)

    qc.s(q2)
    qc.x(q2)
    qc.sdg(q2)

    qc.s(q3)
    qc.y(q3)
    qc.sdg(q3)

    qc.cx(q0, q2)
    qc.rz(-0.4, q0)
    qc.cx(q0, q2)

    qc.h(q1)
    qc.cx(q0, q1)
    qc.h(q1)

    _pad_mixed_single_qubit(qc, q2, pad_level)
    _pad_mixed_single_qubit(qc, q3, pad_level)
    return qc


def build_identity_cleanup_circuit(pad_level: int = 1) -> QuantumCircuit:
    """
    Identity-heavy circuit that exercises many local cancellation rules.
    """
    qr = QuantumRegister(4, "q")
    qc = QuantumCircuit(qr, name="identity_cleanup")
    q0, q1, q2, q3 = qr[0], qr[1], qr[2], qr[3]

    qc.rz(0.0, q0)
    qc.rx(0.0, q0)
    qc.ry(0.0, q0)

    qc.z(q1)
    qc.z(q1)
    qc.y(q1)
    qc.y(q1)

    qc.s(q2)
    qc.sdg(q2)
    qc.t(q2)
    qc.tdg(q2)

    qc.cz(q2, q3)
    qc.cz(q2, q3)

    qc.cx(q0, q1)
    qc.rz(0.35, q0)
    qc.cx(q0, q1)

    qc.h(q3)
    qc.cx(q2, q3)
    qc.h(q3)

    qc.rx(0.3, q3)
    qc.rx(-0.3, q3)
    qc.ry(0.15, q3)
    qc.ry(0.2, q3)

    _pad_mixed_single_qubit(qc, q3, pad_level)
    return qc


def build_routing_circuit(pad_level: int = 1) -> QuantumCircuit:
    """
    Routing-focused circuit with non-nearest-neighbor 2Q gates.
    """
    from qiskit.circuit.library import ECRGate

    qr = QuantumRegister(5, "q")
    qc = QuantumCircuit(qr, name="routing")
    q0, q1, q2, q3, q4 = qr[0], qr[1], qr[2], qr[3], qr[4]

    qc.cx(q0, q2)
    qc.cz(q1, q3)
    qc.cy(q2, q4)
    qc.append(ECRGate(), [q0, q2])

    # Explicit SWAP-CX-SWAP motif for unroute action.
    qc.swap(q0, q1)
    qc.cx(q2, q0)
    qc.swap(q0, q1)

    _pad_mixed_single_qubit(qc, q2, pad_level)
    return qc


def build_entangler_family_circuit(pad_level: int = 1) -> QuantumCircuit:
    """
    Additional 2Q gate families used by routing-related rewrite actions.
    """
    from qiskit.circuit.library import ECRGate, RXXGate, RYYGate, RZZGate, iSwapGate

    qr = QuantumRegister(5, "q")
    qc = QuantumCircuit(qr, name="entangler_family")
    q0, q1, q2, q3, q4 = qr[0], qr[1], qr[2], qr[3], qr[4]

    qc.append(iSwapGate(), [q0, q2])
    qc.append(ECRGate(), [q1, q3])
    qc.append(RZZGate(0.35), [q0, q2])
    qc.append(RXXGate(0.45), [q1, q3])
    qc.append(RYYGate(-0.25), [q2, q4])
    qc.append(RZZGate(-0.35), [q2, q3])

    qc.rx(0.2, q4)
    qc.rx(-0.2, q4)
    _pad_mixed_single_qubit(qc, q4, pad_level)
    return qc


def make_seeded_challenge_builder(
    seed: int,
    num_qubits: int = 5,
    depth: int = 28,
) -> Callable[[int], QuantumCircuit]:
    """
    Build a deterministic "harder" circuit generator for curriculum training.

    The generated circuit mixes:
    - random single-qubit gates,
    - random two-qubit entangling gates,
    - deliberate rewrite opportunities.
    """

    fixed_seed = int(seed)
    nq = max(3, int(num_qubits))
    base_depth = max(6, int(depth))

    def _builder(pad_level: int = 1) -> QuantumCircuit:
        rng = random.Random(fixed_seed + 1009 * int(pad_level))
        qr = QuantumRegister(nq, "q")
        qc = QuantumCircuit(qr, name=f"challenge_{fixed_seed}")

        steps = base_depth + max(0, int(pad_level) - 1) * 4
        for i in range(steps):
            q = qr[rng.randrange(nq)]
            choice = rng.random()
            if choice < 0.60:
                theta = rng.uniform(-math.pi, math.pi)
                g = rng.choice(("rx", "ry", "rz", "h", "x", "y", "z", "s", "sdg", "t", "tdg"))
                if g == "rx":
                    qc.rx(theta, q)
                elif g == "ry":
                    qc.ry(theta, q)
                elif g == "rz":
                    qc.rz(theta, q)
                elif g == "h":
                    qc.h(q)
                elif g == "x":
                    qc.x(q)
                elif g == "y":
                    qc.y(q)
                elif g == "z":
                    qc.z(q)
                elif g == "s":
                    qc.s(q)
                elif g == "sdg":
                    qc.sdg(q)
                elif g == "tdg":
                    qc.tdg(q)
                else:
                    qc.t(q)
            else:
                a = rng.randrange(nq)
                b = rng.randrange(nq - 1)
                if b >= a:
                    b += 1
                qa, qb = qr[a], qr[b]
                g2 = rng.choice(("cx", "cz", "cy", "iswap", "rzz"))
                if g2 == "cx":
                    qc.cx(qa, qb)
                elif g2 == "cz":
                    qc.cz(qa, qb)
                elif g2 == "cy":
                    qc.cy(qa, qb)
                elif g2 == "iswap":
                    from qiskit.circuit.library import iSwapGate

                    qc.append(iSwapGate(), [qa, qb])
                else:
                    from qiskit.circuit.library import RZZGate

                    qc.append(RZZGate(rng.uniform(-math.pi, math.pi)), [qa, qb])

            # Inject periodic simplification patterns.
            if i % 9 == 0:
                q1 = qr[rng.randrange(nq)]
                qc.rz(0.3, q1)
                qc.rz(-0.3, q1)
            if i % 11 == 0:
                q2 = qr[rng.randrange(nq)]
                qc.s(q2)
                qc.sdg(q2)
            if i % 13 == 0:
                a = rng.randrange(nq)
                b = (a + 1) % nq
                qc.cx(qr[a], qr[b])
                qc.cx(qr[a], qr[b])
            if i % 17 == 0:
                q3 = qr[rng.randrange(nq)]
                qc.h(q3)
                qc.z(q3)
                qc.h(q3)

        # Add deterministic padding opportunities.
        _pad_mixed_single_qubit(qc, qr[rng.randrange(nq)], pad_level)
        return qc

    return _builder


BASELINE_BUILDERS: Dict[str, Callable[[int], QuantumCircuit]] = {
    # Internal / debug
    "toy": build_toy_circuit,
    "conjugation": build_conjugation_circuit,
    "identity_cleanup": build_identity_cleanup_circuit,
    "routing": build_routing_circuit,
    "entangler_family": build_entangler_family_circuit,

    # Demo-visible, classical-start narrative
    "parity": build_parity_circuit,
    "half_adder": build_half_adder_circuit,
    "majority": build_majority_circuit,
    "line": build_line_circuit,
}

DEMO_BUILDERS: Dict[str, Callable[[int], QuantumCircuit]] = {
    "parity": build_parity_circuit,
    "half_adder": build_half_adder_circuit,
    "majority": build_majority_circuit,
    "line": build_line_circuit,
}


# Weights used by mixed training sampling.
# Lower weight for less helpful/overlapping examples.
TRAIN_MIXED_BASELINE_WEIGHTS: Dict[str, float] = {
    "toy": 0.8,
    "parity": 1.0,
    "half_adder": 1.1,
    "majority": 0.0,
    "line": 0.15,
    "conjugation": 1.0,
    "identity_cleanup": 1.0,
    "routing": 1.2,
    "entangler_family": 1.0,
}


def get_mixed_training_weights() -> Dict[str, float]:
    """
    Return mixed-training baseline weights aligned with registered builders.
    """
    return {
        name: float(TRAIN_MIXED_BASELINE_WEIGHTS.get(name, 1.0))
        for name in BASELINE_BUILDERS.keys()
    }


def get_builder(name: str) -> Callable[[int], QuantumCircuit]:
    """
    Get a baseline circuit builder by name.
    """
    key = name.strip().lower()
    if key not in BASELINE_BUILDERS:
        raise KeyError(f"Unknown baseline circuit '{name}'. Options: {sorted(BASELINE_BUILDERS)}")
    return BASELINE_BUILDERS[key]


def get_demo_builder_names() -> list[str]:
    """
    Names of demo-visible baselines (meaningful classical-start options).
    """
    return list(DEMO_BUILDERS.keys())


def get_demo_builder(name: str) -> Callable[[int], QuantumCircuit]:
    """
    Get a demo-visible baseline circuit builder by name.
    """
    key = name.strip().lower()
    if key not in DEMO_BUILDERS:
        raise KeyError(f"Unknown demo baseline '{name}'. Options: {sorted(DEMO_BUILDERS)}")
    return DEMO_BUILDERS[key]
