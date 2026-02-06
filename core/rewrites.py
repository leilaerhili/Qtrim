"""
core/rewrites.py

Correctness-preserving local rewrite rules for quantum circuit optimization.

This module provides a small, high-impact action set that an RL agent can apply
to a quantum circuit. Each action attempts to perform ONE local rewrite
(pattern match + transform). If no valid application exists, the action returns
a no-op result.

Design goals
- Simple, safe, and deterministic.
- Local pattern-matching (adjacent or near-adjacent ops).
- Rebuilds a new Qiskit QuantumCircuit (does not mutate input).
- Keeps action space small (hackathon-friendly).

Supported patterns (by default)
1) Cancel double CX:      CX(a,b) CX(a,b) -> (remove both)
2) Cancel inverse RZ:     RZ(t) RZ(-t) -> (remove both)  [same qubit]
3) Merge adjacent RZ:     RZ(t1) RZ(t2) -> RZ(t1+t2)     [same qubit]
4) Swap commuting gates:  swap adjacent ops if they commute
5) Remove identity RZ:    RZ(0) -> (remove)

Notes
- These rules preserve circuit semantics.
- We intentionally keep the rules conservative; it is better to miss some
  possible rewrites than to introduce incorrect ones.
- For a 12-hour hackathon, avoid complex global rewrites or hardware-specific
  rewriting.

Usage
- Use list_actions() to get action ids and names.
- Use apply_action(circuit, action_id) to attempt one rewrite.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import math

from qiskit.circuit import QuantumCircuit, Qubit
from qiskit.circuit.instruction import Instruction


# -----------------------------
# Public results + action space
# -----------------------------

@dataclass(frozen=True)
class RewriteResult:
    """Result of attempting a rewrite action on a circuit."""
    changed: bool
    action_id: int
    action_name: str
    message: str
    old_len: int
    new_len: int
    # Optional indices of the instruction window that was rewritten (in the
    # internal flattened instruction list). Useful for debugging/visualization.
    window: Optional[Tuple[int, int]] = None


ActionFn = Callable[[QuantumCircuit], Tuple[QuantumCircuit, RewriteResult]]


# Keep action ids stable for RL.
ACTION_NOOP = 0
ACTION_CANCEL_DOUBLE_CX = 1
ACTION_CANCEL_INVERSE_RZ = 2
ACTION_MERGE_ADJACENT_RZ = 3
ACTION_SWAP_COMMUTING = 4
ACTION_REMOVE_IDENTITY_RZ = 5


def list_actions() -> List[Tuple[int, str]]:
    """Return (action_id, action_name) list in stable order."""
    return [
        (ACTION_NOOP, "noop"),
        (ACTION_CANCEL_DOUBLE_CX, "cancel_double_cx"),
        (ACTION_CANCEL_INVERSE_RZ, "cancel_inverse_rz"),
        (ACTION_MERGE_ADJACENT_RZ, "merge_adjacent_rz"),
        (ACTION_SWAP_COMMUTING, "swap_commuting"),
        (ACTION_REMOVE_IDENTITY_RZ, "remove_identity_rz"),
    ]


# -----------------------------
# Internal representation helpers
# -----------------------------

@dataclass(frozen=True)
class Op:
    """Lightweight operation representation for local rewriting."""
    inst: Instruction
    qargs: Tuple[Qubit, ...]
    cargs: Tuple = ()

    @property
    def name(self) -> str:
        return self.inst.name

    @property
    def params(self) -> Tuple[float, ...]:
        # Qiskit stores params as list-like; we normalize to tuple of floats when possible.
        p = getattr(self.inst, "params", [])
        try:
            return tuple(float(x) for x in p)
        except Exception:
            # If symbolic params exist, keep as-is (string conversion would be risky).
            return tuple(p)  # type: ignore[return-value]


def _flatten_ops(circ: QuantumCircuit) -> List[Op]:
    """Flatten circuit data into list of Ops, ignoring classical conditionals."""
    ops: List[Op] = []
    for ci in circ.data:
        ops.append(
            Op(
                inst=ci.operation,
                qargs=tuple(ci.qubits),
                cargs=tuple(ci.clbits),
            )
        )

    return ops


def _rebuild_from_ops(template: QuantumCircuit, ops: Sequence[Op]) -> QuantumCircuit:
    """
    Rebuild a new QuantumCircuit with the same registers as template,
    appending provided ops in order.
    """
    new_circ = QuantumCircuit(*template.qregs, *template.cregs, name=template.name)
    for op in ops:
        new_circ.append(op.inst, op.qargs, op.cargs)
    return new_circ


# -----------------------------
# Numeric helpers
# -----------------------------

def _is_close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def _wrap_angle(theta: float) -> float:
    """
    Wrap angle to [-pi, pi) for stable comparisons and nicer circuits.
    """
    # Handle NaNs conservatively.
    if not math.isfinite(theta):
        return theta
    two_pi = 2.0 * math.pi
    x = (theta + math.pi) % two_pi - math.pi
    # Map -pi to +pi for consistency if desired; here we keep [-pi, pi).
    return x


def _is_identity_rz(theta: float, tol: float = 1e-9) -> bool:
    """
    RZ(theta) is identity up to global phase when theta is a multiple of 2*pi.
    We'll treat near-0 as removable.
    """
    t = _wrap_angle(theta)
    return _is_close(t, 0.0, tol=tol)


# -----------------------------
# Commutation checks (conservative)
# -----------------------------

def _disjoint_qubits(a: Op, b: Op) -> bool:
    return set(a.qargs).isdisjoint(set(b.qargs))


def _commute(a: Op, b: Op) -> bool:
    """
    Conservative commutation predicate.

    We allow swapping adjacent ops when:
    - They act on disjoint qubits (always commute).
    - Both are single-qubit Z-rotations (rz) on the same qubit (commute).
    - Both are single-qubit ops on the same qubit and are both Z-rotations (already covered).
    Otherwise: return False (don't swap).
    """
    if _disjoint_qubits(a, b):
        return True
    # Same qubit + both RZ: commute (order doesn't matter; they can be merged later).
    if a.name == "rz" and b.name == "rz" and len(a.qargs) == 1 and len(b.qargs) == 1 and a.qargs[0] == b.qargs[0]:
        return True
    return False


# -----------------------------
# Rewrite implementations
# -----------------------------

def _noop(circ: QuantumCircuit) -> Tuple[QuantumCircuit, RewriteResult]:
    ops = _flatten_ops(circ)
    res = RewriteResult(
        changed=False,
        action_id=ACTION_NOOP,
        action_name="noop",
        message="No operation performed.",
        old_len=len(ops),
        new_len=len(ops),
        window=None,
    )
    return circ.copy(), res


def _cancel_double_cx(circ: QuantumCircuit) -> Tuple[QuantumCircuit, RewriteResult]:
    ops = _flatten_ops(circ)
    n = len(ops)

    for i in range(n - 1):
        a, b = ops[i], ops[i + 1]
        if a.name == "cx" and b.name == "cx" and a.qargs == b.qargs:
            new_ops = ops[:i] + ops[i + 2 :]
            new_circ = _rebuild_from_ops(circ, new_ops)
            res = RewriteResult(
                changed=True,
                action_id=ACTION_CANCEL_DOUBLE_CX,
                action_name="cancel_double_cx",
                message=f"Cancelled adjacent CX pair at positions {i},{i+1}.",
                old_len=n,
                new_len=len(new_ops),
                window=(i, i + 1),
            )
            return new_circ, res

    res = RewriteResult(
        changed=False,
        action_id=ACTION_CANCEL_DOUBLE_CX,
        action_name="cancel_double_cx",
        message="No cancellable adjacent CX pair found.",
        old_len=n,
        new_len=n,
        window=None,
    )
    return circ.copy(), res


def _cancel_inverse_rz(circ: QuantumCircuit, tol: float = 1e-9) -> Tuple[QuantumCircuit, RewriteResult]:
    ops = _flatten_ops(circ)
    n = len(ops)

    for i in range(n - 1):
        a, b = ops[i], ops[i + 1]
        if a.name == "rz" and b.name == "rz" and len(a.qargs) == 1 and a.qargs == b.qargs:
            t1, t2 = a.params[0], b.params[0]
            if _is_close(_wrap_angle(t1 + t2), 0.0, tol=tol):
                new_ops = ops[:i] + ops[i + 2 :]
                new_circ = _rebuild_from_ops(circ, new_ops)
                res = RewriteResult(
                    changed=True,
                    action_id=ACTION_CANCEL_INVERSE_RZ,
                    action_name="cancel_inverse_rz",
                    message=f"Cancelled adjacent inverse RZ at positions {i},{i+1}.",
                    old_len=n,
                    new_len=len(new_ops),
                    window=(i, i + 1),
                )
                return new_circ, res

    res = RewriteResult(
        changed=False,
        action_id=ACTION_CANCEL_INVERSE_RZ,
        action_name="cancel_inverse_rz",
        message="No adjacent inverse RZ pair found.",
        old_len=n,
        new_len=n,
        window=None,
    )
    return circ.copy(), res


def _merge_adjacent_rz(circ: QuantumCircuit, tol: float = 1e-9) -> Tuple[QuantumCircuit, RewriteResult]:
    ops = _flatten_ops(circ)
    n = len(ops)

    for i in range(n - 1):
        a, b = ops[i], ops[i + 1]
        if a.name == "rz" and b.name == "rz" and len(a.qargs) == 1 and a.qargs == b.qargs:
            t = _wrap_angle(a.params[0] + b.params[0])
            # If merge yields identity, remove both (equivalent to cancellation).
            if _is_identity_rz(t, tol=tol):
                new_ops = ops[:i] + ops[i + 2 :]
                msg = f"Merged adjacent RZ into identity and removed at positions {i},{i+1}."
            else:
                merged_inst = a.inst.copy()
                merged_inst.params = [t]
                merged = Op(inst=merged_inst, qargs=a.qargs, cargs=a.cargs)
                new_ops = ops[:i] + [merged] + ops[i + 2 :]
                msg = f"Merged adjacent RZ at positions {i},{i+1} into RZ({t:.6g})."
            new_circ = _rebuild_from_ops(circ, new_ops)
            res = RewriteResult(
                changed=True,
                action_id=ACTION_MERGE_ADJACENT_RZ,
                action_name="merge_adjacent_rz",
                message=msg,
                old_len=n,
                new_len=len(new_ops),
                window=(i, i + 1),
            )
            return new_circ, res

    res = RewriteResult(
        changed=False,
        action_id=ACTION_MERGE_ADJACENT_RZ,
        action_name="merge_adjacent_rz",
        message="No mergeable adjacent RZ pair found.",
        old_len=n,
        new_len=n,
        window=None,
    )
    return circ.copy(), res


def _swap_commuting(circ: QuantumCircuit) -> Tuple[QuantumCircuit, RewriteResult]:
    ops = _flatten_ops(circ)
    n = len(ops)

    for i in range(n - 1):
        a, b = ops[i], ops[i + 1]
        if _commute(a, b):
            new_ops = ops[:i] + [b, a] + ops[i + 2 :]
            new_circ = _rebuild_from_ops(circ, new_ops)
            res = RewriteResult(
                changed=True,
                action_id=ACTION_SWAP_COMMUTING,
                action_name="swap_commuting",
                message=f"Swapped commuting ops at positions {i},{i+1}: {a.name} <-> {b.name}.",
                old_len=n,
                new_len=n,
                window=(i, i + 1),
            )
            return new_circ, res

    res = RewriteResult(
        changed=False,
        action_id=ACTION_SWAP_COMMUTING,
        action_name="swap_commuting",
        message="No swappable commuting adjacent ops found.",
        old_len=n,
        new_len=n,
        window=None,
    )
    return circ.copy(), res


def _remove_identity_rz(circ: QuantumCircuit, tol: float = 1e-9) -> Tuple[QuantumCircuit, RewriteResult]:
    ops = _flatten_ops(circ)
    n = len(ops)

    for i, op in enumerate(ops):
        if op.name == "rz" and len(op.qargs) == 1:
            theta = op.params[0]
            if _is_identity_rz(theta, tol=tol):
                new_ops = ops[:i] + ops[i + 1 :]
                new_circ = _rebuild_from_ops(circ, new_ops)
                res = RewriteResult(
                    changed=True,
                    action_id=ACTION_REMOVE_IDENTITY_RZ,
                    action_name="remove_identity_rz",
                    message=f"Removed identity RZ at position {i}.",
                    old_len=n,
                    new_len=len(new_ops),
                    window=(i, i),
                )
                return new_circ, res

    res = RewriteResult(
        changed=False,
        action_id=ACTION_REMOVE_IDENTITY_RZ,
        action_name="remove_identity_rz",
        message="No identity RZ found.",
        old_len=n,
        new_len=n,
        window=None,
    )
    return circ.copy(), res


# -----------------------------
# Public dispatcher
# -----------------------------

_ACTIONS: Dict[int, ActionFn] = {
    ACTION_NOOP: _noop,
    ACTION_CANCEL_DOUBLE_CX: _cancel_double_cx,
    ACTION_CANCEL_INVERSE_RZ: _cancel_inverse_rz,
    ACTION_MERGE_ADJACENT_RZ: _merge_adjacent_rz,
    ACTION_SWAP_COMMUTING: _swap_commuting,
    ACTION_REMOVE_IDENTITY_RZ: _remove_identity_rz,
}


def apply_action(circuit: QuantumCircuit, action_id: int) -> Tuple[QuantumCircuit, RewriteResult]:
    """
    Attempt to apply a rewrite action to a circuit.

    Parameters
    ----------
    circuit:
        Input circuit (will not be mutated).
    action_id:
        Integer id from list_actions().

    Returns
    -------
    (new_circuit, result):
        new_circuit is a new QuantumCircuit object (may be identical if no-op).
        result describes whether a rewrite occurred.
    """
    if action_id not in _ACTIONS:
        raise ValueError(f"Unknown action_id={action_id}. Valid: {[a for a, _ in list_actions()]}")
    return _ACTIONS[action_id](circuit)


# -----------------------------
# Self-test / quick demo
# -----------------------------

def _quick_demo() -> None:
    """
    Quick manual sanity test you can run:
        python -m core.rewrites
    """
    from qiskit.circuit import QuantumRegister

    qr = QuantumRegister(3, "q")
    qc = QuantumCircuit(qr)
    qc.cx(qr[0], qr[1])
    qc.cx(qr[0], qr[1])  # cancellable
    qc.rz(0.7, qr[2])
    qc.rz(-0.7, qr[2])   # cancellable
    qc.rz(0.2, qr[2])
    qc.rz(0.3, qr[2])    # mergeable

    print("Original circuit:")
    print(qc)

    for action_id, name in list_actions():
        new_qc, res = apply_action(qc, action_id)
        if res.changed:
            print(f"\nApplied action {action_id} ({name}): {res.message}")
            print(new_qc)
            break


if __name__ == "__main__":
    _quick_demo()
