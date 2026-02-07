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
6) Cancel double H:       H H -> (remove)
7) Cancel double X:       X X -> (remove)
8) Merge adjacent RX:     RX(t1) RX(t2) -> RX(t1+t2)
9) Merge adjacent RY:     RY(t1) RY(t2) -> RY(t1+t2)
10) Remove identity RX:   RX(0) -> (remove)
11) Remove identity RY:   RY(0) -> (remove)
12) Cancel double Z:      Z Z -> (remove)
13) Cancel double Y:      Y Y -> (remove)
14) Cancel S/Sdg:         S Sdg -> (remove)
15) Cancel T/Tdg:         T Tdg -> (remove)
16) Cancel double CZ:     CZ(a,b) CZ(a,b) -> (remove both)
17) Commute RZ on CX ctl: RZ(c) CX(c,t) -> CX(c,t) RZ(c)
18) CX-RZ- CX (control):  CX(c,t) RZ(c) CX(c,t) -> RZ(c)
19) Toggle CX/CZ with H:  H(t) CX(c,t) H(t) <-> CZ(c,t)
20) Non-local cancel (<=12): remove cancelable pairs within 12-gate window if
    all in-between ops are disjoint from the pair's qubits.
21) Fuse phase chain (<=7): merge consecutive phase gates on one qubit into RZ.
22) Conjugate H-G-H:        H X H -> Z, H Z H -> X, H Y H -> Y.
23) Conjugate S-G-Sdg:      S X Sdg -> Y, S Y Sdg -> X.
24) Non-local commute cancel (<=12): remove cancelable pairs within 12-gate
    window if all in-between ops commute with both endpoints.
25) 2-qubit resynthesis (<=7): replace a 2-qubit window with an equivalent
    shorter 2-qubit circuit if it reduces gate count.
26) 3-qubit resynthesis (<=7): replace a 3-qubit window with CCX/CCZ/CSWAP
    or RCCX if it matches and reduces gate count.
27) Route CX via SWAP (hw-aware): if a CX is not on a coupled edge but a
    length-2 path exists, insert SWAPs to route it.
28) Remove SWAP routing (hw-aware): if SWAP-CX-SWAP matches and CX is directly
    coupled, replace with a single CX.
29) Depth-aware commute: swap commuting adjacent ops if it reduces local depth.
30) Route CZ/CY via SWAP (hw-aware): similar routing for CZ and CY.
31) Global depth-aware commute: search longer windows and pick best local
    swap that reduces depth.
32) Multi-swap depth scheduling: perform multiple commuting swaps (<=3) in a
    window if it reduces depth the most.
33) Single-qubit chain fusion (<=7): collapse any consecutive single-qubit
    chain on a qubit into a single unitary.
34) Route ECR/iSWAP/RZZ via SWAP (hw-aware): route additional 2Q gates using
    a length-2 path with direction rules where applicable.
35) Route RXX/RYY/RZZ variants via SWAP (hw-aware) for parametric interactions.
36) Big-window depth scheduling: multi-swap depth optimization across a
    larger window.
37) Global multi-swap depth scheduling: greedily apply multiple commuting
    swaps across the full circuit to reduce depth.
38) Multi-qubit fusion (<=3Q, <=5 ops): replace a short multi-qubit window
    with a single UnitaryGate if it reduces gate count.
39) 4-qubit resynthesis (<=7): replace a 4-qubit window with C3X/RC3X if it
    matches and reduces gate count.
40) Route additional directed gates (CX/ECR) with direction rules; undirected
    routing is used for symmetric gates.

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
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import math
import numpy as np

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
ACTION_CANCEL_DOUBLE_H = 6
ACTION_CANCEL_DOUBLE_X = 7
ACTION_MERGE_ADJACENT_RX = 8
ACTION_MERGE_ADJACENT_RY = 9
ACTION_REMOVE_IDENTITY_RX = 10
ACTION_REMOVE_IDENTITY_RY = 11
ACTION_CANCEL_DOUBLE_Z = 12
ACTION_CANCEL_DOUBLE_Y = 13
ACTION_CANCEL_S_SDG = 14
ACTION_CANCEL_T_TDG = 15
ACTION_CANCEL_DOUBLE_CZ = 16
ACTION_COMMUTE_RZ_THROUGH_CX_CONTROL = 17
ACTION_CANCEL_CX_RZ_CX_CONTROL = 18
ACTION_TOGGLE_CX_CZ_WITH_H = 19
ACTION_CANCEL_NONLOCAL_PAIR_12 = 20
ACTION_FUSE_PHASE_CHAIN_7 = 21
ACTION_CONJUGATE_H_G_H = 22
ACTION_CONJUGATE_S_G_SDG = 23
ACTION_CANCEL_NONLOCAL_COMMUTE_12 = 24
ACTION_RESYNTH_2Q_WINDOW_7 = 25
ACTION_RESYNTH_3Q_WINDOW_5 = 26
ACTION_ROUTE_CX_WITH_SWAP = 27
ACTION_UNROUTE_CX_WITH_SWAP = 28
ACTION_DEPTH_AWARE_COMMUTE = 29
ACTION_ROUTE_CZ_CY_WITH_SWAP = 30
ACTION_DEPTH_AWARE_COMMUTE_GLOBAL = 31
ACTION_DEPTH_AWARE_MULTI_SWAP = 32
ACTION_FUSE_1Q_CHAIN_7 = 33
ACTION_ROUTE_ECR_ISWAP_RZZ_WITH_SWAP = 34
ACTION_ROUTE_RXX_RYY_RZZ_WITH_SWAP = 35
ACTION_DEPTH_AWARE_MULTI_SWAP_GLOBAL = 36
ACTION_DEPTH_AWARE_MULTI_SWAP_FULL = 37
ACTION_FUSE_MULTI_Q_UNITARY = 38
ACTION_RESYNTH_4Q_WINDOW_7 = 39
ACTION_ROUTE_DIRECTED_MORE = 40


def list_actions() -> List[Tuple[int, str]]:
    """Return (action_id, action_name) list in stable order."""
    return [
        (ACTION_NOOP, "noop"),
        (ACTION_CANCEL_DOUBLE_CX, "cancel_double_cx"),
        (ACTION_CANCEL_INVERSE_RZ, "cancel_inverse_rz"),
        (ACTION_MERGE_ADJACENT_RZ, "merge_adjacent_rz"),
        (ACTION_SWAP_COMMUTING, "swap_commuting"),
        (ACTION_REMOVE_IDENTITY_RZ, "remove_identity_rz"),
        (ACTION_CANCEL_DOUBLE_H, "cancel_double_h"),
        (ACTION_CANCEL_DOUBLE_X, "cancel_double_x"),
        (ACTION_MERGE_ADJACENT_RX, "merge_adjacent_rx"),
        (ACTION_MERGE_ADJACENT_RY, "merge_adjacent_ry"),
        (ACTION_REMOVE_IDENTITY_RX, "remove_identity_rx"),
        (ACTION_REMOVE_IDENTITY_RY, "remove_identity_ry"),
        (ACTION_CANCEL_DOUBLE_Z, "cancel_double_z"),
        (ACTION_CANCEL_DOUBLE_Y, "cancel_double_y"),
        (ACTION_CANCEL_S_SDG, "cancel_s_sdg"),
        (ACTION_CANCEL_T_TDG, "cancel_t_tdg"),
        (ACTION_CANCEL_DOUBLE_CZ, "cancel_double_cz"),
        (ACTION_COMMUTE_RZ_THROUGH_CX_CONTROL, "commute_rz_through_cx_control"),
        (ACTION_CANCEL_CX_RZ_CX_CONTROL, "cancel_cx_rz_cx_control"),
        (ACTION_TOGGLE_CX_CZ_WITH_H, "toggle_cx_cz_with_h"),
        (ACTION_CANCEL_NONLOCAL_PAIR_12, "cancel_nonlocal_pair_12"),
        (ACTION_FUSE_PHASE_CHAIN_7, "fuse_phase_chain_7"),
        (ACTION_CONJUGATE_H_G_H, "conjugate_h_g_h"),
        (ACTION_CONJUGATE_S_G_SDG, "conjugate_s_g_sdg"),
        (ACTION_CANCEL_NONLOCAL_COMMUTE_12, "cancel_nonlocal_commute_12"),
        (ACTION_RESYNTH_2Q_WINDOW_7, "resynth_2q_window_7"),
        (ACTION_RESYNTH_3Q_WINDOW_5, "resynth_3q_window_7"),
        (ACTION_ROUTE_CX_WITH_SWAP, "route_cx_with_swap"),
        (ACTION_UNROUTE_CX_WITH_SWAP, "unroute_cx_with_swap"),
        (ACTION_DEPTH_AWARE_COMMUTE, "depth_aware_commute"),
        (ACTION_ROUTE_CZ_CY_WITH_SWAP, "route_cz_cy_with_swap"),
        (ACTION_DEPTH_AWARE_COMMUTE_GLOBAL, "depth_aware_commute_global"),
        (ACTION_DEPTH_AWARE_MULTI_SWAP, "depth_aware_multi_swap"),
        (ACTION_FUSE_1Q_CHAIN_7, "fuse_1q_chain_7"),
        (ACTION_ROUTE_ECR_ISWAP_RZZ_WITH_SWAP, "route_ecr_iswap_rzz_with_swap"),
        (ACTION_ROUTE_RXX_RYY_RZZ_WITH_SWAP, "route_rxx_ryy_rzz_with_swap"),
        (ACTION_DEPTH_AWARE_MULTI_SWAP_GLOBAL, "depth_aware_multi_swap_global"),
        (ACTION_DEPTH_AWARE_MULTI_SWAP_FULL, "depth_aware_multi_swap_full"),
        (ACTION_FUSE_MULTI_Q_UNITARY, "fuse_multi_q_unitary"),
        (ACTION_RESYNTH_4Q_WINDOW_7, "resynth_4q_window_7"),
        (ACTION_ROUTE_DIRECTED_MORE, "route_directed_more"),
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
# Hardware coupling map helpers
# -----------------------------

_COUPLING_MAP: Optional[set[frozenset[int]]] = None
_COUPLING_DIRECTED: Optional[set[Tuple[int, int]]] = None


def set_coupling_map(edges: Sequence[Tuple[int, int]], *, directed: bool = False) -> None:
    """
    Set the hardware coupling map.

    If directed is False, edges are treated as undirected (i, j).
    If directed is True, edges are treated as directed (i -> j).
    """
    global _COUPLING_MAP
    global _COUPLING_DIRECTED
    if directed:
        _COUPLING_DIRECTED = {(int(a), int(b)) for a, b in edges}
        _COUPLING_MAP = {frozenset((int(a), int(b))) for a, b in edges}
    else:
        _COUPLING_MAP = {frozenset((int(a), int(b))) for a, b in edges}
        _COUPLING_DIRECTED = None


def _qubit_index_map(circ: QuantumCircuit) -> Dict[Qubit, int]:
    return {q: i for i, q in enumerate(circ.qubits)}


def _is_coupled(i: int, j: int, *, directed_ok: bool = False) -> bool:
    if _COUPLING_MAP is None and _COUPLING_DIRECTED is None:
        return False
    if directed_ok and _COUPLING_DIRECTED is not None:
        return (i, j) in _COUPLING_DIRECTED
    if _COUPLING_MAP is not None:
        return frozenset((i, j)) in _COUPLING_MAP
    return False
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


def _is_identity_rotation(theta: float, tol: float = 1e-9) -> bool:
    """
    Rotation by a multiple of 2*pi is identity up to global phase.
    """
    t = _wrap_angle(theta)
    return _is_close(t, 0.0, tol=tol)


# -----------------------------
# Commutation checks (conservative)
# -----------------------------

def _disjoint_qubits(a: Op, b: Op) -> bool:
    return set(a.qargs).isdisjoint(set(b.qargs))


_X_AXIS = {"x", "rx"}
_Y_AXIS = {"y", "ry"}
_Z_AXIS = {"z", "rz", "s", "sdg", "t", "tdg"}


def _axis(op: Op) -> Optional[str]:
    if op.name in _X_AXIS:
        return "x"
    if op.name in _Y_AXIS:
        return "y"
    if op.name in _Z_AXIS:
        return "z"
    return None


def _commute(a: Op, b: Op) -> bool:
    """
    Conservative commutation predicate.

    We allow swapping adjacent ops when:
    - They act on disjoint qubits (always commute).
    - Both are single-qubit ops on the same qubit and share an axis (X, Y, or Z).
    - A CX with Z-axis op on its control or X-axis op on its target.
    - A CZ with Z-axis op on either qubit.
    Otherwise: return False (don't swap).
    """
    if _disjoint_qubits(a, b):
        return True
    if len(a.qargs) == 1 and len(b.qargs) == 1 and a.qargs[0] == b.qargs[0]:
        ax = _axis(a)
        bx = _axis(b)
        if ax is not None and ax == bx:
            return True
    if a.name == "cx" and len(b.qargs) == 1:
        control = a.qargs[0]
        target = a.qargs[1]
        bx = _axis(b)
        if b.qargs[0] == control and bx == "z":
            return True
        if b.qargs[0] == target and bx == "x":
            return True
    if b.name == "cx" and len(a.qargs) == 1:
        control = b.qargs[0]
        target = b.qargs[1]
        ax = _axis(a)
        if a.qargs[0] == control and ax == "z":
            return True
        if a.qargs[0] == target and ax == "x":
            return True
    if a.name == "cz" and len(b.qargs) == 1:
        if b.qargs[0] in a.qargs and _axis(b) == "z":
            return True
    if b.name == "cz" and len(a.qargs) == 1:
        if a.qargs[0] in b.qargs and _axis(a) == "z":
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
            if _is_identity_rotation(t, tol=tol):
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
            if _is_identity_rotation(theta, tol=tol):
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


def _cancel_double_h(circ: QuantumCircuit) -> Tuple[QuantumCircuit, RewriteResult]:
    ops = _flatten_ops(circ)
    n = len(ops)

    for i in range(n - 1):
        a, b = ops[i], ops[i + 1]
        if a.name == "h" and b.name == "h" and a.qargs == b.qargs:
            new_ops = ops[:i] + ops[i + 2 :]
            new_circ = _rebuild_from_ops(circ, new_ops)
            res = RewriteResult(
                changed=True,
                action_id=ACTION_CANCEL_DOUBLE_H,
                action_name="cancel_double_h",
                message=f"Cancelled adjacent H pair at positions {i},{i+1}.",
                old_len=n,
                new_len=len(new_ops),
                window=(i, i + 1),
            )
            return new_circ, res

    res = RewriteResult(
        changed=False,
        action_id=ACTION_CANCEL_DOUBLE_H,
        action_name="cancel_double_h",
        message="No cancellable adjacent H pair found.",
        old_len=n,
        new_len=n,
        window=None,
    )
    return circ.copy(), res


def _cancel_double_x(circ: QuantumCircuit) -> Tuple[QuantumCircuit, RewriteResult]:
    ops = _flatten_ops(circ)
    n = len(ops)

    for i in range(n - 1):
        a, b = ops[i], ops[i + 1]
        if a.name == "x" and b.name == "x" and a.qargs == b.qargs:
            new_ops = ops[:i] + ops[i + 2 :]
            new_circ = _rebuild_from_ops(circ, new_ops)
            res = RewriteResult(
                changed=True,
                action_id=ACTION_CANCEL_DOUBLE_X,
                action_name="cancel_double_x",
                message=f"Cancelled adjacent X pair at positions {i},{i+1}.",
                old_len=n,
                new_len=len(new_ops),
                window=(i, i + 1),
            )
            return new_circ, res

    res = RewriteResult(
        changed=False,
        action_id=ACTION_CANCEL_DOUBLE_X,
        action_name="cancel_double_x",
        message="No cancellable adjacent X pair found.",
        old_len=n,
        new_len=n,
        window=None,
    )
    return circ.copy(), res


def _merge_adjacent_rx(circ: QuantumCircuit, tol: float = 1e-9) -> Tuple[QuantumCircuit, RewriteResult]:
    ops = _flatten_ops(circ)
    n = len(ops)

    for i in range(n - 1):
        a, b = ops[i], ops[i + 1]
        if a.name == "rx" and b.name == "rx" and len(a.qargs) == 1 and a.qargs == b.qargs:
            t = _wrap_angle(a.params[0] + b.params[0])
            if _is_identity_rotation(t, tol=tol):
                new_ops = ops[:i] + ops[i + 2 :]
                msg = f"Merged adjacent RX into identity and removed at positions {i},{i+1}."
            else:
                merged_inst = a.inst.copy()
                merged_inst.params = [t]
                merged = Op(inst=merged_inst, qargs=a.qargs, cargs=a.cargs)
                new_ops = ops[:i] + [merged] + ops[i + 2 :]
                msg = f"Merged adjacent RX at positions {i},{i+1} into RX({t:.6g})."
            new_circ = _rebuild_from_ops(circ, new_ops)
            res = RewriteResult(
                changed=True,
                action_id=ACTION_MERGE_ADJACENT_RX,
                action_name="merge_adjacent_rx",
                message=msg,
                old_len=n,
                new_len=len(new_ops),
                window=(i, i + 1),
            )
            return new_circ, res

    res = RewriteResult(
        changed=False,
        action_id=ACTION_MERGE_ADJACENT_RX,
        action_name="merge_adjacent_rx",
        message="No mergeable adjacent RX pair found.",
        old_len=n,
        new_len=n,
        window=None,
    )
    return circ.copy(), res


def _merge_adjacent_ry(circ: QuantumCircuit, tol: float = 1e-9) -> Tuple[QuantumCircuit, RewriteResult]:
    ops = _flatten_ops(circ)
    n = len(ops)

    for i in range(n - 1):
        a, b = ops[i], ops[i + 1]
        if a.name == "ry" and b.name == "ry" and len(a.qargs) == 1 and a.qargs == b.qargs:
            t = _wrap_angle(a.params[0] + b.params[0])
            if _is_identity_rotation(t, tol=tol):
                new_ops = ops[:i] + ops[i + 2 :]
                msg = f"Merged adjacent RY into identity and removed at positions {i},{i+1}."
            else:
                merged_inst = a.inst.copy()
                merged_inst.params = [t]
                merged = Op(inst=merged_inst, qargs=a.qargs, cargs=a.cargs)
                new_ops = ops[:i] + [merged] + ops[i + 2 :]
                msg = f"Merged adjacent RY at positions {i},{i+1} into RY({t:.6g})."
            new_circ = _rebuild_from_ops(circ, new_ops)
            res = RewriteResult(
                changed=True,
                action_id=ACTION_MERGE_ADJACENT_RY,
                action_name="merge_adjacent_ry",
                message=msg,
                old_len=n,
                new_len=len(new_ops),
                window=(i, i + 1),
            )
            return new_circ, res

    res = RewriteResult(
        changed=False,
        action_id=ACTION_MERGE_ADJACENT_RY,
        action_name="merge_adjacent_ry",
        message="No mergeable adjacent RY pair found.",
        old_len=n,
        new_len=n,
        window=None,
    )
    return circ.copy(), res


def _remove_identity_rx(circ: QuantumCircuit, tol: float = 1e-9) -> Tuple[QuantumCircuit, RewriteResult]:
    ops = _flatten_ops(circ)
    n = len(ops)

    for i, op in enumerate(ops):
        if op.name == "rx" and len(op.qargs) == 1:
            theta = op.params[0]
            if _is_identity_rotation(theta, tol=tol):
                new_ops = ops[:i] + ops[i + 1 :]
                new_circ = _rebuild_from_ops(circ, new_ops)
                res = RewriteResult(
                    changed=True,
                    action_id=ACTION_REMOVE_IDENTITY_RX,
                    action_name="remove_identity_rx",
                    message=f"Removed identity RX at position {i}.",
                    old_len=n,
                    new_len=len(new_ops),
                    window=(i, i),
                )
                return new_circ, res

    res = RewriteResult(
        changed=False,
        action_id=ACTION_REMOVE_IDENTITY_RX,
        action_name="remove_identity_rx",
        message="No identity RX found.",
        old_len=n,
        new_len=n,
        window=None,
    )
    return circ.copy(), res


def _remove_identity_ry(circ: QuantumCircuit, tol: float = 1e-9) -> Tuple[QuantumCircuit, RewriteResult]:
    ops = _flatten_ops(circ)
    n = len(ops)

    for i, op in enumerate(ops):
        if op.name == "ry" and len(op.qargs) == 1:
            theta = op.params[0]
            if _is_identity_rotation(theta, tol=tol):
                new_ops = ops[:i] + ops[i + 1 :]
                new_circ = _rebuild_from_ops(circ, new_ops)
                res = RewriteResult(
                    changed=True,
                    action_id=ACTION_REMOVE_IDENTITY_RY,
                    action_name="remove_identity_ry",
                    message=f"Removed identity RY at position {i}.",
                    old_len=n,
                    new_len=len(new_ops),
                    window=(i, i),
                )
                return new_circ, res

    res = RewriteResult(
        changed=False,
        action_id=ACTION_REMOVE_IDENTITY_RY,
        action_name="remove_identity_ry",
        message="No identity RY found.",
        old_len=n,
        new_len=n,
        window=None,
    )
    return circ.copy(), res


def _cancel_double_z(circ: QuantumCircuit) -> Tuple[QuantumCircuit, RewriteResult]:
    ops = _flatten_ops(circ)
    n = len(ops)

    for i in range(n - 1):
        a, b = ops[i], ops[i + 1]
        if a.name == "z" and b.name == "z" and a.qargs == b.qargs:
            new_ops = ops[:i] + ops[i + 2 :]
            new_circ = _rebuild_from_ops(circ, new_ops)
            res = RewriteResult(
                changed=True,
                action_id=ACTION_CANCEL_DOUBLE_Z,
                action_name="cancel_double_z",
                message=f"Cancelled adjacent Z pair at positions {i},{i+1}.",
                old_len=n,
                new_len=len(new_ops),
                window=(i, i + 1),
            )
            return new_circ, res

    res = RewriteResult(
        changed=False,
        action_id=ACTION_CANCEL_DOUBLE_Z,
        action_name="cancel_double_z",
        message="No cancellable adjacent Z pair found.",
        old_len=n,
        new_len=n,
        window=None,
    )
    return circ.copy(), res


def _cancel_double_y(circ: QuantumCircuit) -> Tuple[QuantumCircuit, RewriteResult]:
    ops = _flatten_ops(circ)
    n = len(ops)

    for i in range(n - 1):
        a, b = ops[i], ops[i + 1]
        if a.name == "y" and b.name == "y" and a.qargs == b.qargs:
            new_ops = ops[:i] + ops[i + 2 :]
            new_circ = _rebuild_from_ops(circ, new_ops)
            res = RewriteResult(
                changed=True,
                action_id=ACTION_CANCEL_DOUBLE_Y,
                action_name="cancel_double_y",
                message=f"Cancelled adjacent Y pair at positions {i},{i+1}.",
                old_len=n,
                new_len=len(new_ops),
                window=(i, i + 1),
            )
            return new_circ, res

    res = RewriteResult(
        changed=False,
        action_id=ACTION_CANCEL_DOUBLE_Y,
        action_name="cancel_double_y",
        message="No cancellable adjacent Y pair found.",
        old_len=n,
        new_len=n,
        window=None,
    )
    return circ.copy(), res


def _cancel_s_sdg(circ: QuantumCircuit) -> Tuple[QuantumCircuit, RewriteResult]:
    ops = _flatten_ops(circ)
    n = len(ops)

    for i in range(n - 1):
        a, b = ops[i], ops[i + 1]
        if a.qargs == b.qargs:
            pair = (a.name, b.name)
            if pair in (("s", "sdg"), ("sdg", "s")):
                new_ops = ops[:i] + ops[i + 2 :]
                new_circ = _rebuild_from_ops(circ, new_ops)
                res = RewriteResult(
                    changed=True,
                    action_id=ACTION_CANCEL_S_SDG,
                    action_name="cancel_s_sdg",
                    message=f"Cancelled adjacent S/Sdg pair at positions {i},{i+1}.",
                    old_len=n,
                    new_len=len(new_ops),
                    window=(i, i + 1),
                )
                return new_circ, res

    res = RewriteResult(
        changed=False,
        action_id=ACTION_CANCEL_S_SDG,
        action_name="cancel_s_sdg",
        message="No cancellable adjacent S/Sdg pair found.",
        old_len=n,
        new_len=n,
        window=None,
    )
    return circ.copy(), res


def _cancel_t_tdg(circ: QuantumCircuit) -> Tuple[QuantumCircuit, RewriteResult]:
    ops = _flatten_ops(circ)
    n = len(ops)

    for i in range(n - 1):
        a, b = ops[i], ops[i + 1]
        if a.qargs == b.qargs:
            pair = (a.name, b.name)
            if pair in (("t", "tdg"), ("tdg", "t")):
                new_ops = ops[:i] + ops[i + 2 :]
                new_circ = _rebuild_from_ops(circ, new_ops)
                res = RewriteResult(
                    changed=True,
                    action_id=ACTION_CANCEL_T_TDG,
                    action_name="cancel_t_tdg",
                    message=f"Cancelled adjacent T/Tdg pair at positions {i},{i+1}.",
                    old_len=n,
                    new_len=len(new_ops),
                    window=(i, i + 1),
                )
                return new_circ, res

    res = RewriteResult(
        changed=False,
        action_id=ACTION_CANCEL_T_TDG,
        action_name="cancel_t_tdg",
        message="No cancellable adjacent T/Tdg pair found.",
        old_len=n,
        new_len=n,
        window=None,
    )
    return circ.copy(), res


def _cancel_double_cz(circ: QuantumCircuit) -> Tuple[QuantumCircuit, RewriteResult]:
    ops = _flatten_ops(circ)
    n = len(ops)

    for i in range(n - 1):
        a, b = ops[i], ops[i + 1]
        if a.name == "cz" and b.name == "cz" and a.qargs == b.qargs:
            new_ops = ops[:i] + ops[i + 2 :]
            new_circ = _rebuild_from_ops(circ, new_ops)
            res = RewriteResult(
                changed=True,
                action_id=ACTION_CANCEL_DOUBLE_CZ,
                action_name="cancel_double_cz",
                message=f"Cancelled adjacent CZ pair at positions {i},{i+1}.",
                old_len=n,
                new_len=len(new_ops),
                window=(i, i + 1),
            )
            return new_circ, res

    res = RewriteResult(
        changed=False,
        action_id=ACTION_CANCEL_DOUBLE_CZ,
        action_name="cancel_double_cz",
        message="No cancellable adjacent CZ pair found.",
        old_len=n,
        new_len=n,
        window=None,
    )
    return circ.copy(), res


def _commute_rz_through_cx_control(circ: QuantumCircuit) -> Tuple[QuantumCircuit, RewriteResult]:
    ops = _flatten_ops(circ)
    n = len(ops)

    for i in range(n - 1):
        a, b = ops[i], ops[i + 1]
        if a.name == "rz" and b.name == "cx" and len(a.qargs) == 1:
            if a.qargs[0] == b.qargs[0]:
                new_ops = ops[:i] + [b, a] + ops[i + 2 :]
                new_circ = _rebuild_from_ops(circ, new_ops)
                res = RewriteResult(
                    changed=True,
                    action_id=ACTION_COMMUTE_RZ_THROUGH_CX_CONTROL,
                    action_name="commute_rz_through_cx_control",
                    message=f"Swapped RZ on control through CX at positions {i},{i+1}.",
                    old_len=n,
                    new_len=n,
                    window=(i, i + 1),
                )
                return new_circ, res
        if a.name == "cx" and b.name == "rz" and len(b.qargs) == 1:
            if b.qargs[0] == a.qargs[0]:
                new_ops = ops[:i] + [b, a] + ops[i + 2 :]
                new_circ = _rebuild_from_ops(circ, new_ops)
                res = RewriteResult(
                    changed=True,
                    action_id=ACTION_COMMUTE_RZ_THROUGH_CX_CONTROL,
                    action_name="commute_rz_through_cx_control",
                    message=f"Swapped RZ on control through CX at positions {i},{i+1}.",
                    old_len=n,
                    new_len=n,
                    window=(i, i + 1),
                )
                return new_circ, res

    res = RewriteResult(
        changed=False,
        action_id=ACTION_COMMUTE_RZ_THROUGH_CX_CONTROL,
        action_name="commute_rz_through_cx_control",
        message="No CX-control RZ swap found.",
        old_len=n,
        new_len=n,
        window=None,
    )
    return circ.copy(), res


def _cancel_cx_rz_cx_control(circ: QuantumCircuit) -> Tuple[QuantumCircuit, RewriteResult]:
    ops = _flatten_ops(circ)
    n = len(ops)

    for i in range(n - 2):
        a, b, c = ops[i], ops[i + 1], ops[i + 2]
        if a.name == "cx" and c.name == "cx" and a.qargs == c.qargs:
            if b.name == "rz" and len(b.qargs) == 1 and b.qargs[0] == a.qargs[0]:
                new_ops = ops[:i] + [b] + ops[i + 3 :]
                new_circ = _rebuild_from_ops(circ, new_ops)
                res = RewriteResult(
                    changed=True,
                    action_id=ACTION_CANCEL_CX_RZ_CX_CONTROL,
                    action_name="cancel_cx_rz_cx_control",
                    message=f"Removed CX-RZ-CX with control RZ at positions {i},{i+1},{i+2}.",
                    old_len=n,
                    new_len=len(new_ops),
                    window=(i, i + 2),
                )
                return new_circ, res

    res = RewriteResult(
        changed=False,
        action_id=ACTION_CANCEL_CX_RZ_CX_CONTROL,
        action_name="cancel_cx_rz_cx_control",
        message="No CX-RZ(control)-CX pattern found.",
        old_len=n,
        new_len=n,
        window=None,
    )
    return circ.copy(), res


def _toggle_cx_cz_with_h(circ: QuantumCircuit) -> Tuple[QuantumCircuit, RewriteResult]:
    ops = _flatten_ops(circ)
    n = len(ops)

    for i in range(n - 2):
        a, b, c = ops[i], ops[i + 1], ops[i + 2]
        # H(t) CX(c,t) H(t) -> CZ(c,t)
        if a.name == "h" and b.name == "cx" and c.name == "h":
            if len(a.qargs) == 1 and len(c.qargs) == 1:
                target = a.qargs[0]
                if c.qargs[0] == target and len(b.qargs) == 2 and b.qargs[1] == target:
                    from qiskit.circuit.library import CZGate
                    new_ops = ops[:i] + [Op(inst=CZGate(), qargs=b.qargs, cargs=b.cargs)] + ops[i + 3 :]
                    new_circ = _rebuild_from_ops(circ, new_ops)
                    res = RewriteResult(
                        changed=True,
                        action_id=ACTION_TOGGLE_CX_CZ_WITH_H,
                        action_name="toggle_cx_cz_with_h",
                        message=f"Rewrote H-CX-H to CZ at positions {i},{i+1},{i+2}.",
                        old_len=n,
                        new_len=len(new_ops),
                        window=(i, i + 2),
                    )
                    return new_circ, res
        # H(t) CZ(c,t) H(t) -> CX(c,t)
        if a.name == "h" and b.name == "cz" and c.name == "h":
            if len(a.qargs) == 1 and len(c.qargs) == 1:
                target = a.qargs[0]
                if c.qargs[0] == target and len(b.qargs) == 2 and b.qargs[1] == target:
                    from qiskit.circuit.library import CXGate
                    new_ops = ops[:i] + [Op(inst=CXGate(), qargs=b.qargs, cargs=b.cargs)] + ops[i + 3 :]
                    new_circ = _rebuild_from_ops(circ, new_ops)
                    res = RewriteResult(
                        changed=True,
                        action_id=ACTION_TOGGLE_CX_CZ_WITH_H,
                        action_name="toggle_cx_cz_with_h",
                        message=f"Rewrote H-CZ-H to CX at positions {i},{i+1},{i+2}.",
                        old_len=n,
                        new_len=len(new_ops),
                        window=(i, i + 2),
                    )
                    return new_circ, res

    res = RewriteResult(
        changed=False,
        action_id=ACTION_TOGGLE_CX_CZ_WITH_H,
        action_name="toggle_cx_cz_with_h",
        message="No H-CX-H or H-CZ-H pattern found.",
        old_len=n,
        new_len=n,
        window=None,
    )
    return circ.copy(), res


def _ops_disjoint_from(op: Op, qset: set[Qubit]) -> bool:
    return set(op.qargs).isdisjoint(qset)


def _is_cancelable_pair(a: Op, b: Op, tol: float = 1e-9) -> bool:
    if a.name == "cx" and b.name == "cx" and a.qargs == b.qargs:
        return True
    if a.name == "cz" and b.name == "cz" and a.qargs == b.qargs:
        return True
    if a.name == "h" and b.name == "h" and a.qargs == b.qargs:
        return True
    if a.name == "x" and b.name == "x" and a.qargs == b.qargs:
        return True
    if a.name == "y" and b.name == "y" and a.qargs == b.qargs:
        return True
    if a.name == "z" and b.name == "z" and a.qargs == b.qargs:
        return True
    if a.name == "s" and b.name == "sdg" and a.qargs == b.qargs:
        return True
    if a.name == "sdg" and b.name == "s" and a.qargs == b.qargs:
        return True
    if a.name == "t" and b.name == "tdg" and a.qargs == b.qargs:
        return True
    if a.name == "tdg" and b.name == "t" and a.qargs == b.qargs:
        return True
    if a.name == "rz" and b.name == "rz" and a.qargs == b.qargs and len(a.qargs) == 1:
        return _is_identity_rotation(a.params[0] + b.params[0], tol=tol)
    if a.name == "rx" and b.name == "rx" and a.qargs == b.qargs and len(a.qargs) == 1:
        return _is_identity_rotation(a.params[0] + b.params[0], tol=tol)
    if a.name == "ry" and b.name == "ry" and a.qargs == b.qargs and len(a.qargs) == 1:
        return _is_identity_rotation(a.params[0] + b.params[0], tol=tol)
    return False


def _cancel_nonlocal_pair_12(circ: QuantumCircuit, window: int = 12) -> Tuple[QuantumCircuit, RewriteResult]:
    ops = _flatten_ops(circ)
    n = len(ops)

    max_span = max(1, int(window) - 1)
    for i in range(n):
        for j in range(i + 1, min(n, i + max_span + 1)):
            a, b = ops[i], ops[j]
            if not _is_cancelable_pair(a, b):
                continue
            # Prune: require same arity and identical qubits for non-rotation pairs.
            if a.name not in ("rz", "rx", "ry"):
                if a.qargs != b.qargs:
                    continue
            qset = set(a.qargs)
            if not qset.issuperset(set(b.qargs)):
                qset = set(a.qargs) | set(b.qargs)
            # Only allow if all in-between ops are disjoint from the pair's qubits.
            blocked = False
            for k in range(i + 1, j):
                if not _ops_disjoint_from(ops[k], qset):
                    blocked = True
                    break
            if blocked:
                continue
            new_ops = ops[:i] + ops[i + 1 : j] + ops[j + 1 :]
            new_circ = _rebuild_from_ops(circ, new_ops)
            res = RewriteResult(
                changed=True,
                action_id=ACTION_CANCEL_NONLOCAL_PAIR_12,
                action_name="cancel_nonlocal_pair_12",
                message=f"Cancelled non-local pair at positions {i},{j} within window {window}.",
                old_len=n,
                new_len=len(new_ops),
                window=(i, j),
            )
            return new_circ, res

    res = RewriteResult(
        changed=False,
        action_id=ACTION_CANCEL_NONLOCAL_PAIR_12,
        action_name="cancel_nonlocal_pair_12",
        message="No cancelable non-local pair found within 12-gate window.",
        old_len=n,
        new_len=n,
        window=None,
    )
    return circ.copy(), res


def _cancel_nonlocal_commute_12(circ: QuantumCircuit, window: int = 12) -> Tuple[QuantumCircuit, RewriteResult]:
    ops = _flatten_ops(circ)
    n = len(ops)

    max_span = max(1, int(window) - 1)
    for i in range(n):
        for j in range(i + 1, min(n, i + max_span + 1)):
            a, b = ops[i], ops[j]
            if not _is_cancelable_pair(a, b):
                continue
            if a.name not in ("rz", "rx", "ry"):
                if a.qargs != b.qargs:
                    continue
            blocked = False
            for k in range(i + 1, j):
                if not _commute(ops[k], a) or not _commute(ops[k], b):
                    blocked = True
                    break
            if blocked:
                continue
            new_ops = ops[:i] + ops[i + 1 : j] + ops[j + 1 :]
            new_circ = _rebuild_from_ops(circ, new_ops)
            res = RewriteResult(
                changed=True,
                action_id=ACTION_CANCEL_NONLOCAL_COMMUTE_12,
                action_name="cancel_nonlocal_commute_12",
                message=f"Cancelled commuting non-local pair at positions {i},{j} within window {window}.",
                old_len=n,
                new_len=len(new_ops),
                window=(i, j),
            )
            return new_circ, res

    res = RewriteResult(
        changed=False,
        action_id=ACTION_CANCEL_NONLOCAL_COMMUTE_12,
        action_name="cancel_nonlocal_commute_12",
        message="No commuting non-local pair found within 12-gate window.",
        old_len=n,
        new_len=n,
        window=None,
    )
    return circ.copy(), res


def _equal_up_to_global_phase(a: np.ndarray, b: np.ndarray, tol: float = 1e-8) -> bool:
    if a.shape != b.shape:
        return False
    idx = None
    for i in range(a.size):
        if abs(a.flat[i]) > tol or abs(b.flat[i]) > tol:
            idx = i
            break
    if idx is None:
        return True
    if abs(b.flat[idx]) <= tol:
        return False
    phase = a.flat[idx] / b.flat[idx]
    return np.allclose(a, phase * b, atol=tol)


def _resynth_2q_window_7(circ: QuantumCircuit, window: int = 7) -> Tuple[QuantumCircuit, RewriteResult]:
    ops = _flatten_ops(circ)
    n = len(ops)
    max_len = max(2, int(window))

    from qiskit.circuit.library import CXGate, CZGate, CYGate, SwapGate
    from qiskit.circuit.library import ECRGate, iSwapGate, RXXGate, RYYGate, RZZGate
    from qiskit.quantum_info import Operator

    base_gates = [
        ("cx", CXGate()),
        ("cz", CZGate()),
        ("cy", CYGate()),
        ("swap", SwapGate()),
        ("iswap", iSwapGate()),
        ("ecr", ECRGate()),
        ("rxx_pi_2", RXXGate(math.pi / 2.0)),
        ("ryy_pi_2", RYYGate(math.pi / 2.0)),
        ("rzz_pi_2", RZZGate(math.pi / 2.0)),
    ]

    candidate_ops: List[Tuple[str, List[Instruction]]] = [("identity", [])]
    for name, gate in base_gates:
        candidate_ops.append((name, [gate]))
    max_candidates = 2500

    def _should_prune(names: List[str]) -> bool:
        # Skip long runs of the same gate to limit search.
        run = 1
        for i in range(1, len(names)):
            if names[i] == names[i - 1]:
                run += 1
                if run >= 3:
                    return True
            else:
                run = 1
        return False

    for n1, g1 in base_gates:
        for n2, g2 in base_gates:
            if _should_prune([n1, n2]):
                continue
            candidate_ops.append((f"{n1}_{n2}", [g1, g2]))
            if len(candidate_ops) >= max_candidates:
                break
        if len(candidate_ops) >= max_candidates:
            break

    for n1, g1 in base_gates:
        if len(candidate_ops) >= max_candidates:
            break
        for n2, g2 in base_gates:
            if len(candidate_ops) >= max_candidates:
                break
            for n3, g3 in base_gates:
                if _should_prune([n1, n2, n3]):
                    continue
                candidate_ops.append((f"{n1}_{n2}_{n3}", [g1, g2, g3]))
                if len(candidate_ops) >= max_candidates:
                    break

    for n1, g1 in base_gates:
        if len(candidate_ops) >= max_candidates:
            break
        for n2, g2 in base_gates:
            if len(candidate_ops) >= max_candidates:
                break
            for n3, g3 in base_gates:
                if len(candidate_ops) >= max_candidates:
                    break
                for n4, g4 in base_gates:
                    if _should_prune([n1, n2, n3, n4]):
                        continue
                    candidate_ops.append((f"{n1}_{n2}_{n3}_{n4}", [g1, g2, g3, g4]))
                    if len(candidate_ops) >= max_candidates:
                        break

    for n1, g1 in base_gates:
        if len(candidate_ops) >= max_candidates:
            break
        for n2, g2 in base_gates:
            if len(candidate_ops) >= max_candidates:
                break
            for n3, g3 in base_gates:
                if len(candidate_ops) >= max_candidates:
                    break
                for n4, g4 in base_gates:
                    if len(candidate_ops) >= max_candidates:
                        break
                    for n5, g5 in base_gates:
                        if _should_prune([n1, n2, n3, n4, n5]):
                            continue
                        candidate_ops.append((f"{n1}_{n2}_{n3}_{n4}_{n5}", [g1, g2, g3, g4, g5]))
                        if len(candidate_ops) >= max_candidates:
                            break

    for i in range(n):
        for j in range(i + 1, min(n, i + max_len)):
            window_ops = ops[i : j + 1]
            qubits: List[Qubit] = []
            for op in window_ops:
                for q in op.qargs:
                    if q not in qubits:
                        qubits.append(q)
                if len(qubits) > 2:
                    break
            if len(qubits) == 0 or len(qubits) > 2:
                continue

            sub = QuantumCircuit(2)
            qmap = {qubits[0]: 0}
            if len(qubits) == 2:
                qmap[qubits[1]] = 1
            for op in window_ops:
                if op.name in ("measure", "barrier"):
                    break
                sub.append(op.inst, [qmap[q] for q in op.qargs], [])
            else:
                U = Operator(sub).data
                best = None
                for name, gates in candidate_ops:
                    cand = QuantumCircuit(2)
                    if name != "identity":
                        for gate in gates:
                            if gate.num_qubits == 2:
                                cand.append(gate, [0, 1])
                            else:
                                cand.append(gate, [0])
                    V = Operator(cand).data
                    if _equal_up_to_global_phase(U, V):
                        best = (name, gates)
                        break

                if best is None:
                    continue
                name, gates = best
                if len(gates) >= len(window_ops):
                    continue

                if name == "identity":
                    new_ops = ops[:i] + ops[j + 1 :]
                else:
                    replacement: List[Op] = []
                    valid_replacement = True
                    for inst in gates:
                        if inst.num_qubits < 1 or inst.num_qubits > len(qubits):
                            valid_replacement = False
                            break
                        replacement.append(
                            Op(
                                inst=inst,
                                qargs=tuple(qubits[: inst.num_qubits]),
                                cargs=(),
                            )
                        )
                    if not valid_replacement:
                        continue
                    new_ops = ops[:i] + replacement + ops[j + 1 :]

                new_circ = _rebuild_from_ops(circ, new_ops)
                res = RewriteResult(
                    changed=True,
                    action_id=ACTION_RESYNTH_2Q_WINDOW_7,
                    action_name="resynth_2q_window_7",
                    message=f"Resynthesized 2Q window {i}-{j} to {name}.",
                    old_len=n,
                    new_len=len(new_ops),
                    window=(i, j),
                )
                return new_circ, res

    res = RewriteResult(
        changed=False,
        action_id=ACTION_RESYNTH_2Q_WINDOW_7,
        action_name="resynth_2q_window_7",
        message="No resynthesizable 2Q window found.",
        old_len=n,
        new_len=n,
        window=None,
    )
    return circ.copy(), res


def _resynth_3q_window_7(circ: QuantumCircuit, window: int = 7) -> Tuple[QuantumCircuit, RewriteResult]:
    ops = _flatten_ops(circ)
    n = len(ops)
    max_len = max(2, int(window))

    from qiskit.circuit.library import CCXGate, CCZGate, CSwapGate, RCCXGate
    from qiskit.quantum_info import Operator
    candidates = [
        ("ccx", CCXGate(), 3),
        ("ccz", CCZGate(), 3),
        ("cswap", CSwapGate(), 3),
        ("rccx", RCCXGate(), 3),
    ]

    for i in range(n):
        for j in range(i + 1, min(n, i + max_len)):
            window_ops = ops[i : j + 1]
            qubits: List[Qubit] = []
            for op in window_ops:
                for q in op.qargs:
                    if q not in qubits:
                        qubits.append(q)
                if len(qubits) > 3:
                    break
            if len(qubits) != 3:
                continue

            sub = QuantumCircuit(3)
            qmap = {qubits[0]: 0, qubits[1]: 1, qubits[2]: 2}
            for op in window_ops:
                if op.name in ("measure", "barrier"):
                    break
                sub.append(op.inst, [qmap[q] for q in op.qargs], [])
            else:
                U = Operator(sub).data
                match = None
                for name, gate, nqubits in candidates:
                    for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
                        cand = QuantumCircuit(3)
                        cand.append(gate, list(perm[:nqubits]))
                        V = Operator(cand).data
                        if _equal_up_to_global_phase(U, V):
                            match = (name, gate, perm)
                            break
                    if match is not None:
                        break
                if match is None:
                    continue
                name, gate, perm = match
                if 1 >= len(window_ops):
                    continue

                qargs = (qubits[perm[0]], qubits[perm[1]], qubits[perm[2]])
                new_ops = ops[:i] + [Op(inst=gate, qargs=qargs, cargs=())] + ops[j + 1 :]
                new_circ = _rebuild_from_ops(circ, new_ops)
                res = RewriteResult(
                    changed=True,
                    action_id=ACTION_RESYNTH_3Q_WINDOW_5,
                    action_name="resynth_3q_window_7",
                    message=f"Resynthesized 3Q window {i}-{j} to {name}.",
                    old_len=n,
                    new_len=len(new_ops),
                    window=(i, j),
                )
                return new_circ, res

    res = RewriteResult(
        changed=False,
        action_id=ACTION_RESYNTH_3Q_WINDOW_5,
        action_name="resynth_3q_window_7",
        message="No resynthesizable 3Q window found.",
        old_len=n,
        new_len=n,
        window=None,
    )
    return circ.copy(), res


def _route_cx_with_swap(circ: QuantumCircuit) -> Tuple[QuantumCircuit, RewriteResult]:
    ops = _flatten_ops(circ)
    n = len(ops)
    if _COUPLING_MAP is None:
        res = RewriteResult(
            changed=False,
            action_id=ACTION_ROUTE_CX_WITH_SWAP,
            action_name="route_cx_with_swap",
            message="No coupling map set.",
            old_len=n,
            new_len=n,
            window=None,
        )
        return circ.copy(), res

    qmap = _qubit_index_map(circ)
    from qiskit.circuit.library import SwapGate, CXGate

    for i in range(n):
        op = ops[i]
        if op.name != "cx" or len(op.qargs) != 2:
            continue
        ctrl, tgt = op.qargs
        ci = qmap[ctrl]
        ti = qmap[tgt]
        if _is_coupled(ci, ti, directed_ok=True):
            continue
        # Find an intermediate qubit k such that ci-k and k-ti are coupled.
        for k in range(len(circ.qubits)):
            if k in (ci, ti):
                continue
            if _is_coupled(ci, k, directed_ok=True) and _is_coupled(k, ti, directed_ok=True):
                qk = circ.qubits[k]
                new_ops = (
                    ops[:i]
                    + [Op(inst=SwapGate(), qargs=(qk, tgt), cargs=())]
                    + [Op(inst=CXGate(), qargs=(ctrl, qk), cargs=())]
                    + [Op(inst=SwapGate(), qargs=(qk, tgt), cargs=())]
                    + ops[i + 1 :]
                )
                new_circ = _rebuild_from_ops(circ, new_ops)
                res = RewriteResult(
                    changed=True,
                    action_id=ACTION_ROUTE_CX_WITH_SWAP,
                    action_name="route_cx_with_swap",
                    message=f"Routed CX via SWAP using intermediate qubit {k}.",
                    old_len=n,
                    new_len=len(new_ops),
                    window=(i, i),
                )
                return new_circ, res

    res = RewriteResult(
        changed=False,
        action_id=ACTION_ROUTE_CX_WITH_SWAP,
        action_name="route_cx_with_swap",
        message="No CX found requiring SWAP routing.",
        old_len=n,
        new_len=n,
        window=None,
    )
    return circ.copy(), res


def _unroute_cx_with_swap(circ: QuantumCircuit) -> Tuple[QuantumCircuit, RewriteResult]:
    ops = _flatten_ops(circ)
    n = len(ops)
    if _COUPLING_MAP is None:
        res = RewriteResult(
            changed=False,
            action_id=ACTION_UNROUTE_CX_WITH_SWAP,
            action_name="unroute_cx_with_swap",
            message="No coupling map set.",
            old_len=n,
            new_len=n,
            window=None,
        )
        return circ.copy(), res

    qmap = _qubit_index_map(circ)
    from qiskit.circuit.library import CXGate

    for i in range(n - 2):
        a, b, c = ops[i], ops[i + 1], ops[i + 2]
        if a.name != "swap" or c.name != "swap" or b.name != "cx":
            continue
        if a.qargs != c.qargs:
            continue
        swap_a, swap_b = a.qargs
        ctrl, tgt = b.qargs
        if tgt != swap_a and tgt != swap_b:
            continue
        # If CX target is the swapped qubit, unroute to the other end.
        other = swap_b if tgt == swap_a else swap_a
        ci = qmap[ctrl]
        oi = qmap[other]
        if not _is_coupled(ci, oi, directed_ok=True):
            continue
        new_ops = ops[:i] + [Op(inst=CXGate(), qargs=(ctrl, other), cargs=())] + ops[i + 3 :]
        new_circ = _rebuild_from_ops(circ, new_ops)
        res = RewriteResult(
            changed=True,
            action_id=ACTION_UNROUTE_CX_WITH_SWAP,
            action_name="unroute_cx_with_swap",
            message=f"Removed SWAP routing and restored direct CX at positions {i}-{i+2}.",
            old_len=n,
            new_len=len(new_ops),
            window=(i, i + 2),
        )
        return new_circ, res

    res = RewriteResult(
        changed=False,
        action_id=ACTION_UNROUTE_CX_WITH_SWAP,
        action_name="unroute_cx_with_swap",
        message="No SWAP-CX-SWAP pattern found to unroute.",
        old_len=n,
        new_len=n,
        window=None,
    )
    return circ.copy(), res


def _depth_aware_commute(circ: QuantumCircuit, window: int = 7) -> Tuple[QuantumCircuit, RewriteResult]:
    ops = _flatten_ops(circ)
    n = len(ops)
    max_len = max(2, int(window))

    def _local_depth(seq: Sequence[Op]) -> int:
        sub = QuantumCircuit(*circ.qregs, *circ.cregs)
        for op in seq:
            sub.append(op.inst, op.qargs, op.cargs)
        try:
            return int(sub.depth())
        except Exception:
            return 0

    for i in range(n - 1):
        a, b = ops[i], ops[i + 1]
        if not _commute(a, b):
            continue
        start = max(0, i - (max_len // 2))
        end = min(n, start + max_len)
        window_ops = ops[start:end]
        before = _local_depth(window_ops)
        swapped = window_ops.copy()
        local_i = i - start
        swapped[local_i], swapped[local_i + 1] = swapped[local_i + 1], swapped[local_i]
        after = _local_depth(swapped)
        if after < before:
            new_ops = ops[:i] + [b, a] + ops[i + 2 :]
            new_circ = _rebuild_from_ops(circ, new_ops)
            res = RewriteResult(
                changed=True,
                action_id=ACTION_DEPTH_AWARE_COMMUTE,
                action_name="depth_aware_commute",
                message=f"Swapped commuting ops at positions {i},{i+1} to reduce depth {before}->{after}.",
                old_len=n,
                new_len=n,
                window=(i, i + 1),
            )
            return new_circ, res

    res = RewriteResult(
        changed=False,
        action_id=ACTION_DEPTH_AWARE_COMMUTE,
        action_name="depth_aware_commute",
        message="No commuting swap reduced local depth.",
        old_len=n,
        new_len=n,
        window=None,
    )
    return circ.copy(), res


def _route_cz_cy_with_swap(circ: QuantumCircuit) -> Tuple[QuantumCircuit, RewriteResult]:
    ops = _flatten_ops(circ)
    n = len(ops)
    if _COUPLING_MAP is None:
        res = RewriteResult(
            changed=False,
            action_id=ACTION_ROUTE_CZ_CY_WITH_SWAP,
            action_name="route_cz_cy_with_swap",
            message="No coupling map set.",
            old_len=n,
            new_len=n,
            window=None,
        )
        return circ.copy(), res

    qmap = _qubit_index_map(circ)
    from qiskit.circuit.library import SwapGate, CZGate, CYGate

    for i in range(n):
        op = ops[i]
        if op.name not in ("cz", "cy") or len(op.qargs) != 2:
            continue
        qa, qb = op.qargs
        ai = qmap[qa]
        bi = qmap[qb]
        if _is_coupled(ai, bi):
            continue
        for k in range(len(circ.qubits)):
            if k in (ai, bi):
                continue
            if _is_coupled(ai, k) and _is_coupled(k, bi):
                qk = circ.qubits[k]
                gate = CZGate() if op.name == "cz" else CYGate()
                new_ops = (
                    ops[:i]
                    + [Op(inst=SwapGate(), qargs=(qk, qb), cargs=())]
                    + [Op(inst=gate, qargs=(qa, qk), cargs=())]
                    + [Op(inst=SwapGate(), qargs=(qk, qb), cargs=())]
                    + ops[i + 1 :]
                )
                new_circ = _rebuild_from_ops(circ, new_ops)
                res = RewriteResult(
                    changed=True,
                    action_id=ACTION_ROUTE_CZ_CY_WITH_SWAP,
                    action_name="route_cz_cy_with_swap",
                    message=f"Routed {op.name.upper()} via SWAP using intermediate qubit {k}.",
                    old_len=n,
                    new_len=len(new_ops),
                    window=(i, i),
                )
                return new_circ, res

    res = RewriteResult(
        changed=False,
        action_id=ACTION_ROUTE_CZ_CY_WITH_SWAP,
        action_name="route_cz_cy_with_swap",
        message="No CZ/CY found requiring SWAP routing.",
        old_len=n,
        new_len=n,
        window=None,
    )
    return circ.copy(), res


def _depth_aware_commute_global(circ: QuantumCircuit, window: int = 15) -> Tuple[QuantumCircuit, RewriteResult]:
    ops = _flatten_ops(circ)
    n = len(ops)
    max_len = max(2, int(window))

    def _local_depth(seq: Sequence[Op]) -> int:
        sub = QuantumCircuit(*circ.qregs, *circ.cregs)
        for op in seq:
            sub.append(op.inst, op.qargs, op.cargs)
        try:
            return int(sub.depth())
        except Exception:
            return 0

    best = None
    for i in range(n - 1):
        a, b = ops[i], ops[i + 1]
        if not _commute(a, b):
            continue
        start = max(0, i - (max_len // 2))
        end = min(n, start + max_len)
        window_ops = ops[start:end]
        before = _local_depth(window_ops)
        swapped = window_ops.copy()
        local_i = i - start
        swapped[local_i], swapped[local_i + 1] = swapped[local_i + 1], swapped[local_i]
        after = _local_depth(swapped)
        if after < before:
            improvement = before - after
            if best is None or improvement > best[0]:
                best = (improvement, i, before, after)

    if best is not None:
        _, i, before, after = best
        a, b = ops[i], ops[i + 1]
        new_ops = ops[:i] + [b, a] + ops[i + 2 :]
        new_circ = _rebuild_from_ops(circ, new_ops)
        res = RewriteResult(
            changed=True,
            action_id=ACTION_DEPTH_AWARE_COMMUTE_GLOBAL,
            action_name="depth_aware_commute_global",
            message=f"Swapped commuting ops at positions {i},{i+1} to reduce depth {before}->{after}.",
            old_len=n,
            new_len=n,
            window=(i, i + 1),
        )
        return new_circ, res

    res = RewriteResult(
        changed=False,
        action_id=ACTION_DEPTH_AWARE_COMMUTE_GLOBAL,
        action_name="depth_aware_commute_global",
        message="No commuting swap reduced depth in global search.",
        old_len=n,
        new_len=n,
        window=None,
    )
    return circ.copy(), res


def _depth_aware_multi_swap(circ: QuantumCircuit, window: int = 15, max_swaps: int = 3) -> Tuple[QuantumCircuit, RewriteResult]:
    ops = _flatten_ops(circ)
    n = len(ops)
    max_len = max(2, int(window))

    def _local_depth(seq: Sequence[Op]) -> int:
        sub = QuantumCircuit(*circ.qregs, *circ.cregs)
        for op in seq:
            sub.append(op.inst, op.qargs, op.cargs)
        try:
            return int(sub.depth())
        except Exception:
            return 0

    best = None
    for start in range(0, n):
        end = min(n, start + max_len)
        window_ops = ops[start:end]
        if len(window_ops) < 2:
            continue
        base_depth = _local_depth(window_ops)
        swaps_made = []
        candidate = window_ops.copy()
        for _ in range(max_swaps):
            improved = False
            for i in range(len(candidate) - 1):
                if not _commute(candidate[i], candidate[i + 1]):
                    continue
                trial = candidate.copy()
                trial[i], trial[i + 1] = trial[i + 1], trial[i]
                if _local_depth(trial) < _local_depth(candidate):
                    candidate = trial
                    swaps_made.append((start + i, start + i + 1))
                    improved = True
                    break
            if not improved:
                break
        new_depth = _local_depth(candidate)
        if new_depth < base_depth and swaps_made:
            improvement = base_depth - new_depth
            if best is None or improvement > best[0]:
                best = (improvement, start, end, swaps_made, base_depth, new_depth)

    if best is not None:
        _, start, end, swaps_made, before, after = best
        new_ops = ops.copy()
        for i, j in swaps_made:
            new_ops[i], new_ops[j] = new_ops[j], new_ops[i]
        new_circ = _rebuild_from_ops(circ, new_ops)
        res = RewriteResult(
            changed=True,
            action_id=ACTION_DEPTH_AWARE_MULTI_SWAP,
            action_name="depth_aware_multi_swap",
            message=f"Applied {len(swaps_made)} commuting swaps to reduce depth {before}->{after}.",
            old_len=n,
            new_len=n,
            window=(start, end - 1),
        )
        return new_circ, res

    res = RewriteResult(
        changed=False,
        action_id=ACTION_DEPTH_AWARE_MULTI_SWAP,
        action_name="depth_aware_multi_swap",
        message="No multi-swap sequence reduced depth.",
        old_len=n,
        new_len=n,
        window=None,
    )
    return circ.copy(), res


def _fuse_1q_chain_7(circ: QuantumCircuit, window: int = 7) -> Tuple[QuantumCircuit, RewriteResult]:
    ops = _flatten_ops(circ)
    n = len(ops)
    max_len = max(2, int(window))

    from qiskit.quantum_info import Operator
    from qiskit.circuit.library import UnitaryGate

    for i in range(n):
        if len(ops[i].qargs) != 1:
            continue
        target = ops[i].qargs[0]
        end = i
        while end < n and (end - i) < max_len:
            op = ops[end]
            if len(op.qargs) != 1 or op.qargs[0] != target:
                break
            end += 1

        if end - i < 2:
            continue
        chain = ops[i:end]
        sub = QuantumCircuit(1)
        for op in chain:
            if op.name in ("measure", "barrier"):
                break
            sub.append(op.inst, [0], [])
        else:
            U = Operator(sub).data
            new_ops = ops[:i] + [Op(inst=UnitaryGate(U), qargs=(target,), cargs=())] + ops[end:]
            new_circ = _rebuild_from_ops(circ, new_ops)
            res = RewriteResult(
                changed=True,
                action_id=ACTION_FUSE_1Q_CHAIN_7,
                action_name="fuse_1q_chain_7",
                message=f"Fused 1Q chain at positions {i}-{end - 1}.",
                old_len=n,
                new_len=len(new_ops),
                window=(i, end - 1),
            )
            return new_circ, res

    res = RewriteResult(
        changed=False,
        action_id=ACTION_FUSE_1Q_CHAIN_7,
        action_name="fuse_1q_chain_7",
        message="No 1Q chain found to fuse.",
        old_len=n,
        new_len=n,
        window=None,
    )
    return circ.copy(), res


def _route_ecr_iswap_rzz_with_swap(circ: QuantumCircuit) -> Tuple[QuantumCircuit, RewriteResult]:
    ops = _flatten_ops(circ)
    n = len(ops)
    if _COUPLING_MAP is None:
        res = RewriteResult(
            changed=False,
            action_id=ACTION_ROUTE_ECR_ISWAP_RZZ_WITH_SWAP,
            action_name="route_ecr_iswap_rzz_with_swap",
            message="No coupling map set.",
            old_len=n,
            new_len=n,
            window=None,
        )
        return circ.copy(), res

    qmap = _qubit_index_map(circ)
    from qiskit.circuit.library import SwapGate, ECRGate, iSwapGate, RZZGate

    for i in range(n):
        op = ops[i]
        if op.name not in ("ecr", "iswap", "rzz") or len(op.qargs) != 2:
            continue
        qa, qb = op.qargs
        ai = qmap[qa]
        bi = qmap[qb]
        directed = op.name == "ecr"
        if _is_coupled(ai, bi, directed_ok=directed):
            continue
        for k in range(len(circ.qubits)):
            if k in (ai, bi):
                continue
            if _is_coupled(ai, k, directed_ok=directed) and _is_coupled(k, bi, directed_ok=directed):
                qk = circ.qubits[k]
                if op.name == "ecr":
                    gate = ECRGate()
                elif op.name == "iswap":
                    gate = iSwapGate()
                else:
                    gate = RZZGate(float(op.params[0]) if op.params else 0.0)
                new_ops = (
                    ops[:i]
                    + [Op(inst=SwapGate(), qargs=(qk, qb), cargs=())]
                    + [Op(inst=gate, qargs=(qa, qk), cargs=())]
                    + [Op(inst=SwapGate(), qargs=(qk, qb), cargs=())]
                    + ops[i + 1 :]
                )
                new_circ = _rebuild_from_ops(circ, new_ops)
                res = RewriteResult(
                    changed=True,
                    action_id=ACTION_ROUTE_ECR_ISWAP_RZZ_WITH_SWAP,
                    action_name="route_ecr_iswap_rzz_with_swap",
                    message=f"Routed {op.name.upper()} via SWAP using intermediate qubit {k}.",
                    old_len=n,
                    new_len=len(new_ops),
                    window=(i, i),
                )
                return new_circ, res

    res = RewriteResult(
        changed=False,
        action_id=ACTION_ROUTE_ECR_ISWAP_RZZ_WITH_SWAP,
        action_name="route_ecr_iswap_rzz_with_swap",
        message="No ECR/iSWAP/RZZ found requiring SWAP routing.",
        old_len=n,
        new_len=n,
        window=None,
    )
    return circ.copy(), res


def _route_rxx_ryy_rzz_with_swap(circ: QuantumCircuit) -> Tuple[QuantumCircuit, RewriteResult]:
    ops = _flatten_ops(circ)
    n = len(ops)
    if _COUPLING_MAP is None:
        res = RewriteResult(
            changed=False,
            action_id=ACTION_ROUTE_RXX_RYY_RZZ_WITH_SWAP,
            action_name="route_rxx_ryy_rzz_with_swap",
            message="No coupling map set.",
            old_len=n,
            new_len=n,
            window=None,
        )
        return circ.copy(), res

    qmap = _qubit_index_map(circ)
    from qiskit.circuit.library import SwapGate, RXXGate, RYYGate, RZZGate

    for i in range(n):
        op = ops[i]
        if op.name not in ("rxx", "ryy", "rzz") or len(op.qargs) != 2:
            continue
        qa, qb = op.qargs
        ai = qmap[qa]
        bi = qmap[qb]
        if _is_coupled(ai, bi):
            continue
        for k in range(len(circ.qubits)):
            if k in (ai, bi):
                continue
            if _is_coupled(ai, k) and _is_coupled(k, bi):
                qk = circ.qubits[k]
                theta = float(op.params[0]) if op.params else 0.0
                if op.name == "rxx":
                    gate = RXXGate(theta)
                elif op.name == "ryy":
                    gate = RYYGate(theta)
                else:
                    gate = RZZGate(theta)
                new_ops = (
                    ops[:i]
                    + [Op(inst=SwapGate(), qargs=(qk, qb), cargs=())]
                    + [Op(inst=gate, qargs=(qa, qk), cargs=())]
                    + [Op(inst=SwapGate(), qargs=(qk, qb), cargs=())]
                    + ops[i + 1 :]
                )
                new_circ = _rebuild_from_ops(circ, new_ops)
                res = RewriteResult(
                    changed=True,
                    action_id=ACTION_ROUTE_RXX_RYY_RZZ_WITH_SWAP,
                    action_name="route_rxx_ryy_rzz_with_swap",
                    message=f"Routed {op.name.upper()} via SWAP using intermediate qubit {k}.",
                    old_len=n,
                    new_len=len(new_ops),
                    window=(i, i),
                )
                return new_circ, res

    res = RewriteResult(
        changed=False,
        action_id=ACTION_ROUTE_RXX_RYY_RZZ_WITH_SWAP,
        action_name="route_rxx_ryy_rzz_with_swap",
        message="No RXX/RYY/RZZ found requiring SWAP routing.",
        old_len=n,
        new_len=n,
        window=None,
    )
    return circ.copy(), res


def _depth_aware_multi_swap_global(circ: QuantumCircuit, window: int = 25, max_swaps: int = 5) -> Tuple[QuantumCircuit, RewriteResult]:
    return _depth_aware_multi_swap(circ, window=window, max_swaps=max_swaps)


def _depth_aware_multi_swap_full(circ: QuantumCircuit, max_swaps: int = 8) -> Tuple[QuantumCircuit, RewriteResult]:
    ops = _flatten_ops(circ)
    n = len(ops)

    def _depth_from_ops(seq: Sequence[Op]) -> int:
        sub = QuantumCircuit(*circ.qregs, *circ.cregs)
        for op in seq:
            sub.append(op.inst, op.qargs, op.cargs)
        try:
            return int(sub.depth())
        except Exception:
            return 0

    current = ops.copy()
    swaps_made: List[Tuple[int, int]] = []
    base_depth = _depth_from_ops(current)

    for _ in range(max_swaps):
        best = None
        for i in range(n - 1):
            if not _commute(current[i], current[i + 1]):
                continue
            trial = current.copy()
            trial[i], trial[i + 1] = trial[i + 1], trial[i]
            d = _depth_from_ops(trial)
            if d < base_depth and (best is None or d < best[0]):
                best = (d, i, trial)
        if best is None:
            break
        new_depth, idx, new_ops = best
        current = new_ops
        swaps_made.append((idx, idx + 1))
        base_depth = new_depth

    if swaps_made:
        new_circ = _rebuild_from_ops(circ, current)
        res = RewriteResult(
            changed=True,
            action_id=ACTION_DEPTH_AWARE_MULTI_SWAP_FULL,
            action_name="depth_aware_multi_swap_full",
            message=f"Applied {len(swaps_made)} swaps to reduce depth to {base_depth}.",
            old_len=n,
            new_len=len(current),
            window=(0, n - 1),
        )
        return new_circ, res

    res = RewriteResult(
        changed=False,
        action_id=ACTION_DEPTH_AWARE_MULTI_SWAP_FULL,
        action_name="depth_aware_multi_swap_full",
        message="No global multi-swap reduced depth.",
        old_len=n,
        new_len=n,
        window=None,
    )
    return circ.copy(), res


def _fuse_multi_q_unitary(circ: QuantumCircuit, max_qubits: int = 3, max_ops: int = 5) -> Tuple[QuantumCircuit, RewriteResult]:
    ops = _flatten_ops(circ)
    n = len(ops)

    from qiskit.quantum_info import Operator
    from qiskit.circuit.library import UnitaryGate

    for i in range(n):
        for j in range(i + 1, min(n, i + max_ops)):
            window_ops = ops[i : j + 1]
            qubits: List[Qubit] = []
            for op in window_ops:
                for q in op.qargs:
                    if q not in qubits:
                        qubits.append(q)
                if len(qubits) > max_qubits:
                    break
            if len(qubits) < 2 or len(qubits) > max_qubits:
                continue
            if len(window_ops) <= 1:
                continue

            sub = QuantumCircuit(len(qubits))
            qmap = {q: idx for idx, q in enumerate(qubits)}
            for op in window_ops:
                if op.name in ("measure", "barrier"):
                    break
                sub.append(op.inst, [qmap[q] for q in op.qargs], [])
            else:
                U = Operator(sub).data
                gate = UnitaryGate(U)
                if 1 >= len(window_ops):
                    continue
                new_ops = ops[:i] + [Op(inst=gate, qargs=tuple(qubits), cargs=())] + ops[j + 1 :]
                new_circ = _rebuild_from_ops(circ, new_ops)
                res = RewriteResult(
                    changed=True,
                    action_id=ACTION_FUSE_MULTI_Q_UNITARY,
                    action_name="fuse_multi_q_unitary",
                    message=f"Fused {len(window_ops)}-op window {i}-{j} into {len(qubits)}Q unitary.",
                    old_len=n,
                    new_len=len(new_ops),
                    window=(i, j),
                )
                return new_circ, res

    res = RewriteResult(
        changed=False,
        action_id=ACTION_FUSE_MULTI_Q_UNITARY,
        action_name="fuse_multi_q_unitary",
        message="No multi-qubit window found to fuse.",
        old_len=n,
        new_len=n,
        window=None,
    )
    return circ.copy(), res


def _resynth_4q_window_7(circ: QuantumCircuit, window: int = 7) -> Tuple[QuantumCircuit, RewriteResult]:
    ops = _flatten_ops(circ)
    n = len(ops)
    max_len = max(2, int(window))

    from qiskit.circuit.library import C3XGate, RC3XGate
    from qiskit.quantum_info import Operator

    candidates = [
        ("c3x", C3XGate(), 4),
        ("rc3x", RC3XGate(), 4),
    ]

    for i in range(n):
        for j in range(i + 1, min(n, i + max_len)):
            window_ops = ops[i : j + 1]
            qubits: List[Qubit] = []
            for op in window_ops:
                for q in op.qargs:
                    if q not in qubits:
                        qubits.append(q)
                if len(qubits) > 4:
                    break
            if len(qubits) != 4:
                continue

            sub = QuantumCircuit(4)
            qmap = {qubits[0]: 0, qubits[1]: 1, qubits[2]: 2, qubits[3]: 3}
            for op in window_ops:
                if op.name in ("measure", "barrier"):
                    break
                sub.append(op.inst, [qmap[q] for q in op.qargs], [])
            else:
                U = Operator(sub).data
                match = None
                perms = [
                    (0, 1, 2, 3),
                    (0, 1, 3, 2),
                    (0, 2, 1, 3),
                    (0, 2, 3, 1),
                    (0, 3, 1, 2),
                    (0, 3, 2, 1),
                    (1, 0, 2, 3),
                    (1, 0, 3, 2),
                    (1, 2, 0, 3),
                    (1, 2, 3, 0),
                    (1, 3, 0, 2),
                    (1, 3, 2, 0),
                    (2, 0, 1, 3),
                    (2, 0, 3, 1),
                    (2, 1, 0, 3),
                    (2, 1, 3, 0),
                    (2, 3, 0, 1),
                    (2, 3, 1, 0),
                    (3, 0, 1, 2),
                    (3, 0, 2, 1),
                    (3, 1, 0, 2),
                    (3, 1, 2, 0),
                    (3, 2, 0, 1),
                    (3, 2, 1, 0),
                ]
                for name, gate, nqubits in candidates:
                    for perm in perms:
                        cand = QuantumCircuit(4)
                        cand.append(gate, list(perm[:nqubits]))
                        V = Operator(cand).data
                        if _equal_up_to_global_phase(U, V):
                            match = (name, gate, perm)
                            break
                    if match is not None:
                        break
                if match is None:
                    continue
                name, gate, perm = match
                if 1 >= len(window_ops):
                    continue

                qargs = (qubits[perm[0]], qubits[perm[1]], qubits[perm[2]], qubits[perm[3]])
                new_ops = ops[:i] + [Op(inst=gate, qargs=qargs, cargs=())] + ops[j + 1 :]
                new_circ = _rebuild_from_ops(circ, new_ops)
                res = RewriteResult(
                    changed=True,
                    action_id=ACTION_RESYNTH_4Q_WINDOW_7,
                    action_name="resynth_4q_window_7",
                    message=f"Resynthesized 4Q window {i}-{j} to {name}.",
                    old_len=n,
                    new_len=len(new_ops),
                    window=(i, j),
                )
                return new_circ, res

    res = RewriteResult(
        changed=False,
        action_id=ACTION_RESYNTH_4Q_WINDOW_7,
        action_name="resynth_4q_window_7",
        message="No resynthesizable 4Q window found.",
        old_len=n,
        new_len=n,
        window=None,
    )
    return circ.copy(), res


def _route_directed_more(circ: QuantumCircuit) -> Tuple[QuantumCircuit, RewriteResult]:
    ops = _flatten_ops(circ)
    n = len(ops)
    if _COUPLING_MAP is None:
        res = RewriteResult(
            changed=False,
            action_id=ACTION_ROUTE_DIRECTED_MORE,
            action_name="route_directed_more",
            message="No coupling map set.",
            old_len=n,
            new_len=n,
            window=None,
        )
        return circ.copy(), res

    qmap = _qubit_index_map(circ)
    from qiskit.circuit.library import SwapGate, CXGate, ECRGate

    for i in range(n):
        op = ops[i]
        if op.name not in ("cx", "ecr") or len(op.qargs) != 2:
            continue
        qa, qb = op.qargs
        ai = qmap[qa]
        bi = qmap[qb]
        if _is_coupled(ai, bi, directed_ok=True):
            continue
        for k in range(len(circ.qubits)):
            if k in (ai, bi):
                continue
            if _is_coupled(ai, k, directed_ok=True) and _is_coupled(k, bi, directed_ok=True):
                qk = circ.qubits[k]
                gate = CXGate() if op.name == "cx" else ECRGate()
                new_ops = (
                    ops[:i]
                    + [Op(inst=SwapGate(), qargs=(qk, qb), cargs=())]
                    + [Op(inst=gate, qargs=(qa, qk), cargs=())]
                    + [Op(inst=SwapGate(), qargs=(qk, qb), cargs=())]
                    + ops[i + 1 :]
                )
                new_circ = _rebuild_from_ops(circ, new_ops)
                res = RewriteResult(
                    changed=True,
                    action_id=ACTION_ROUTE_DIRECTED_MORE,
                    action_name="route_directed_more",
                    message=f"Routed {op.name.upper()} via directed SWAP using intermediate qubit {k}.",
                    old_len=n,
                    new_len=len(new_ops),
                    window=(i, i),
                )
                return new_circ, res

    res = RewriteResult(
        changed=False,
        action_id=ACTION_ROUTE_DIRECTED_MORE,
        action_name="route_directed_more",
        message="No directed CX/ECR found requiring SWAP routing.",
        old_len=n,
        new_len=n,
        window=None,
    )
    return circ.copy(), res


def _phase_angle_for_op(op: Op) -> Optional[float]:
    if op.name == "rz":
        return float(op.params[0])
    if op.name == "z":
        return math.pi
    if op.name == "s":
        return math.pi / 2.0
    if op.name == "sdg":
        return -math.pi / 2.0
    if op.name == "t":
        return math.pi / 4.0
    if op.name == "tdg":
        return -math.pi / 4.0
    return None


def _fuse_phase_chain_7(circ: QuantumCircuit, window: int = 7) -> Tuple[QuantumCircuit, RewriteResult]:
    ops = _flatten_ops(circ)
    n = len(ops)

    max_len = max(2, int(window))
    for i in range(n):
        if len(ops[i].qargs) != 1:
            continue
        target = ops[i].qargs[0]
        # Build the longest consecutive phase-only chain up to window length.
        angles: List[float] = []
        end = i
        while end < n and (end - i) < max_len:
            op = ops[end]
            if len(op.qargs) != 1 or op.qargs[0] != target:
                break
            angle = _phase_angle_for_op(op)
            if angle is None:
                break
            angles.append(angle)
            end += 1

        if len(angles) < 2:
            continue

        total = _wrap_angle(sum(angles))
        if _is_identity_rotation(total):
            new_ops = ops[:i] + ops[end:]
            msg = f"Fused phase chain into identity at positions {i},{end - 1}."
        else:
            from qiskit.circuit.library import RZGate
            new_ops = ops[:i] + [Op(inst=RZGate(total), qargs=(target,), cargs=())] + ops[end:]
            msg = f"Fused phase chain into RZ({total:.6g}) at positions {i},{end - 1}."

        new_circ = _rebuild_from_ops(circ, new_ops)
        res = RewriteResult(
            changed=True,
            action_id=ACTION_FUSE_PHASE_CHAIN_7,
            action_name="fuse_phase_chain_7",
            message=msg,
            old_len=n,
            new_len=len(new_ops),
            window=(i, end - 1),
        )
        return new_circ, res

    res = RewriteResult(
        changed=False,
        action_id=ACTION_FUSE_PHASE_CHAIN_7,
        action_name="fuse_phase_chain_7",
        message="No phase-only chain found within 7-gate window.",
        old_len=n,
        new_len=n,
        window=None,
    )
    return circ.copy(), res


def _conjugate_h_g_h(circ: QuantumCircuit) -> Tuple[QuantumCircuit, RewriteResult]:
    ops = _flatten_ops(circ)
    n = len(ops)

    for i in range(n - 2):
        a, b, c = ops[i], ops[i + 1], ops[i + 2]
        if a.name != "h" or c.name != "h":
            continue
        if len(a.qargs) != 1 or len(c.qargs) != 1:
            continue
        if a.qargs != b.qargs or a.qargs != c.qargs:
            continue
        if b.name not in ("x", "y", "z"):
            continue

        from qiskit.circuit.library import XGate, YGate, ZGate
        if b.name == "x":
            repl = XGate()
            repl_name = "z"
            repl = ZGate()
        elif b.name == "z":
            repl_name = "x"
            repl = XGate()
        else:
            repl_name = "y"
            repl = YGate()

        new_ops = ops[:i] + [Op(inst=repl, qargs=a.qargs, cargs=a.cargs)] + ops[i + 3 :]
        new_circ = _rebuild_from_ops(circ, new_ops)
        res = RewriteResult(
            changed=True,
            action_id=ACTION_CONJUGATE_H_G_H,
            action_name="conjugate_h_g_h",
            message=f"Rewrote H-{b.name.upper()}-H to {repl_name.upper()} at positions {i},{i+1},{i+2}.",
            old_len=n,
            new_len=len(new_ops),
            window=(i, i + 2),
        )
        return new_circ, res

    res = RewriteResult(
        changed=False,
        action_id=ACTION_CONJUGATE_H_G_H,
        action_name="conjugate_h_g_h",
        message="No H-G-H pattern found.",
        old_len=n,
        new_len=n,
        window=None,
    )
    return circ.copy(), res


def _conjugate_s_g_sdg(circ: QuantumCircuit) -> Tuple[QuantumCircuit, RewriteResult]:
    ops = _flatten_ops(circ)
    n = len(ops)

    for i in range(n - 2):
        a, b, c = ops[i], ops[i + 1], ops[i + 2]
        if (a.name, c.name) not in (("s", "sdg"), ("sdg", "s")):
            continue
        if len(a.qargs) != 1 or len(c.qargs) != 1:
            continue
        if a.qargs != b.qargs or a.qargs != c.qargs:
            continue
        if b.name not in ("x", "y"):
            continue

        from qiskit.circuit.library import XGate, YGate
        if b.name == "x":
            repl = YGate()
            repl_name = "y"
        else:
            repl = XGate()
            repl_name = "x"

        new_ops = ops[:i] + [Op(inst=repl, qargs=a.qargs, cargs=a.cargs)] + ops[i + 3 :]
        new_circ = _rebuild_from_ops(circ, new_ops)
        res = RewriteResult(
            changed=True,
            action_id=ACTION_CONJUGATE_S_G_SDG,
            action_name="conjugate_s_g_sdg",
            message=f"Rewrote {a.name.upper()}-{b.name.upper()}-{c.name.upper()} to {repl_name.upper()} at positions {i},{i+1},{i+2}.",
            old_len=n,
            new_len=len(new_ops),
            window=(i, i + 2),
        )
        return new_circ, res

    res = RewriteResult(
        changed=False,
        action_id=ACTION_CONJUGATE_S_G_SDG,
        action_name="conjugate_s_g_sdg",
        message="No S-G-Sdg pattern found.",
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
    ACTION_CANCEL_DOUBLE_H: _cancel_double_h,
    ACTION_CANCEL_DOUBLE_X: _cancel_double_x,
    ACTION_MERGE_ADJACENT_RX: _merge_adjacent_rx,
    ACTION_MERGE_ADJACENT_RY: _merge_adjacent_ry,
    ACTION_REMOVE_IDENTITY_RX: _remove_identity_rx,
    ACTION_REMOVE_IDENTITY_RY: _remove_identity_ry,
    ACTION_CANCEL_DOUBLE_Z: _cancel_double_z,
    ACTION_CANCEL_DOUBLE_Y: _cancel_double_y,
    ACTION_CANCEL_S_SDG: _cancel_s_sdg,
    ACTION_CANCEL_T_TDG: _cancel_t_tdg,
    ACTION_CANCEL_DOUBLE_CZ: _cancel_double_cz,
    ACTION_COMMUTE_RZ_THROUGH_CX_CONTROL: _commute_rz_through_cx_control,
    ACTION_CANCEL_CX_RZ_CX_CONTROL: _cancel_cx_rz_cx_control,
    ACTION_TOGGLE_CX_CZ_WITH_H: _toggle_cx_cz_with_h,
    ACTION_CANCEL_NONLOCAL_PAIR_12: _cancel_nonlocal_pair_12,
    ACTION_FUSE_PHASE_CHAIN_7: _fuse_phase_chain_7,
    ACTION_CONJUGATE_H_G_H: _conjugate_h_g_h,
    ACTION_CONJUGATE_S_G_SDG: _conjugate_s_g_sdg,
    ACTION_CANCEL_NONLOCAL_COMMUTE_12: _cancel_nonlocal_commute_12,
    ACTION_RESYNTH_2Q_WINDOW_7: _resynth_2q_window_7,
    ACTION_RESYNTH_3Q_WINDOW_5: _resynth_3q_window_7,
    ACTION_ROUTE_CX_WITH_SWAP: _route_cx_with_swap,
    ACTION_UNROUTE_CX_WITH_SWAP: _unroute_cx_with_swap,
    ACTION_DEPTH_AWARE_COMMUTE: _depth_aware_commute,
    ACTION_ROUTE_CZ_CY_WITH_SWAP: _route_cz_cy_with_swap,
    ACTION_DEPTH_AWARE_COMMUTE_GLOBAL: _depth_aware_commute_global,
    ACTION_DEPTH_AWARE_MULTI_SWAP: _depth_aware_multi_swap,
    ACTION_FUSE_1Q_CHAIN_7: _fuse_1q_chain_7,
    ACTION_ROUTE_ECR_ISWAP_RZZ_WITH_SWAP: _route_ecr_iswap_rzz_with_swap,
    ACTION_ROUTE_RXX_RYY_RZZ_WITH_SWAP: _route_rxx_ryy_rzz_with_swap,
    ACTION_DEPTH_AWARE_MULTI_SWAP_GLOBAL: _depth_aware_multi_swap_global,
    ACTION_DEPTH_AWARE_MULTI_SWAP_FULL: _depth_aware_multi_swap_full,
    ACTION_FUSE_MULTI_Q_UNITARY: _fuse_multi_q_unitary,
    ACTION_RESYNTH_4Q_WINDOW_7: _resynth_4q_window_7,
    ACTION_ROUTE_DIRECTED_MORE: _route_directed_more,
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


def applicable_action_mask(
    circuit: QuantumCircuit,
    *,
    allowed_action_ids: Optional[Set[int]] = None,
) -> List[int]:
    """
    Compute per-action applicability mask for the given circuit.

    Returns a binary list aligned with global action ids where:
    - 1 means action is currently applicable (would change the circuit),
    - 0 means no change (or action disallowed by allowed_action_ids).
    """
    actions = list_actions()
    if allowed_action_ids is None:
        allowed = {aid for aid, _ in actions}
    else:
        allowed = {int(aid) for aid in allowed_action_ids}

    mask: List[int] = [0] * len(actions)
    for aid, _ in actions:
        if aid not in allowed:
            mask[aid] = 0
            continue
        try:
            _, res = apply_action(circuit, aid)
            mask[aid] = 1 if res.changed else 0
        except Exception:
            mask[aid] = 0
    return mask


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
    def _self_test() -> None:
        """
        Lightweight edge-case checks for angle wraparound.
        """
        from qiskit.circuit import QuantumRegister

        qr = QuantumRegister(1, "q")

        # Merge to identity via wraparound: pi + pi = 2pi -> identity.
        qc = QuantumCircuit(qr)
        qc.rz(math.pi, qr[0])
        qc.rz(math.pi, qr[0])
        merged, res = apply_action(qc, ACTION_MERGE_ADJACENT_RZ)
        assert res.changed and len(merged.data) == 0, "Expected RZ(pi)+RZ(pi) to cancel."

        # Cancel inverse with wraparound: 3pi + pi = 4pi -> identity.
        qc = QuantumCircuit(qr)
        qc.rz(3.0 * math.pi, qr[0])
        qc.rz(1.0 * math.pi, qr[0])
        merged, res = apply_action(qc, ACTION_MERGE_ADJACENT_RZ)
        assert res.changed and len(merged.data) == 0, "Expected RZ(3pi)+RZ(pi) to cancel."

        # Cancel inverse using explicit negative angle.
        qc = QuantumCircuit(qr)
        qc.rz(0.9 * math.pi, qr[0])
        qc.rz(-0.9 * math.pi, qr[0])
        canceled, res = apply_action(qc, ACTION_CANCEL_INVERSE_RZ)
        assert res.changed and len(canceled.data) == 0, "Expected inverse RZ to cancel."

        # RX merge to identity via wraparound.
        qc = QuantumCircuit(qr)
        qc.rx(math.pi, qr[0])
        qc.rx(math.pi, qr[0])
        merged, res = apply_action(qc, ACTION_MERGE_ADJACENT_RX)
        assert res.changed and len(merged.data) == 0, "Expected RX(pi)+RX(pi) to cancel."

        print("Self-test passed.")

    _self_test()
    _quick_demo()
