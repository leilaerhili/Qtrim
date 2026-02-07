"""
Shared JSON schema for observations/actions between PC and phone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


CONSTRAINT_PROFILE_IDS = {
    "balanced": 0,
    "default": 0,
    "low_latency": 1,
    "high_fidelity": 2,
    "low_noise": 2,
    "min_cx": 3,
    "few_cx": 3,
    "low_cost": 4,
}

OBSERVATION_KEYS = (
    "gate_count",
    "depth",
    "num_cnot",
    "num_rz",
    "last_action_id",
    "constraint_profile",
)


def constraint_profile_to_id(profile: str) -> int:
    key = str(profile).strip().lower()
    return int(CONSTRAINT_PROFILE_IDS.get(key, 0))


@dataclass(frozen=True)
class PriorityWeights:
    two_qubit_gates: float = 0.50
    depth: float = 0.20
    total_gates: float = 0.10
    swap_gates: float = 0.20

    def to_json(self) -> Dict[str, float]:
        return {
            "two_qubit_gates": float(self.two_qubit_gates),
            "depth": float(self.depth),
            "total_gates": float(self.total_gates),
            "swap_gates": float(self.swap_gates),
        }

    @staticmethod
    def from_json(data: Dict[str, Any]) -> "PriorityWeights":
        return PriorityWeights(
            two_qubit_gates=float(data.get("two_qubit_gates", 0.0)),
            depth=float(data.get("depth", 0.0)),
            total_gates=float(data.get("total_gates", 0.0)),
            swap_gates=float(data.get("swap_gates", 0.0)),
        )


@dataclass(frozen=True)
class PriorityBudgets:
    max_depth: Optional[int] = None
    max_latency_ms: Optional[float] = None
    max_shots: Optional[int] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "max_depth": int(self.max_depth) if self.max_depth is not None else None,
            "max_latency_ms": (
                float(self.max_latency_ms) if self.max_latency_ms is not None else None
            ),
            "max_shots": int(self.max_shots) if self.max_shots is not None else None,
        }

    @staticmethod
    def from_json(data: Dict[str, Any]) -> "PriorityBudgets":
        return PriorityBudgets(
            max_depth=int(data["max_depth"]) if data.get("max_depth") is not None else None,
            max_latency_ms=(
                float(data["max_latency_ms"])
                if data.get("max_latency_ms") is not None
                else None
            ),
            max_shots=int(data["max_shots"]) if data.get("max_shots") is not None else None,
        )


@dataclass(frozen=True)
class PriorityContext:
    queue_level: str = "normal"
    noise_level: str = "normal"
    backend: str = "unknown"

    def to_json(self) -> Dict[str, str]:
        return {
            "queue_level": str(self.queue_level),
            "noise_level": str(self.noise_level),
            "backend": str(self.backend),
        }

    @staticmethod
    def from_json(data: Dict[str, Any]) -> "PriorityContext":
        return PriorityContext(
            queue_level=str(data.get("queue_level", "normal")),
            noise_level=str(data.get("noise_level", "normal")),
            backend=str(data.get("backend", "unknown")),
        )


@dataclass(frozen=True)
class PriorityProfilePayload:
    profile_id: str = "balanced"
    weights: Optional[PriorityWeights] = None
    budgets: Optional[PriorityBudgets] = None
    context: Optional[PriorityContext] = None

    def to_json(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"profile_id": str(self.profile_id)}
        if self.weights is not None:
            out["weights"] = self.weights.to_json()
        if self.budgets is not None:
            out["budgets"] = self.budgets.to_json()
        if self.context is not None:
            out["context"] = self.context.to_json()
        return out

    @staticmethod
    def from_json(data: Dict[str, Any]) -> "PriorityProfilePayload":
        weights = None
        budgets = None
        context = None
        if isinstance(data.get("weights"), dict):
            weights = PriorityWeights.from_json(dict(data["weights"]))
        if isinstance(data.get("budgets"), dict):
            budgets = PriorityBudgets.from_json(dict(data["budgets"]))
        if isinstance(data.get("context"), dict):
            context = PriorityContext.from_json(dict(data["context"]))
        return PriorityProfilePayload(
            profile_id=str(data.get("profile_id", "balanced")),
            weights=weights,
            budgets=budgets,
            context=context,
        )


def priority_payload_from_json(data: Optional[Dict[str, Any]]) -> PriorityProfilePayload:
    if not isinstance(data, dict):
        return PriorityProfilePayload()
    return PriorityProfilePayload.from_json(data)


@dataclass(frozen=True)
class ObservationPayload:
    gate_count: int
    depth: int
    num_cnot: int
    num_rz: int
    constraint_profile: str = "balanced"
    priority_profile_id: str = "balanced"
    last_action_id: int = 0

    def to_json(self) -> Dict[str, Any]:
        return {
            "gate_count": int(self.gate_count),
            "depth": int(self.depth),
            "num_cnot": int(self.num_cnot),
            "num_rz": int(self.num_rz),
            "constraint_profile": str(self.constraint_profile),
            "priority_profile_id": str(self.priority_profile_id),
            "last_action_id": int(self.last_action_id),
        }

    @staticmethod
    def from_json(data: Dict[str, Any]) -> "ObservationPayload":
        constraint_profile = str(data.get("constraint_profile", "balanced"))
        return ObservationPayload(
            gate_count=int(data["gate_count"]),
            depth=int(data["depth"]),
            num_cnot=int(data["num_cnot"]),
            num_rz=int(data["num_rz"]),
            constraint_profile=constraint_profile,
            priority_profile_id=str(data.get("priority_profile_id", constraint_profile)),
            last_action_id=int(data.get("last_action_id", 0)),
        )


@dataclass(frozen=True)
class ActionPayload:
    action_id: int

    def to_json(self) -> Dict[str, Any]:
        return {"action_id": int(self.action_id)}

    @staticmethod
    def from_json(data: Dict[str, Any]) -> "ActionPayload":
        return ActionPayload(action_id=int(data["action_id"]))


def observation_vector_from_payload(payload: ObservationPayload) -> Tuple[float, ...]:
    profile_for_id = payload.priority_profile_id or payload.constraint_profile
    return (
        float(payload.gate_count),
        float(payload.depth),
        float(payload.num_cnot),
        float(payload.num_rz),
        float(payload.last_action_id),
        float(constraint_profile_to_id(profile_for_id)),
    )


def payload_from_metrics(
    metrics: Any,
    constraint_profile: str = "balanced",
    priority_profile_id: Optional[str] = None,
    last_action_id: int = 0,
) -> ObservationPayload:
    if isinstance(metrics, dict):
        gate_count = metrics["gate_count"]
        depth = metrics["depth"]
        num_cnot = metrics.get("cx_count", metrics.get("num_cnot"))
        num_rz = metrics.get("rz_count", metrics.get("num_rz", 0))
    else:
        gate_count = metrics.gate_count
        depth = metrics.depth
        num_cnot = getattr(metrics, "cx_count", getattr(metrics, "num_cnot", 0))
        num_rz = getattr(metrics, "rz_count", getattr(metrics, "num_rz", 0))
    selected_profile = str(priority_profile_id or constraint_profile)
    return ObservationPayload(
        gate_count=int(gate_count),
        depth=int(depth),
        num_cnot=int(num_cnot),
        num_rz=int(num_rz),
        constraint_profile=str(constraint_profile),
        priority_profile_id=selected_profile,
        last_action_id=int(last_action_id),
    )
