"""
Shared JSON schema for observations/actions between PC and phone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple


CONSTRAINT_PROFILE_IDS = {
    "balanced": 0,
    "default": 0,
    "low_latency": 1,
    "low_noise": 2,
    "min_cx": 3,
    "few_cx": 3,
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
    key = profile.strip().lower()
    return int(CONSTRAINT_PROFILE_IDS.get(key, 0))


@dataclass(frozen=True)
class ObservationPayload:
    gate_count: int
    depth: int
    num_cnot: int
    num_rz: int
    constraint_profile: str = "balanced"
    last_action_id: int = 0

    def to_json(self) -> Dict[str, Any]:
        return {
            "gate_count": int(self.gate_count),
            "depth": int(self.depth),
            "num_cnot": int(self.num_cnot),
            "num_rz": int(self.num_rz),
            "constraint_profile": str(self.constraint_profile),
            "last_action_id": int(self.last_action_id),
        }

    @staticmethod
    def from_json(data: Dict[str, Any]) -> "ObservationPayload":
        return ObservationPayload(
            gate_count=int(data["gate_count"]),
            depth=int(data["depth"]),
            num_cnot=int(data["num_cnot"]),
            num_rz=int(data["num_rz"]),
            constraint_profile=str(data.get("constraint_profile", "balanced")),
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
    """
    Convert JSON payload to the float vector expected by the RL policy.
    """
    return (
        float(payload.gate_count),
        float(payload.depth),
        float(payload.num_cnot),
        float(payload.num_rz),
        float(payload.last_action_id),
        float(constraint_profile_to_id(payload.constraint_profile)),
    )


def payload_from_metrics(
    metrics: Any,
    constraint_profile: str = "balanced",
    last_action_id: int = 0,
) -> ObservationPayload:
    """
    Helper to build payloads from a CircuitMetrics-like object or dict.
    """
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

    return ObservationPayload(
        gate_count=int(gate_count),
        depth=int(depth),
        num_cnot=int(num_cnot),
        num_rz=int(num_rz),
        constraint_profile=constraint_profile,
        last_action_id=int(last_action_id),
    )
