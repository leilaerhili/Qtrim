from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from core.metrics import cost_weights_to_priority_dict, resolve_priority_weights

app = FastAPI()

# Baseline circuits for demo (later replaced by real core circuits)
BASELINE = {
    "parity": {"gate_count": 64, "depth": 24, "cost": 118},
    "half_adder": {"gate_count": 72, "depth": 28, "cost": 132},
    "majority_vote": {"gate_count": 80, "depth": 31, "cost": 150},
    "linear_dataflow_pipeline": {"gate_count": 95, "depth": 40, "cost": 190},
}

_PROFILE_SCALES = {
    "balanced": {"gate_count": 0.82, "depth": 0.82, "cost": 0.77},
    "high_fidelity": {"gate_count": 0.84, "depth": 0.86, "cost": 0.73},
    "low_latency": {"gate_count": 0.88, "depth": 0.70, "cost": 0.79},
    "low_cost": {"gate_count": 0.76, "depth": 0.80, "cost": 0.75},
}


def _dump_model(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return dict(model.model_dump())
    return dict(model.dict())


class PriorityWeightsModel(BaseModel):
    two_qubit_gates: Optional[float] = None
    depth: Optional[float] = None
    total_gates: Optional[float] = None
    swap_gates: Optional[float] = None

    def as_dict(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k in ("two_qubit_gates", "depth", "total_gates", "swap_gates"):
            v = getattr(self, k, None)
            if v is not None:
                out[k] = float(v)
        return out


class PriorityBudgetsModel(BaseModel):
    max_depth: Optional[int] = None
    max_latency_ms: Optional[float] = None
    max_shots: Optional[int] = None


class PriorityContextModel(BaseModel):
    queue_level: str = "normal"
    noise_level: str = "normal"
    backend: str = "unknown"


class OptimizeRequest(BaseModel):
    circuit_id: str
    profile_id: str = "auto"
    weights: Optional[PriorityWeightsModel] = None
    budgets: Optional[PriorityBudgetsModel] = None
    context: Optional[PriorityContextModel] = None
    # Backward compatibility with older caller:
    constraint_profile: Optional[str] = None


@app.get("/health")
def health():
    return {"ok": True}


def _effective_profile_id(req: OptimizeRequest) -> str:
    if req.profile_id and str(req.profile_id).strip().lower() != "auto":
        return str(req.profile_id).strip().lower()
    if req.constraint_profile:
        return str(req.constraint_profile).strip().lower()
    return "balanced"


def _optimize_mock(
    before: Dict[str, int],
    profile_id: str,
    budgets: PriorityBudgetsModel,
    context: PriorityContextModel,
) -> Dict[str, int]:
    scales = dict(_PROFILE_SCALES.get(profile_id, _PROFILE_SCALES["balanced"]))

    if context.queue_level.lower() == "high":
        scales["depth"] *= 0.90
    if context.noise_level.lower() == "high":
        scales["cost"] *= 0.94
        scales["gate_count"] *= 0.96

    after = {
        "gate_count": max(1, int(round(float(before["gate_count"]) * scales["gate_count"]))),
        "depth": max(1, int(round(float(before["depth"]) * scales["depth"]))),
        "cost": max(1, int(round(float(before["cost"]) * scales["cost"]))),
    }

    if budgets.max_depth is not None and budgets.max_depth > 0:
        after["depth"] = min(after["depth"], int(budgets.max_depth))

    if budgets.max_latency_ms is not None and budgets.max_latency_ms > 0:
        # Demo proxy: 5ms per depth unit.
        max_depth_from_latency = max(1, int(float(budgets.max_latency_ms) / 5.0))
        after["depth"] = min(after["depth"], max_depth_from_latency)

    return after


@app.post("/optimize")
def optimize(req: OptimizeRequest):
    if req.circuit_id not in BASELINE:
        raise HTTPException(status_code=400, detail=f"Unknown circuit_id: {req.circuit_id}")

    before = dict(BASELINE[req.circuit_id])
    profile_id = _effective_profile_id(req)
    weights_override = req.weights.as_dict() if req.weights is not None else None
    budgets = req.budgets or PriorityBudgetsModel()
    context = req.context or PriorityContextModel()

    effective_weights = cost_weights_to_priority_dict(
        resolve_priority_weights(profile_id=profile_id, override_weights=weights_override)
    )

    after = _optimize_mock(
        before=before,
        profile_id=profile_id,
        budgets=budgets,
        context=context,
    )

    before_qasm = None
    after_qasm = None

    return {
        "circuit_id": req.circuit_id,
        "profile_id": profile_id,
        "effective_weights": effective_weights,
        "budgets": _dump_model(budgets),
        "context": _dump_model(context),
        "before": before,
        "after": after,
        "meta": {"policy_source": "mock_priority_router"},
        "before_qasm": before_qasm,
        "after_qasm": after_qasm,
    }
