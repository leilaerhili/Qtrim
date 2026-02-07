from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from qiskit.circuit import QuantumCircuit

from core.circuits_baseline import get_builder
from core.env_quantum_opt import EnvConfig, QuantumOptEnv
from core.metrics import (
    CostWeights,
    compute_metrics,
    cost_weights_to_priority_dict,
    resolve_priority_weights,
)
from core.shared_schema import observation_vector_from_payload, payload_from_metrics

app = FastAPI()

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
POLICY_STORE = REPO_ROOT / "core" / "policy_store"
DEFAULT_ANDROID_MODEL_PATH = POLICY_STORE / "tiny_infer_handoff_seed0_android_int8_bs1.onnx"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)


ANDROID_INFER_URL = str(os.getenv("QTRIM_ANDROID_INFER_URL", "http://127.0.0.1:9001/infer"))
ANDROID_TIMEOUT_S = _env_float("QTRIM_ANDROID_TIMEOUT_S", 5.0)
DEFAULT_PAD_LEVEL = max(1, _env_int("QTRIM_OPT_PAD_LEVEL", 2))
DEFAULT_MAX_STEPS = max(1, _env_int("QTRIM_OPT_MAX_STEPS", 30))

# Keep compatibility with existing Streamlit circuit ids.
CIRCUIT_ALIASES = {
    "majority_vote": "majority",
    "linear_dataflow_pipeline": "line",
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
    # Optional runtime overrides:
    android_infer_url: Optional[str] = None
    android_model_name: Optional[str] = None
    max_steps: Optional[int] = None
    pad_level: Optional[int] = None


@app.get("/health")
def health():
    return {"ok": True}


def _is_auto_profile(profile_id: Optional[str]) -> bool:
    if profile_id is None:
        return True
    key = str(profile_id).strip().lower()
    return key in ("", "auto", "phone", "android")


def _profile_url_from_infer_url(infer_url: str) -> str:
    url = str(infer_url).strip()
    if not url:
        return ""
    if url.endswith("/infer"):
        return url[: -len("/infer")] + "/profile"
    return url.rstrip("/") + "/profile"


def _fetch_android_profile_id(infer_url: str, timeout_s: float) -> Optional[str]:
    profile_url = _profile_url_from_infer_url(infer_url)
    if not profile_url:
        return None
    try:
        resp = requests.get(profile_url, timeout=float(timeout_s))
        resp.raise_for_status()
        data = resp.json()
        profile_id = data.get("profile_id")
        if profile_id:
            return str(profile_id).strip().lower()
    except Exception:
        return None
    return None


def _resolve_profile_id(req: OptimizeRequest, infer_url: str) -> str:
    if not _is_auto_profile(req.profile_id):
        return str(req.profile_id).strip().lower()

    phone_profile = _fetch_android_profile_id(infer_url, ANDROID_TIMEOUT_S)
    if phone_profile:
        return phone_profile

    if req.constraint_profile:
        return str(req.constraint_profile).strip().lower()
    return "balanced"


def _resolve_circuit_id(circuit_id: str) -> str:
    key = str(circuit_id).strip().lower()
    return CIRCUIT_ALIASES.get(key, key)


def _metric_payload(circuit: QuantumCircuit, weights: CostWeights) -> Dict[str, Any]:
    m = compute_metrics(circuit, weights=weights)
    return {
        "gate_count": int(m.gate_count),
        "depth": int(m.depth),
        "cost": float(round(m.cost, 6)),
    }


def _circuit_qasm(circuit: QuantumCircuit) -> str:
    try:
        from qiskit import qasm2
        return qasm2.dumps(circuit)
    except Exception:
        try:
            return circuit.qasm()
        except Exception:
            return str(circuit.draw(output="text"))


def _parse_action_id(data: Any) -> int:
    if not isinstance(data, dict):
        raise ValueError("Android response must be a JSON object.")
    for key in ("action_id", "action", "id"):
        if key in data:
            return int(data[key])
    raise ValueError("Android response must include 'action_id'.")


def infer_action_with_android(
    *,
    infer_url: str,
    timeout_s: float,
    observation_payload: Dict[str, Any],
    observation_vector: List[float],
    action_mask: Optional[List[int]],
    profile_id: str,
    effective_weights: Dict[str, float],
    model_name: str,
    step: int,
) -> int:
    req_payload = {
        "observation": dict(observation_payload),
        "observation_vector": list(observation_vector),
        "action_mask": list(action_mask) if action_mask is not None else None,
        "priority": {
            "profile_id": str(profile_id),
            "weights": dict(effective_weights),
        },
        "model": {"name": str(model_name)},
        "step": int(step),
    }
    resp = requests.post(
        str(infer_url),
        json=req_payload,
        timeout=float(timeout_s),
    )
    resp.raise_for_status()
    return _parse_action_id(resp.json())


@dataclass
class OptimizationArtifacts:
    profile_id: str
    before_circuit: QuantumCircuit
    after_circuit: QuantumCircuit
    before_metrics: Dict[str, Any]
    after_metrics: Dict[str, Any]
    effective_weights: Dict[str, float]
    resolved_circuit_id: str
    step_trace: List[Dict[str, Any]]


def optimize_circuit_object(req: OptimizeRequest) -> OptimizationArtifacts:
    weights_override = req.weights.as_dict() if req.weights is not None else None
    budgets = req.budgets or PriorityBudgetsModel()
    context = req.context or PriorityContextModel()
    infer_url = str(req.android_infer_url or ANDROID_INFER_URL)
    profile_id = _resolve_profile_id(req, infer_url)
    max_steps = max(1, int(req.max_steps or DEFAULT_MAX_STEPS))
    pad_level = max(1, int(req.pad_level or DEFAULT_PAD_LEVEL))
    model_name = str(req.android_model_name or DEFAULT_ANDROID_MODEL_PATH.name)

    resolved_circuit_id = _resolve_circuit_id(req.circuit_id)
    try:
        circuit_builder = get_builder(resolved_circuit_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown circuit_id: {req.circuit_id}. "
                f"Resolved key: '{resolved_circuit_id}'."
            ),
        ) from exc

    resolved_weights = resolve_priority_weights(
        profile_id=profile_id,
        override_weights=weights_override,
    )
    effective_weights = cost_weights_to_priority_dict(resolved_weights)

    env = QuantumOptEnv(
        circuit_builder=circuit_builder,
        pad_level=pad_level,
        config=EnvConfig(
            max_steps=max_steps,
            constraint_profile=profile_id,
            priority_profile_id=profile_id,
            priority_weights=weights_override,
            max_depth_budget=budgets.max_depth,
            max_latency_ms=budgets.max_latency_ms,
            max_shots=budgets.max_shots,
            context_queue_level=context.queue_level,
            context_noise_level=context.noise_level,
            context_backend=context.backend,
        ),
        seed=0,
    )

    _, info = env.reset()
    before_circuit = env.get_circuit()
    best_circuit = before_circuit.copy()
    best_cost = float(info["metrics"]["cost"])
    last_action_id = 0
    done = False
    truncated = False
    step_trace: List[Dict[str, Any]] = []

    while not (done or truncated):
        obs_payload = payload_from_metrics(
            metrics=info["metrics"],
            constraint_profile=profile_id,
            priority_profile_id=profile_id,
            last_action_id=last_action_id,
        )
        obs_vector = list(observation_vector_from_payload(obs_payload))

        try:
            action_id = infer_action_with_android(
                infer_url=infer_url,
                timeout_s=ANDROID_TIMEOUT_S,
                observation_payload=obs_payload.to_json(),
                observation_vector=obs_vector,
                action_mask=info.get("action_mask"),
                profile_id=profile_id,
                effective_weights=effective_weights,
                model_name=model_name,
                step=int(info.get("step", 0)),
            )
        except Exception as exc:
            raise RuntimeError(
                "Android inference request failed. "
                f"url={infer_url} step={int(info.get('step', 0))} error={exc}"
            ) from exc

        if action_id < 0 or action_id >= int(env.action_space.n):
            action_id = 0

        _, reward, done, truncated, info = env.step(action_id)
        last_action_id = int(action_id)

        current_cost = float(info["metrics"]["cost"])
        if current_cost < best_cost:
            best_cost = current_cost
            best_circuit = env.get_circuit()

        last_action = info.get("last_action", {})
        step_trace.append(
            {
                "step": int(info.get("step", len(step_trace) + 1)),
                "action_id": int(last_action.get("action_id", action_id)),
                "action_name": str(last_action.get("action_name", f"action_{action_id}")),
                "changed": bool(last_action.get("changed", False)),
                "reward": float(reward),
                "cost": float(round(current_cost, 6)),
            }
        )

    return OptimizationArtifacts(
        profile_id=profile_id,
        before_circuit=before_circuit,
        after_circuit=best_circuit,
        before_metrics=_metric_payload(before_circuit, weights=resolved_weights),
        after_metrics=_metric_payload(best_circuit, weights=resolved_weights),
        effective_weights=effective_weights,
        resolved_circuit_id=resolved_circuit_id,
        step_trace=step_trace,
    )


@app.post("/optimize")
def optimize(req: OptimizeRequest):
    budgets = req.budgets or PriorityBudgetsModel()
    context = req.context or PriorityContextModel()

    try:
        result = optimize_circuit_object(req)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    profile_id = result.profile_id
    model_name = str(req.android_model_name or DEFAULT_ANDROID_MODEL_PATH.name)
    infer_url = str(req.android_infer_url or ANDROID_INFER_URL)
    model_exists_local = bool((POLICY_STORE / model_name).exists())

    return {
        "circuit_id": req.circuit_id,
        "resolved_circuit_id": result.resolved_circuit_id,
        "profile_id": profile_id,
        "effective_weights": result.effective_weights,
        "budgets": _dump_model(budgets),
        "context": _dump_model(context),
        "before": result.before_metrics,
        "after": result.after_metrics,
        "meta": {
            "policy_source": "android_inference_handoff",
            "android_infer_url": infer_url,
            "android_model_name": model_name,
            "android_model_exists_in_pc_policy_store": model_exists_local,
            "steps_executed": len(result.step_trace),
        },
        "before_qasm": _circuit_qasm(result.before_circuit),
        "after_qasm": _circuit_qasm(result.after_circuit),
        "steps": result.step_trace,
    }
