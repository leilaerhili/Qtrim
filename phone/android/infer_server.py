"""
Android-side ONNX inference server for QTrim.

This service receives observation payloads from the PC API and returns
an action id predicted by the exported ONNX policy.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
import numpy as np
import onnxruntime as ort
from pydantic import BaseModel

# Ensure repository root is importable when launched as a script.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.shared_schema import ObservationPayload, observation_vector_from_payload


@dataclass
class OnnxRunner:
    session: Any
    input_name: str
    output_name: str
    providers: List[str]

    def predict_action(self, observation_vector: List[float], action_mask: Optional[List[int]]) -> int:
        obs = np.asarray(observation_vector, dtype=np.float32).reshape(1, -1)
        q_values = self.session.run([self.output_name], {self.input_name: obs})[0]
        q_values = np.asarray(q_values, dtype=np.float32)
        if q_values.ndim == 1:
            q_values = q_values.reshape(1, -1)

        if action_mask is not None:
            mask = np.asarray(action_mask, dtype=np.int32).reshape(-1)
            if mask.shape[0] == q_values.shape[1]:
                q_values = q_values.copy()
                q_values[0, mask <= 0] = np.finfo(np.float32).min

        return int(np.argmax(q_values, axis=1)[0])


class InferRequest(BaseModel):
    observation: Optional[Dict[str, Any]] = None
    observation_vector: Optional[List[float]] = None
    action_mask: Optional[List[int]] = None


def _resolve_observation_vector(req: InferRequest) -> List[float]:
    if req.observation_vector:
        return [float(v) for v in req.observation_vector]
    if isinstance(req.observation, dict):
        payload = ObservationPayload.from_json(dict(req.observation))
        return list(observation_vector_from_payload(payload))
    raise ValueError("Missing observation. Provide observation_vector or observation payload.")


def _load_runner(model_path: Path, cpu_only: bool) -> OnnxRunner:
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    available = [str(p) for p in ort.get_available_providers()]
    if cpu_only:
        providers = ["CPUExecutionProvider"]
    elif "QNNExecutionProvider" in available:
        providers = ["QNNExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(str(model_path), providers=providers)
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if not inputs or not outputs:
        raise RuntimeError("Invalid ONNX model session (missing input/output tensors).")
    return OnnxRunner(
        session=session,
        input_name=str(inputs[0].name),
        output_name=str(outputs[0].name),
        providers=[str(p) for p in session.get_providers()],
    )


def create_app(model_path: Path, cpu_only: bool = False) -> FastAPI:
    runner = _load_runner(model_path=model_path, cpu_only=cpu_only)
    app = FastAPI()

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "ok": True,
            "model_path": str(model_path),
            "providers": list(runner.providers),
        }

    @app.post("/infer")
    def infer(req: InferRequest) -> Dict[str, int]:
        try:
            vec = _resolve_observation_vector(req)
            action_id = runner.predict_action(vec, req.action_mask)
            return {"action_id": int(action_id)}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QTrim Android ONNX inference server")
    parser.add_argument(
        "--model",
        default=str(
            REPO_ROOT / "core" / "policy_store" / "tiny_infer_handoff_seed0_android_int8_bs1.onnx"
        ),
        help="Path to ONNX model",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--cpu-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    app = create_app(model_path=Path(args.model).resolve(), cpu_only=bool(args.cpu_only))
    import uvicorn

    uvicorn.run(app, host=str(args.host), port=int(args.port))
