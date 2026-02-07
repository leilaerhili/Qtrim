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
from fastapi.responses import HTMLResponse
import numpy as np
import onnxruntime as ort
from pydantic import BaseModel

# Ensure repository root is importable when launched as a script.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.shared_schema import ObservationPayload, observation_vector_from_payload


ALLOWED_PROFILE_IDS = ("balanced", "high_fidelity", "low_latency", "low_cost")
PROFILE_LABELS = {
    "balanced": "Balanced",
    "high_fidelity": "High Fidelity",
    "low_latency": "Low Latency",
    "low_cost": "Low Cost",
}
_PROFILE_ALIASES = {
    "default": "balanced",
    "low_noise": "high_fidelity",
    "noise": "high_fidelity",
    "min_cx": "high_fidelity",
    "few_cx": "high_fidelity",
    "cx": "high_fidelity",
    "latency": "low_latency",
    "cost": "low_cost",
}


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


class ProfileRequest(BaseModel):
    profile_id: str


def _normalize_profile_id(profile_id: str) -> str:
    key = str(profile_id).strip().lower().replace(" ", "_")
    return _PROFILE_ALIASES.get(key, key)


def _validate_profile_id(profile_id: str) -> str:
    normalized = _normalize_profile_id(profile_id)
    if normalized not in ALLOWED_PROFILE_IDS:
        raise ValueError(
            f"Unsupported profile_id '{profile_id}'. "
            f"Choose one of: {', '.join(ALLOWED_PROFILE_IDS)}."
        )
    return normalized


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


def create_app(
    model_path: Path,
    cpu_only: bool = False,
    default_profile_id: str = "balanced",
) -> FastAPI:
    runner = _load_runner(model_path=model_path, cpu_only=cpu_only)
    app = FastAPI()
    try:
        current_profile_id = _validate_profile_id(default_profile_id)
    except Exception:
        current_profile_id = "balanced"

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "ok": True,
            "model_path": str(model_path),
            "providers": list(runner.providers),
        }

    @app.get("/profile")
    def get_profile() -> Dict[str, Any]:
        return {
            "profile_id": current_profile_id,
            "allowed_profiles": list(ALLOWED_PROFILE_IDS),
        }

    @app.post("/profile")
    def set_profile(req: ProfileRequest) -> Dict[str, Any]:
        nonlocal current_profile_id
        try:
            current_profile_id = _validate_profile_id(req.profile_id)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "profile_id": current_profile_id,
            "allowed_profiles": list(ALLOWED_PROFILE_IDS),
        }

    @app.get("/", response_class=HTMLResponse)
    def profile_ui() -> str:
        options = []
        for pid in ALLOWED_PROFILE_IDS:
            selected = " selected" if pid == current_profile_id else ""
            label = PROFILE_LABELS.get(pid, pid)
            options.append(f'<option value="{pid}"{selected}>{label}</option>')
        options_html = "\n".join(options)
        return f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>QTrim Priority Profile</title>
    <style>
      body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        margin: 0;
        padding: 24px;
        background: #0f1420;
        color: #eef3ff;
      }}
      .card {{
        max-width: 420px;
        margin: 0 auto;
        padding: 20px;
        border-radius: 14px;
        background: #151b2a;
        border: 1px solid rgba(120, 170, 255, 0.2);
      }}
      h2 {{
        margin: 0 0 8px 0;
        font-size: 1.4rem;
      }}
      select, button {{
        width: 100%;
        font-size: 1rem;
        padding: 10px 12px;
        border-radius: 10px;
        border: 1px solid rgba(120, 170, 255, 0.3);
        background: #0b1020;
        color: #eef3ff;
      }}
      button {{
        margin-top: 12px;
        background: linear-gradient(180deg, #2b5fff, #16307a);
        font-weight: 600;
      }}
      .meta {{
        opacity: 0.8;
        margin-top: 12px;
        font-size: 0.9rem;
      }}
      #status {{
        margin-top: 10px;
        font-size: 0.85rem;
        opacity: 0.85;
      }}
    </style>
  </head>
  <body>
    <div class="card">
      <h2>QTrim Priority Profile</h2>
      <div class="meta">Current: <span id="current">{current_profile_id}</span></div>
      <select id="profile">
        {options_html}
      </select>
      <button onclick="setProfile()">Set Profile</button>
      <div id="status"></div>
    </div>
    <script>
      async function setProfile() {{
        const profile = document.getElementById("profile").value;
        const res = await fetch("/profile", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{ profile_id: profile }}),
        }});
        const data = await res.json();
        if (res.ok) {{
          document.getElementById("current").textContent = data.profile_id || profile;
          document.getElementById("status").textContent = "Saved.";
        }} else {{
          document.getElementById("status").textContent = data.detail || "Failed to save.";
        }}
      }}
    </script>
  </body>
</html>
"""

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
    parser.add_argument(
        "--profile",
        default="balanced",
        help="Initial priority profile (balanced, high_fidelity, low_latency, low_cost)",
    )
    parser.add_argument("--cpu-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    app = create_app(
        model_path=Path(args.model).resolve(),
        cpu_only=bool(args.cpu_only),
        default_profile_id=str(args.profile),
    )
    import uvicorn

    uvicorn.run(app, host=str(args.host), port=int(args.port))
