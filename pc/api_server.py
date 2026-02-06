from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Baseline circuits for demo (later replaced by real core circuits)
BASELINE = {
    "parity": {"gate_count": 64, "depth": 24, "cost": 118},
    "half_adder": {"gate_count": 72, "depth": 28, "cost": 132},
    "majority_vote": {"gate_count": 80, "depth": 31, "cost": 150},
    "linear_dataflow_pipeline": {"gate_count": 95, "depth": 40, "cost": 190},
}

class OptimizeRequest(BaseModel):
    circuit_id: str
    constraint_profile: str  # "low_noise" | "low_latency"

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/optimize")
def optimize(req: OptimizeRequest):
    if req.circuit_id not in BASELINE:
        raise HTTPException(status_code=400, detail=f"Unknown circuit_id: {req.circuit_id}")

    before = BASELINE[req.circuit_id]

    # Fake "optimization" for now (later: call RL + rewrite engine)
    after = {
        "gate_count": int(before["gate_count"] * 0.77),
        "depth": int(before["depth"] * 0.82),
        "cost": int(before["cost"] * 0.75),
    }

    # Future: these can be filled by core using Qiskit / OpenQASM
    before_qasm = None
    after_qasm = None

    return {
        "circuit_id": req.circuit_id,
        "constraint_profile": req.constraint_profile,
        "before": before,
        "after": after,
        "meta": {"policy_source": "mock"},
        "before_qasm": before_qasm,
        "after_qasm": after_qasm,
    }
