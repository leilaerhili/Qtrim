"""
Phone-side client placeholder (Termux/Python).
"""

from __future__ import annotations

import json
from typing import Dict


def build_priority_payload(profile_id: str = "high_fidelity") -> Dict:
    return {
        "profile_id": str(profile_id),
        "weights": {
            "two_qubit_gates": 0.50,
            "depth": 0.30,
            "total_gates": 0.10,
            "swap_gates": 0.10,
        },
        "budgets": {
            "max_depth": 250,
            "max_latency_ms": 1500,
            "max_shots": 2000,
        },
        "context": {
            "queue_level": "high",
            "noise_level": "high",
            "backend": "snapdragon_npu",
        },
    }


if __name__ == "__main__":
    print(json.dumps(build_priority_payload(), indent=2))
