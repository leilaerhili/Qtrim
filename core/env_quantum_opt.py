
"""
core/env_quantum_opt.py

Gymnasium environment for quantum circuit optimization via local rewrite rules.

This environment frames circuit optimization as a sequential decision process:
- State: compact numeric observation vector derived from the current circuit.
- Actions: discrete rewrite rules (see core.rewrites).
- Reward: reduction in a weighted circuit cost (see core.metrics).

The environment is intentionally small and "hackathon-safe":
- Tiny observation space (few floats).
- Small action space (6 actions).
- Conservative, correctness-preserving rewrites.
- Deterministic episode dynamics given a fixed starting circuit and actions.

Typical usage
-------------
from core.env_quantum_opt import QuantumOptEnv
env = QuantumOptEnv(circuit_builder=my_builder_fn)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action_id)

You can also create a vectorized env for Stable-Baselines3.

Notes on correctness
--------------------
We assume the rewrite rules preserve semantics. Optional unitary checks are
available (small circuits only) but are disabled by default for speed.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from qiskit.circuit import QuantumCircuit

from core.metrics import (
    CostWeights,
    compute_cost,
    compute_metrics,
    observation_vector,
    weights_for_profile,
)
from core.rewrites import (
    RewriteResult,
    apply_action,
    list_actions,
)


CircuitBuilder = Callable[[int], QuantumCircuit]
# CircuitBuilder(pad_level) -> QuantumCircuit


@dataclass(frozen=True)
class EnvConfig:
    """
    Configuration for QuantumOptEnv.

    Parameters
    ----------
    max_steps:
        Max number of actions per episode.
    stall_patience:
        End episode early if cost hasn't improved for this many consecutive steps.
    reward_noop:
        Small negative penalty when an action produces no change.
    reward_invalid:
        Penalty when action fails unexpectedly (exceptions).
    constraint_profile:
        Name of cost profile for this env instance (affects reward signal).
    normalize_obs:
        If True, apply simple scaling to observation vector.
    """
    max_steps: int = 30
    stall_patience: int = 8
    reward_noop: float = -0.05
    reward_invalid: float = -1.0
    constraint_profile: str = "balanced"
    normalize_obs: bool = True


class QuantumOptEnv(gym.Env):
    """
    Gymnasium environment: optimize a quantum circuit using local rewrite rules.

    Action space
    ------------
    Discrete(N) where N = len(list_actions()).

    Observation space
    -----------------
    6D vector:
        [gate_count, depth, cx_count, rz_count, last_action_id, constraint_id]

    Episodes
    --------
    - Start with a baseline circuit produced by circuit_builder(pad_level).
    - Agent applies rewrites to reduce cost.
    - Episode ends when:
        - max_steps reached, or
        - no improvement for stall_patience steps.

    """

    metadata = {"render_modes": ["human"], "render_fps": 8}

    def __init__(
        self,
        circuit_builder: CircuitBuilder,
        pad_level: int = 1,
        config: EnvConfig = EnvConfig(),
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        self._circuit_builder = circuit_builder
        self._pad_level = int(pad_level)
        self._config = config

        # Stable action ids.
        self._actions = list_actions()
        self.action_space = spaces.Discrete(len(self._actions))

        # Observation vector: 6 floats.
        # We'll allow a generous upper bound; normalization makes it robust anyway.
        high = np.array([1e6, 1e6, 1e6, 1e6, float(len(self._actions)), 10.0], dtype=np.float32)
        low = np.zeros_like(high, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # RNG
        self._np_random, _ = gym.utils.seeding.np_random(seed)

        # Derived settings
        self._weights: CostWeights = weights_for_profile(config.constraint_profile)
        self._constraint_id: int = self._profile_to_id(config.constraint_profile)

        # Episode state (set in reset)
        self._circ: Optional[QuantumCircuit] = None
        self._last_action_id: int = 0
        self._step_count: int = 0
        self._stall_count: int = 0
        self._last_cost: float = 0.0

    @staticmethod
    def _profile_to_id(profile: str) -> int:
        p = profile.strip().lower()
        mapping = {"balanced": 0, "default": 0, "low_latency": 1, "low_noise": 2, "min_cx": 3}
        return mapping.get(p, 0)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._np_random, _ = gym.utils.seeding.np_random(seed)

        # Allow overriding pad_level via reset options
        pad_level = self._pad_level
        if options and "pad_level" in options:
            pad_level = int(options["pad_level"])

        self._circ = self._circuit_builder(pad_level)
        self._last_action_id = 0
        self._step_count = 0
        self._stall_count = 0
        self._last_cost = compute_cost(self._circ, weights=self._weights)

        obs = self._get_obs()
        info = self._get_info(last_result=None)
        return obs, info

    def step(self, action: int):
        if self._circ is None:
            raise RuntimeError("Environment must be reset() before step().")

        self._step_count += 1
        action_id = int(action)

        old_cost = self._last_cost
        last_result: Optional[RewriteResult] = None

        try:
            new_circ, result = apply_action(self._circ, action_id)
            last_result = result
            self._last_action_id = action_id
            self._circ = new_circ
        except Exception as e:
            # Unexpected failure: penalize and terminate for safety.
            reward = float(self._config.reward_invalid)
            terminated = True
            truncated = False
            info = self._get_info(last_result=None)
            info["error"] = str(e)
            obs = self._get_obs()
            return obs, reward, terminated, truncated, info

        new_cost = compute_cost(self._circ, weights=self._weights)
        self._last_cost = new_cost

        # Reward is improvement in cost.
        improvement = old_cost - new_cost
        reward = float(improvement)

        # Penalize pure no-ops slightly to encourage exploration.
        if last_result is not None and not last_result.changed:
            reward += float(self._config.reward_noop)

        # Stall tracking
        if improvement > 1e-12:
            self._stall_count = 0
        else:
            self._stall_count += 1

        terminated = False
        if self._stall_count >= self._config.stall_patience:
            terminated = True

        truncated = self._step_count >= self._config.max_steps

        obs = self._get_obs()
        info = self._get_info(last_result=last_result)
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        assert self._circ is not None
        vec = observation_vector(
            self._circ,
            last_action_id=self._last_action_id,
            constraint_id=self._constraint_id,
        )
        obs = np.array(vec, dtype=np.float32)

        if self._config.normalize_obs:
            # Simple, stable scaling: keeps magnitudes reasonable.
            # gate_count, depth, cx_count, rz_count can vary; divide by 100 for typicalR.
            obs[0:4] = obs[0:4] / 100.0
            # last_action and constraint id are already small.
        return obs

    def _get_info(self, last_result: Optional[RewriteResult]) -> Dict:
        assert self._circ is not None
        m = compute_metrics(self._circ, weights=self._weights)
        info: Dict = {
            "step": self._step_count,
            "stall_count": self._stall_count,
            "constraint_profile": self._config.constraint_profile,
            "metrics": {
                "gate_count": m.gate_count,
                "depth": m.depth,
                "cx_count": m.cx_count,
                "cost": m.cost,
            },
        }
        if last_result is not None:
            info["last_action"] = {
                "action_id": last_result.action_id,
                "action_name": last_result.action_name,
                "changed": last_result.changed,
                "message": last_result.message,
                "window": last_result.window,
            }
        return info

    def render(self):
        if self._circ is None:
            print("(env not reset)")
            return
        m = compute_metrics(self._circ, weights=self._weights)
        print(f"Step {self._step_count} | cost={m.cost:.3f} | gates={m.gate_count} | depth={m.depth} | cx={m.cx_count}")

    def get_circuit(self) -> QuantumCircuit:
        """Return a copy of the current circuit for visualization."""
        if self._circ is None:
            raise RuntimeError("Environment not reset.")
        return self._circ.copy()

    def set_constraint_profile(self, profile: str) -> None:
        """
        Update cost weights (useful for simulating phone-side constraint profiles).

        This does not reset the circuit; it only changes how reward/cost is computed.
        """
        self._weights = weights_for_profile(profile)
        self._constraint_id = self._profile_to_id(profile)
        # Update last_cost to match new weights for consistent reward deltas.
        if self._circ is not None:
            self._last_cost = compute_cost(self._circ, weights=self._weights)


# -----------------------------
# Example builder + self-test
# -----------------------------

def _example_builder(pad_level: int) -> QuantumCircuit:
    """
    Minimal example circuit builder.

    Replace this with core.circuits_baseline.build_* functions in your project.
    """
    from qiskit.circuit import QuantumRegister
    qr = QuantumRegister(3, "q")
    qc = QuantumCircuit(qr)
    qc.cx(qr[0], qr[1])
    qc.cx(qr[0], qr[1])  # cancellable
    qc.rz(0.7, qr[2])
    qc.rz(-0.7, qr[2])   # cancellable
    # Pad a bit more if requested
    for _ in range(max(0, pad_level - 1)):
        qc.rz(0.1, qr[2])
        qc.rz(0.2, qr[2])
    return qc


def _quick_demo() -> None:
    """
    Quick manual sanity test:
        python -m core.env_quantum_opt
    """
    env = QuantumOptEnv(circuit_builder=_example_builder, pad_level=2, config=EnvConfig(max_steps=10))
    obs, info = env.reset()
    print("Reset obs:", obs)
    print("Reset info:", info["metrics"])

    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        print("  action:", info.get("last_action", {}), "reward:", reward)
        if terminated or truncated:
            print("Episode end.", "terminated=", terminated, "truncated=", truncated)
            break


if __name__ == "__main__":
    _quick_demo()

