"""
core/env_quantum_opt.py

Gymnasium environment for quantum circuit optimization via local rewrite rules.

This environment frames circuit optimization as a sequential decision process:
- State: compact numeric observation vector derived from the current circuit.
- Actions: discrete rewrite rules (see core.rewrites).
- Reward: reduction in a weighted circuit cost (see core.metrics).

The environment is intentionally small and "hackathon-safe":
- Tiny observation space (few floats).
- Small action space (few actions).
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
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from qiskit.circuit import QuantumCircuit

from core.metrics import (
    CostWeights,
    cost_weights_to_priority_dict,
    compute_cost,
    compute_metrics,
    observation_vector,
    resolve_priority_weights,
)
from core.shared_schema import constraint_profile_to_id
from core.rewrites import (
    RewriteResult,
    apply_action,
    applicable_action_mask,
    list_actions,
    set_coupling_map,
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
    reward_inapplicable:
        Additional penalty when a chosen action is currently inapplicable while
        another applicable action exists.
    reward_repeat_action:
        Penalty for repeating the same action id as the previous step.
    reward_repeat_noop:
        Extra penalty for repeated no-op action (spam deterrent).
    reward_invalid:
        Penalty when action fails unexpectedly (exceptions).
    constraint_profile:
        Name of cost profile for this env instance (affects reward signal).
    normalize_obs:
        If True, apply simple scaling to observation vector.
    coupling_edges:
        Optional hardware coupling map edges used by routing rewrites.
        If None, a line topology is inferred from circuit qubit count at reset.
    coupling_directed:
        Whether coupling_edges are directed.
    allowed_action_ids:
        Optional subset of global action ids allowed at step-time.
        Action space remains fixed; disallowed actions become masked no-ops.
    use_applicability_mask:
        If True, compute action applicability mask each state and expose it in info.
    """

    max_steps: int = 30
    stall_patience: int = 8
    reward_noop: float = -0.1
    reward_inapplicable: float = -0.2
    reward_repeat_action: float = -0.03
    reward_repeat_noop: float = -0.1
    reward_invalid: float = -1.0
    constraint_profile: str = "balanced"
    normalize_obs: bool = True
    coupling_edges: Optional[Tuple[Tuple[int, int], ...]] = None
    coupling_directed: bool = False
    allowed_action_ids: Optional[Tuple[int, ...]] = None
    use_applicability_mask: bool = True
    priority_profile_id: str = "auto"
    priority_weights: Optional[Dict[str, float]] = None
    max_depth_budget: Optional[int] = None
    max_latency_ms: Optional[float] = None
    max_shots: Optional[int] = None
    estimated_latency_per_depth_ms: float = 5.0
    estimated_shots_per_step: int = 64
    budget_penalty_scale: float = 1.0
    context_queue_level: str = "normal"
    context_noise_level: str = "normal"
    context_backend: str = "unknown"


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

        # Stable action ids (fixed action-space for SB3 compatibility).
        self._actions = list_actions()
        self.action_space = spaces.Discrete(len(self._actions))
        self._global_to_name = dict(self._actions)
        if config.allowed_action_ids is None:
            self._allowed_action_ids: Optional[set[int]] = None
        else:
            chosen = {int(aid) for aid in config.allowed_action_ids}
            known = {aid for aid, _ in self._actions}
            invalid = sorted(chosen - known)
            if invalid:
                raise ValueError(f"Unknown action ids in allowed_action_ids: {invalid}")
            if not chosen:
                raise ValueError("allowed_action_ids produced an empty action set.")
            self._allowed_action_ids = chosen

        # Observation vector: 6 floats.
        # We'll allow generous bounds; normalization makes it robust anyway.
        high = np.array(
            [1e6, 1e6, 1e6, 1e6, 1e6, 10.0],
            dtype=np.float32,
        )
        low = np.zeros_like(high, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # RNG
        self._np_random, _ = gym.utils.seeding.np_random(seed)

        # Derived settings
        requested_profile = str(config.priority_profile_id).strip().lower()
        if requested_profile in ("", "auto"):
            requested_profile = str(config.constraint_profile).strip().lower()
        self._priority_profile_id = requested_profile
        self._weights: CostWeights = resolve_priority_weights(
            profile_id=self._priority_profile_id,
            override_weights=config.priority_weights,
        )
        self._constraint_id: int = self._profile_to_id(self._priority_profile_id)

        # Episode state (set in reset)
        self._circ: Optional[QuantumCircuit] = None
        self._last_action_id: int = 0
        self._step_count: int = 0
        self._stall_count: int = 0
        self._last_cost: float = 0.0
        self._action_mask: Optional[List[int]] = None
        self._last_budget_penalty: float = 0.0
        self._last_budget_breakdown: Dict[str, float] = {}

    @staticmethod
    def _profile_to_id(profile: str) -> int:
        return int(constraint_profile_to_id(profile))

    @staticmethod
    def _default_coupling_edges(num_qubits: int) -> Tuple[Tuple[int, int], ...]:
        if num_qubits <= 1:
            return tuple()
        return tuple((i, i + 1) for i in range(num_qubits - 1))

    def _estimate_latency_ms(self, depth_value: int) -> float:
        return float(depth_value) * float(max(0.0, self._config.estimated_latency_per_depth_ms))

    def _estimate_shots(self) -> int:
        return int(self._step_count * max(0, int(self._config.estimated_shots_per_step)))

    def _compute_budget_penalty(self, depth_value: int) -> Tuple[float, Dict[str, float]]:
        penalty = 0.0
        breakdown: Dict[str, float] = {}
        scale = float(max(0.0, self._config.budget_penalty_scale))

        if self._config.max_depth_budget is not None and self._config.max_depth_budget > 0:
            limit = float(self._config.max_depth_budget)
            over = max(0.0, float(depth_value) - limit)
            if over > 0.0:
                term = scale * (over / limit)
                penalty += term
                breakdown["depth"] = float(term)

        latency = self._estimate_latency_ms(depth_value)
        if self._config.max_latency_ms is not None and self._config.max_latency_ms > 0.0:
            limit = float(self._config.max_latency_ms)
            over = max(0.0, float(latency) - limit)
            if over > 0.0:
                term = scale * (over / limit)
                penalty += term
                breakdown["latency_ms"] = float(term)

        shots = self._estimate_shots()
        if self._config.max_shots is not None and self._config.max_shots > 0:
            limit = float(self._config.max_shots)
            over = max(0.0, float(shots) - limit)
            if over > 0.0:
                term = scale * (over / limit)
                penalty += term
                breakdown["shots"] = float(term)

        return float(penalty), breakdown

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._np_random, _ = gym.utils.seeding.np_random(seed)

        # Allow overriding pad_level via reset options
        pad_level = self._pad_level
        if options and "pad_level" in options:
            pad_level = int(options["pad_level"])

        self._circ = self._circuit_builder(pad_level)
        configured_edges: Optional[Sequence[Tuple[int, int]]] = self._config.coupling_edges
        if configured_edges is None:
            configured_edges = self._default_coupling_edges(self._circ.num_qubits)
        set_coupling_map(configured_edges, directed=self._config.coupling_directed)
        self._last_action_id = 0
        self._step_count = 0
        self._stall_count = 0
        self._last_cost = compute_cost(self._circ, weights=self._weights)
        self._last_budget_penalty = 0.0
        self._last_budget_breakdown = {}
        self._action_mask = self._compute_action_mask()

        obs = self._get_obs()
        info = self._get_info(last_result=None)
        return obs, info

    def _compute_action_mask(self) -> List[int]:
        if self._circ is None:
            return [0] * len(self._actions)
        if not self._config.use_applicability_mask:
            if self._allowed_action_ids is None:
                return [1] * len(self._actions)
            return [1 if aid in self._allowed_action_ids else 0 for aid, _ in self._actions]
        return applicable_action_mask(
            self._circ,
            allowed_action_ids=self._allowed_action_ids,
        )

    def step(self, action: int):
        if self._circ is None:
            raise RuntimeError("Environment must be reset() before step().")

        self._step_count += 1
        action_id = int(action)
        if action_id < 0 or action_id >= len(self._actions):
            raise ValueError(
                f"Action index out of range: {action_id}. "
                f"Expected [0, {len(self._actions) - 1}]"
            )

        old_cost = self._last_cost
        prev_action_id = self._last_action_id
        last_result: Optional[RewriteResult] = None
        mask = self._action_mask if self._action_mask is not None else self._compute_action_mask()
        has_any_applicable = any(bool(x) for x in mask)
        is_action_applicable = bool(mask[action_id]) if action_id < len(mask) else False

        if not is_action_applicable:
            last_result = RewriteResult(
                changed=False,
                action_id=action_id,
                action_name=self._global_to_name.get(action_id, f"action_{action_id}"),
                message=(
                    "Action masked by curriculum stage."
                    if (self._allowed_action_ids is not None and action_id not in self._allowed_action_ids)
                    else "Action currently inapplicable."
                ),
                old_len=len(self._circ.data),
                new_len=len(self._circ.data),
                window=None,
            )
            self._last_action_id = action_id
        else:
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

        new_metrics = compute_metrics(self._circ, weights=self._weights)
        new_cost = float(new_metrics.cost)
        self._last_cost = float(new_cost)

        # Reward is improvement in cost.
        improvement = old_cost - new_cost
        reward = float(improvement)

        # Penalize pure no-ops slightly to encourage exploration.
        if last_result is not None and not last_result.changed:
            reward += float(self._config.reward_noop)
            if has_any_applicable:
                reward += float(self._config.reward_inapplicable)

        # Stall tracking
        if improvement > 1e-12:
            self._stall_count = 0
        else:
            self._stall_count += 1

        # Discourage action spam.
        if action_id == prev_action_id:
            reward += float(self._config.reward_repeat_action)
            if last_result is not None and not last_result.changed:
                reward += float(self._config.reward_repeat_noop)

        budget_penalty, budget_breakdown = self._compute_budget_penalty(new_metrics.depth)
        self._last_budget_penalty = float(budget_penalty)
        self._last_budget_breakdown = dict(budget_breakdown)
        reward -= float(budget_penalty)

        terminated = self._stall_count >= self._config.stall_patience
        truncated = self._step_count >= self._config.max_steps

        self._action_mask = self._compute_action_mask()
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
            # Simple scaling: keep magnitudes reasonable.
            obs[0:4] = obs[0:4] / 100.0
        return obs

    def _get_info(self, last_result: Optional[RewriteResult]) -> Dict:
        assert self._circ is not None
        m = compute_metrics(self._circ, weights=self._weights)
        estimated_latency_ms = self._estimate_latency_ms(m.depth)
        estimated_shots = self._estimate_shots()
        info: Dict = {
            "step": self._step_count,
            "stall_count": self._stall_count,
            "constraint_profile": self._config.constraint_profile,
            "priority_profile_id": self._priority_profile_id,
            "priority_weights": cost_weights_to_priority_dict(self._weights),
            "budgets": {
                "max_depth": self._config.max_depth_budget,
                "max_latency_ms": self._config.max_latency_ms,
                "max_shots": self._config.max_shots,
            },
            "context": {
                "queue_level": self._config.context_queue_level,
                "noise_level": self._config.context_noise_level,
                "backend": self._config.context_backend,
            },
            "metrics": {
                "gate_count": m.gate_count,
                "depth": m.depth,
                "cx_count": m.cx_count,
                "swap_count": m.swap_count,
                "cost": m.cost,
                "estimated_latency_ms": estimated_latency_ms,
                "estimated_shots": estimated_shots,
            },
            "budget_penalty": {
                "active": bool(self._last_budget_penalty > 0.0),
                "total": float(self._last_budget_penalty),
                "breakdown": dict(self._last_budget_breakdown),
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
        if self._action_mask is None:
            info["action_mask"] = self._compute_action_mask()
        else:
            info["action_mask"] = list(self._action_mask)
        if self._allowed_action_ids is None:
            info["available_actions"] = self._actions
        else:
            info["available_actions"] = [
                (aid, name) for aid, name in self._actions if aid in self._allowed_action_ids
            ]
        return info

    def render(self):
        if self._circ is None:
            print("(env not reset)")
            return
        m = compute_metrics(self._circ, weights=self._weights)
        print(
            f"Step {self._step_count} | cost={m.cost:.3f} | "
            f"gates={m.gate_count} | depth={m.depth} | cx={m.cx_count} | swap={m.swap_count}"
        )

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
        self._priority_profile_id = str(profile).strip().lower()
        self._weights = resolve_priority_weights(
            profile_id=self._priority_profile_id,
            override_weights=self._config.priority_weights,
        )
        self._constraint_id = self._profile_to_id(self._priority_profile_id)
        # Update last_cost to match new weights for consistent reward deltas.
        if self._circ is not None:
            self._last_cost = compute_cost(self._circ, weights=self._weights)
            self._action_mask = self._compute_action_mask()


# -----------------------------
# Example builder + self-test
# -----------------------------

def _example_builder(pad_level: int) -> QuantumCircuit:
    """Minimal example circuit builder (debug-friendly)."""
    from core.circuits_baseline import build_toy_circuit

    return build_toy_circuit(pad_level)


def _quick_demo() -> None:
    """
    Quick manual sanity test:
        python -m core.env_quantum_opt

    By default, this demo shows only "demo-visible" baselines (meaningful choices).
    If your circuits_baseline.py does not define DEMO_BUILDERS/get_demo_builder,
    it falls back to BASELINE_BUILDERS/get_builder.
    """
    import argparse

    # Prefer demo-visible builders if your baseline file defines them.
    try:
        from core.circuits_baseline import DEMO_BUILDERS as _CHOICES, get_demo_builder as _GET
        is_demo = True
    except Exception:
        from core.circuits_baseline import BASELINE_BUILDERS as _CHOICES, get_builder as _GET
        is_demo = False

    parser = argparse.ArgumentParser(description="Quick demo for QuantumOptEnv.")
    default_baseline = sorted(_CHOICES.keys())[0] if _CHOICES else "toy"
    parser.add_argument(
        "--baseline",
        default=default_baseline,
        choices=sorted(_CHOICES.keys()),
        help=("Baseline circuit name (demo choices)" if is_demo else "Baseline circuit name"),
    )
    parser.add_argument("--pad-level", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=10)
    args = parser.parse_args()

    builder = _GET(args.baseline)
    env = QuantumOptEnv(
        circuit_builder=builder,
        pad_level=args.pad_level,
        config=EnvConfig(max_steps=args.max_steps),
    )
    obs, info = env.reset()
    print("Reset obs:", obs)
    print("Reset info:", info["metrics"])

    for _ in range(args.max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        print("  action:", info.get("last_action", {}), "reward:", reward)
        if terminated or truncated:
            print("Episode end.", "terminated=", terminated, "truncated=", truncated)
            break


if __name__ == "__main__":
    _quick_demo()
