# core/train_policy.py
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from qiskit.circuit import QuantumCircuit

from core.env_quantum_opt import QuantumOptEnv, EnvConfig, CircuitBuilder
import shutil



# -----------------------------
# Output paths (no console spam)
# -----------------------------
POLICY_DIR = Path(__file__).resolve().parent / "policy_store"
# clear policy_store at start
if POLICY_DIR.exists():
    shutil.rmtree(POLICY_DIR)
POLICY_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = POLICY_DIR / "ppo_model.zip"
BEST_MODEL_PATH = POLICY_DIR / "ppo_model_best.zip"

CURVE_PNG = POLICY_DIR / "training_curve.png"
CURVE_JSON = POLICY_DIR / "training_curve.json"

SUMMARIES_JSON = POLICY_DIR / "circuit_summaries.json"


# -----------------------------
# Baseline builder selection
# -----------------------------
def _get_builder_registry() -> Tuple[Dict[str, CircuitBuilder], Callable[[str], CircuitBuilder]]:
    """
    Prefer demo-visible builders if available, else fall back to baseline builders.
    Returns:
      - dict of builder names -> builder factory
      - getter function get_builder(name) -> builder
    """
    try:
        from core.circuits_baseline import DEMO_BUILDERS as REG, get_demo_builder as GET  # type: ignore
        return dict(REG), GET
    except Exception:
        from core.circuits_baseline import BASELINE_BUILDERS as REG, get_builder as GET  # type: ignore
        return dict(REG), GET


def make_random_mixed_builder(seed: int = 0) -> CircuitBuilder:
    """
    Returns a CircuitBuilder(pad_level_ignored) -> QuantumCircuit that:
      - randomly chooses which baseline circuit to use each reset
      - randomly chooses pad_level each reset for diversity
    """
    REG, GET = _get_builder_registry()
    names = sorted(REG.keys())
    if not names:
        # ultimate fallback
        from core.circuits_baseline import build_toy_circuit

        def fallback_builder(pad_level: int) -> QuantumCircuit:
            return build_toy_circuit(pad_level)

        return fallback_builder

    rng = np.random.default_rng(seed)

    def builder(_pad_level_ignored: int) -> QuantumCircuit:
        name = names[int(rng.integers(0, len(names)))]
        pad = int(rng.integers(1, 4))  # 1..3 inclusive
        return GET(name)(pad)

    return builder


# -----------------------------
# Env factory
# -----------------------------
def make_env(
    circuit_builder: CircuitBuilder,
    pad_level: int,
    constraint_profile: str,
    seed: Optional[int],
    *,
    max_steps: Optional[int] = None,
    stall_patience: Optional[int] = None,
) -> QuantumOptEnv:
    cfg = EnvConfig(
        constraint_profile=constraint_profile,
        max_steps=max_steps if max_steps is not None else EnvConfig().max_steps,
        stall_patience=stall_patience if stall_patience is not None else EnvConfig().stall_patience,
    )
    return QuantumOptEnv(
        circuit_builder=circuit_builder,
        pad_level=pad_level,
        config=cfg,
        seed=seed,
    )


# -----------------------------
# Plot helpers
# -----------------------------
def _save_training_curve(history: Dict[str, List[float]]) -> None:
    CURVE_JSON.write_text(json.dumps(history, indent=2))

    x = np.array(history["timesteps"], dtype=np.int64)
    y = np.array(history["mean_reward"], dtype=np.float32)
    s = np.array(history["std_reward"], dtype=np.float32)

    plt.figure()
    plt.plot(x, y)
    plt.fill_between(x, y - s, y + s, alpha=0.2)
    plt.xlabel("Timesteps")
    plt.ylabel("Eval mean reward")
    plt.title("Training curve (evaluation reward)")
    plt.tight_layout()
    plt.savefig(CURVE_PNG)
    plt.close()


def _draw_circuit_on_ax(ax, circ: QuantumCircuit, title: str) -> None:
    ax.set_title(title)
    ax.axis("off")
    try:
        circ.draw(output="mpl", ax=ax)
    except Exception:
        txt = circ.draw(output="text").single_string()
        ax.text(0.0, 1.0, txt, va="top", ha="left", family="monospace", fontsize=8)
        ax.axis("off")


def _rollout_policy(env: QuantumOptEnv, model: PPO, deterministic: bool = True) -> Dict:
    obs, info = env.reset(seed=0)
    init_circ = env.get_circuit()
    init_metrics = info.get("metrics", {})

    costs: List[float] = []
    if "cost" in init_metrics:
        costs.append(float(init_metrics["cost"]))

    done = False
    last_info = info
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(int(action))
        last_info = info
        m = info.get("metrics", {})
        if "cost" in m:
            costs.append(float(m["cost"]))
        done = bool(terminated or truncated)

    final_circ = env.get_circuit()
    final_metrics = last_info.get("metrics", {})

    return {
        "initial_circuit": init_circ,
        "final_circuit": final_circ,
        "costs": costs,
        "initial_metrics": init_metrics,
        "final_metrics": final_metrics,
    }


def _save_circuit_comparison(name: str, rollout: Dict) -> Path:
    out = POLICY_DIR / f"circuit_{name}_comparison.png"

    costs = rollout["costs"]
    init_m = rollout["initial_metrics"]
    fin_m = rollout["final_metrics"]

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2])

    ax0 = fig.add_subplot(gs[0, :])
    if len(costs) >= 2:
        ax0.plot(range(len(costs)), costs)
        ax0.set_xlabel("Step")
        ax0.set_ylabel("Cost")
        ax0.set_title("Cost over episode (lower is better)")
    else:
        ax0.text(0.5, 0.5, "Cost not available in info['metrics']", ha="center", va="center")
        ax0.axis("off")

    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])

    init_title = f"Initial | cost={init_m.get('cost','n/a')} gates={init_m.get('gate_count','n/a')} depth={init_m.get('depth','n/a')}"
    fin_title = f"Final   | cost={fin_m.get('cost','n/a')} gates={fin_m.get('gate_count','n/a')} depth={fin_m.get('depth','n/a')}"

    _draw_circuit_on_ax(ax1, rollout["initial_circuit"], init_title)
    _draw_circuit_on_ax(ax2, rollout["final_circuit"], fin_title)

    fig.suptitle(f"Circuit optimization: {name}", fontsize=14)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


# -----------------------------
# Training (quiet)
# -----------------------------
def train(
    total_steps: int = 200_000,
    n_envs: int = 4,
    seed: int = 0,
    eval_every: int = 50_000,
    eval_episodes: int = 20,
    device: str = "auto",
    pad_level: int = 1,
    constraint_profile: str = "balanced",
) -> None:
    # Train on a MIX of circuits + varying pad levels to avoid trivial convergence
    mixed_builder = make_random_mixed_builder(seed=seed)

    def env_fn():
        # pad_level is ignored by mixed_builder, but env requires a value anyway
        return make_env(
            circuit_builder=mixed_builder,
            pad_level=pad_level,
            constraint_profile=constraint_profile,
            seed=None,
        )

    vec_env = make_vec_env(env_fn, n_envs=n_envs, seed=seed)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=0,  # no console tables/log spam
        device=device,
        seed=seed,
    )

    history = {"timesteps": [], "mean_reward": [], "std_reward": []}
    best_mean_reward = -1e18

    steps_done = 0
    while steps_done < total_steps:
        chunk = min(eval_every, total_steps - steps_done)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        steps_done += chunk

        # Evaluate on the SAME mixed distribution (so curve reflects general learning)
        eval_env = make_env(
            circuit_builder=make_random_mixed_builder(seed=seed + 12345),
            pad_level=pad_level,
            constraint_profile=constraint_profile,
            seed=seed + 999,
        )
        mean_r, std_r = evaluate_policy(
            model, eval_env, n_eval_episodes=eval_episodes, deterministic=True
        )
        eval_env.close()

        history["timesteps"].append(int(steps_done))
        history["mean_reward"].append(float(mean_r))
        history["std_reward"].append(float(std_r))
        _save_training_curve(history)

        model.save(str(MODEL_PATH))
        if mean_r > best_mean_reward:
            best_mean_reward = float(mean_r)
            model.save(str(BEST_MODEL_PATH))

    vec_env.close()

    # Per-circuit "initial vs improved" comparisons (deterministic rollout)
    REG, GET = _get_builder_registry()
    summaries: Dict[str, Dict] = {}

    for name in sorted(REG.keys()):
        builder = GET(name)

        env = make_env(
            circuit_builder=builder,
            pad_level=pad_level,
            constraint_profile=constraint_profile,
            seed=0,
        )
        rollout = _rollout_policy(env, model, deterministic=True)
        env.close()

        fig_path = _save_circuit_comparison(name, rollout)

        summaries[name] = {
            "figure": str(fig_path),
            "initial_metrics": rollout["initial_metrics"],
            "final_metrics": rollout["final_metrics"],
        }

    SUMMARIES_JSON.write_text(json.dumps(summaries, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-steps", type=int, default=200_000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=50_000)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--pad-level", type=int, default=1)
    ap.add_argument("--constraint-profile", type=str, default="balanced")
    args = ap.parse_args()

    train(
        total_steps=args.total_steps,
        n_envs=args.n_envs,
        seed=args.seed,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        device=args.device,
        pad_level=args.pad_level,
        constraint_profile=args.constraint_profile,
    )


if __name__ == "__main__":
    main()