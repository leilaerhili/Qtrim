"""
Train PPO policy and save to core/policy_store/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from statistics import mean, pstdev
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from core.circuits_baseline import BASELINE_BUILDERS, get_builder, make_seeded_challenge_builder
from core.env_quantum_opt import EnvConfig, QuantumOptEnv


POLICY_STORE = Path(__file__).resolve().parent / "policy_store"


def make_random_mixed_builder(
    seed: int = 0,
    pad_level_min: int = 1,
    pad_level_max: int = 3,
) -> Callable[[int], "QuantumCircuit"]:
    """
    Create a baseline builder that samples circuit families and pad levels per reset.
    """
    import math
    from qiskit.circuit import QuantumCircuit

    names = sorted(BASELINE_BUILDERS.keys())
    if not names:
        raise RuntimeError("No baseline builders are registered.")
    lo = int(min(pad_level_min, pad_level_max))
    hi = int(max(pad_level_min, pad_level_max))
    rng = np.random.default_rng(seed)

    def _builder(_: int) -> QuantumCircuit:
        name = names[int(rng.integers(0, len(names)))]
        pad = int(rng.integers(lo, hi + 1))
        # Safety for weird bounds.
        if not math.isfinite(pad):
            pad = lo
        return get_builder(name)(pad)

    return _builder


def _make_env_with_builder(
    circuit_builder: Callable[[int], "QuantumCircuit"],
    pad_level: int,
    constraint_profile: str,
    seed: Optional[int] = None,
    monitor: bool = False,
) -> QuantumOptEnv:
    config = EnvConfig(constraint_profile=constraint_profile)
    env = QuantumOptEnv(circuit_builder=circuit_builder, pad_level=pad_level, config=config, seed=seed)
    if monitor:
        return Monitor(env)
    return env


def _make_env(
    baseline: str,
    pad_level: int,
    constraint_profile: str,
    seed: Optional[int] = None,
    monitor: bool = False,
) -> QuantumOptEnv:
    return _make_env_with_builder(
        circuit_builder=get_builder(baseline),
        pad_level=pad_level,
        constraint_profile=constraint_profile,
        seed=seed,
        monitor=monitor,
    )


def _make_holdout_env(
    seed: int,
    pad_level: int,
    constraint_profile: str,
    monitor: bool = False,
) -> QuantumOptEnv:
    builder = make_seeded_challenge_builder(seed=seed, num_qubits=6, depth=36)
    return _make_env_with_builder(
        circuit_builder=builder,
        pad_level=pad_level,
        constraint_profile=constraint_profile,
        seed=seed,
        monitor=monitor,
    )


def _make_train_vec_env(
    *,
    baseline: str,
    pad_level: int,
    constraint_profile: str,
    seed: Optional[int],
    train_mode: str,
    n_envs: int,
    mixed_pad_level_min: int,
    mixed_pad_level_max: int,
) -> DummyVecEnv:
    num_envs = max(1, int(n_envs))
    base_seed = int(seed) if seed is not None else 0
    env_fns = []
    for i in range(num_envs):
        env_seed = (base_seed + i * 997) if seed is not None else None
        if train_mode == "mixed":
            mixed_builder = make_random_mixed_builder(
                seed=base_seed + 1009 * (i + 1),
                pad_level_min=mixed_pad_level_min,
                pad_level_max=mixed_pad_level_max,
            )
            env_fns.append(
                lambda b=mixed_builder, s=env_seed: _make_env_with_builder(
                    b,
                    pad_level=pad_level,
                    constraint_profile=constraint_profile,
                    seed=s,
                    monitor=True,
                )
            )
        else:
            env_fns.append(
                lambda s=env_seed: _make_env(
                    baseline=baseline,
                    pad_level=pad_level,
                    constraint_profile=constraint_profile,
                    seed=s,
                    monitor=True,
                )
            )
    return DummyVecEnv(env_fns)


def _evaluate_training_curve_point(
    model: PPO,
    *,
    baseline: str,
    pad_level: int,
    constraint_profile: str,
    train_mode: str,
    seed: Optional[int],
    n_eval_episodes: int,
    mixed_pad_level_min: int,
    mixed_pad_level_max: int,
) -> Tuple[float, float]:
    if train_mode == "mixed":
        eval_seed = (int(seed) if seed is not None else 0) + 4242
        mixed_builder = make_random_mixed_builder(
            seed=eval_seed,
            pad_level_min=mixed_pad_level_min,
            pad_level_max=mixed_pad_level_max,
        )
        env = DummyVecEnv(
            [
                lambda: _make_env_with_builder(
                    mixed_builder,
                    pad_level=pad_level,
                    constraint_profile=constraint_profile,
                    seed=eval_seed,
                    monitor=True,
                )
            ]
        )
        mean_reward, std_reward = evaluate_policy(
            model,
            env,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
        )
        env.close()
        return float(mean_reward), float(std_reward)
    return evaluate(
        model,
        baseline=baseline,
        pad_level=pad_level,
        constraint_profile=constraint_profile,
        n_eval_episodes=n_eval_episodes,
    )


def _save_training_curve(
    history: Dict[str, List[float]],
    curve_json_path: Path,
    curve_png_path: Optional[Path],
) -> None:
    curve_json_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    if curve_png_path is None:
        return
    if not history["timesteps"]:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    x = np.array(history["timesteps"], dtype=np.int64)
    y = np.array(history["mean_reward"], dtype=np.float32)
    s = np.array(history["std_reward"], dtype=np.float32)

    plt.figure(figsize=(8, 4))
    plt.plot(x, y, label="mean_reward")
    plt.fill_between(x, y - s, y + s, alpha=0.2, label="std")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.title("Training Evaluation Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_png_path, dpi=160)
    plt.close()


def train_policy(
    baseline: str = "toy",
    pad_level: int = 2,
    constraint_profile: str = "balanced",
    total_timesteps: int = 50_000,
    seed: Optional[int] = 0,
    ent_coef: float = 0.01,
    learning_rate: float = 3e-4,
    n_steps: int = 1024,
    batch_size: int = 256,
    n_envs: int = 1,
    device: str = "auto",
    train_mode: str = "fixed",
    mixed_pad_level_min: int = 1,
    mixed_pad_level_max: int = 3,
    eval_every: int = 0,
    eval_episodes_curve: int = 5,
    curve_json_path: Optional[Path] = None,
    curve_png_path: Optional[Path] = None,
) -> PPO:
    env = _make_train_vec_env(
        baseline=baseline,
        pad_level=pad_level,
        constraint_profile=constraint_profile,
        seed=seed,
        train_mode=train_mode,
        n_envs=n_envs,
        mixed_pad_level_min=mixed_pad_level_min,
        mixed_pad_level_max=mixed_pad_level_max,
    )
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=seed,
        ent_coef=float(ent_coef),
        learning_rate=float(learning_rate),
        n_steps=int(n_steps),
        batch_size=int(batch_size),
        device=device,
    )
    if int(eval_every) > 0:
        history = {"timesteps": [], "mean_reward": [], "std_reward": []}
        steps_done = 0
        while steps_done < int(total_timesteps):
            chunk = min(int(eval_every), int(total_timesteps) - steps_done)
            model.learn(total_timesteps=chunk, reset_num_timesteps=False)
            steps_done += chunk
            mean_r, std_r = _evaluate_training_curve_point(
                model,
                baseline=baseline,
                pad_level=pad_level,
                constraint_profile=constraint_profile,
                train_mode=train_mode,
                seed=seed,
                n_eval_episodes=max(1, int(eval_episodes_curve)),
                mixed_pad_level_min=mixed_pad_level_min,
                mixed_pad_level_max=mixed_pad_level_max,
            )
            history["timesteps"].append(int(steps_done))
            history["mean_reward"].append(float(mean_r))
            history["std_reward"].append(float(std_r))
        if curve_json_path is not None:
            _save_training_curve(
                history,
                curve_json_path=curve_json_path,
                curve_png_path=curve_png_path,
            )
    else:
        model.learn(total_timesteps=total_timesteps)
    env.close()
    return model


def save_policy(model: PPO, name: str) -> Path:
    POLICY_STORE.mkdir(parents=True, exist_ok=True)
    path = POLICY_STORE / f"{name}.zip"
    model.save(path)
    return path


def load_policy(
    name: str,
    baseline: str = "toy",
    pad_level: int = 2,
    constraint_profile: str = "balanced",
) -> PPO:
    env = DummyVecEnv([lambda: _make_env(baseline, pad_level, constraint_profile, seed=0, monitor=True)])
    path = POLICY_STORE / f"{name}.zip"
    return PPO.load(path, env=env)


def get_action(model: PPO, obs: np.ndarray, deterministic: bool = True) -> int:
    action, _ = model.predict(obs, deterministic=deterministic)
    return int(action)


def evaluate(
    model: PPO,
    baseline: str,
    pad_level: int,
    constraint_profile: str,
    n_eval_episodes: int = 5,
) -> Tuple[float, float]:
    env = DummyVecEnv([lambda: _make_env(baseline, pad_level, constraint_profile, seed=123, monitor=True)])
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
    )
    return float(mean_reward), float(std_reward)


def evaluate_holdout(
    model: PPO,
    holdout_seed_start: int = 10_000,
    holdout_count: int = 8,
    pad_level: int = 3,
    constraint_profile: str = "balanced",
    n_eval_episodes: int = 2,
) -> Tuple[float, float]:
    scores: List[float] = []
    for s in range(int(holdout_seed_start), int(holdout_seed_start) + int(holdout_count)):
        env = DummyVecEnv(
            [lambda seed=s: _make_holdout_env(seed, pad_level, constraint_profile, monitor=True)]
        )
        score, _ = evaluate_policy(
            model,
            env,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
        )
        scores.append(float(score))
    return float(mean(scores)), float(pstdev(scores)) if len(scores) > 1 else 0.0


def parse_seed_list(raw: str) -> List[int]:
    values = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not values:
        raise ValueError("seeds must contain at least one integer.")
    return [int(x) for x in values]


def train_and_evaluate_seeds(
    *,
    baseline: str,
    pad_level: int,
    constraint_profile: str,
    total_timesteps: int,
    seeds: List[int],
    save_name: str,
    eval_episodes: int,
    holdout_seed_start: int,
    holdout_count: int,
    ent_coef: float,
    learning_rate: float,
    n_steps: int,
    batch_size: int,
    n_envs: int,
    device: str,
    train_mode: str,
    mixed_pad_level_min: int,
    mixed_pad_level_max: int,
    eval_every: int,
    eval_episodes_curve: int,
) -> Dict[str, object]:
    POLICY_STORE.mkdir(parents=True, exist_ok=True)
    runs: List[Dict[str, float | int | str]] = []

    for seed in seeds:
        run_name = f"{save_name}_seed{seed}"
        curve_json = POLICY_STORE / f"{run_name}_curve.json"
        curve_png = POLICY_STORE / f"{run_name}_curve.png"
        model = train_policy(
            baseline=baseline,
            pad_level=pad_level,
            constraint_profile=constraint_profile,
            total_timesteps=total_timesteps,
            seed=seed,
            ent_coef=ent_coef,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_envs=n_envs,
            device=device,
            train_mode=train_mode,
            mixed_pad_level_min=mixed_pad_level_min,
            mixed_pad_level_max=mixed_pad_level_max,
            eval_every=eval_every,
            eval_episodes_curve=eval_episodes_curve,
            curve_json_path=curve_json if eval_every > 0 else None,
            curve_png_path=curve_png if eval_every > 0 else None,
        )
        path = save_policy(model, run_name)
        eval_mean, eval_std = evaluate(
            model,
            baseline=baseline,
            pad_level=pad_level,
            constraint_profile=constraint_profile,
            n_eval_episodes=eval_episodes,
        )
        holdout_mean, holdout_std = evaluate_holdout(
            model,
            holdout_seed_start=holdout_seed_start,
            holdout_count=holdout_count,
            pad_level=max(3, pad_level),
            constraint_profile=constraint_profile,
            n_eval_episodes=max(1, eval_episodes // 2),
        )
        runs.append(
            {
                "seed": int(seed),
                "policy_path": str(path),
                "eval_mean_reward": float(eval_mean),
                "eval_std_reward": float(eval_std),
                "holdout_mean_reward": float(holdout_mean),
                "holdout_std_reward": float(holdout_std),
                "training_curve_json": str(curve_json) if eval_every > 0 else None,
                "training_curve_png": str(curve_png) if eval_every > 0 else None,
            }
        )

    summary: Dict[str, object] = {
        "baseline": baseline,
        "pad_level": int(pad_level),
        "constraint_profile": constraint_profile,
        "timesteps": int(total_timesteps),
        "seeds": [int(s) for s in seeds],
        "ppo_hparams": {
            "ent_coef": float(ent_coef),
            "learning_rate": float(learning_rate),
            "n_steps": int(n_steps),
            "batch_size": int(batch_size),
            "n_envs": int(n_envs),
            "device": str(device),
            "train_mode": str(train_mode),
            "mixed_pad_level_min": int(mixed_pad_level_min),
            "mixed_pad_level_max": int(mixed_pad_level_max),
        },
        "runs": runs,
        "mean_eval_reward_across_seeds": float(mean([float(r["eval_mean_reward"]) for r in runs])),
        "mean_holdout_reward_across_seeds": float(
            mean([float(r["holdout_mean_reward"]) for r in runs])
        ),
    }
    best = max(runs, key=lambda r: float(r["holdout_mean_reward"]))
    best_seed = int(best["seed"])
    best_policy_src = Path(str(best["policy_path"]))
    best_policy_dst = POLICY_STORE / f"{save_name}_best.zip"
    if best_policy_src.exists():
        shutil.copy2(best_policy_src, best_policy_dst)
    summary["best_seed"] = best_seed
    summary["best_policy_path"] = str(best_policy_dst)
    summary["best_holdout_mean_reward"] = float(best["holdout_mean_reward"])
    summary_path = POLICY_STORE / f"{save_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PPO policy for QuantumOptEnv.")
    baseline_choices = sorted(BASELINE_BUILDERS.keys())
    default_baseline = "toy" if "toy" in BASELINE_BUILDERS else baseline_choices[0]
    parser.add_argument("--baseline", default=default_baseline, choices=baseline_choices)
    parser.add_argument("--pad-level", type=int, default=2)
    parser.add_argument("--constraint-profile", default="balanced")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-name", default="ppo_quantum_opt")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument(
        "--seeds",
        default="0",
        help="Comma-separated list of seeds for multi-seed runs, e.g. 0,1,2.",
    )
    parser.add_argument("--holdout-seed-start", type=int, default=10000)
    parser.add_argument("--holdout-count", type=int, default=8)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-mode", choices=["fixed", "mixed"], default="fixed")
    parser.add_argument("--mixed-pad-min", type=int, default=1)
    parser.add_argument("--mixed-pad-max", type=int, default=3)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="If >0, evaluate periodically during training and save reward curves.",
    )
    parser.add_argument("--curve-eval-episodes", type=int, default=5)
    return parser


def _main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    seeds = parse_seed_list(args.seeds)
    summary = train_and_evaluate_seeds(
        baseline=args.baseline,
        pad_level=args.pad_level,
        constraint_profile=args.constraint_profile,
        total_timesteps=args.timesteps,
        seeds=seeds,
        save_name=args.save_name,
        eval_episodes=args.eval_episodes,
        holdout_seed_start=args.holdout_seed_start,
        holdout_count=args.holdout_count,
        ent_coef=args.ent_coef,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_envs=args.n_envs,
        device=args.device,
        train_mode=args.train_mode,
        mixed_pad_level_min=args.mixed_pad_min,
        mixed_pad_level_max=args.mixed_pad_max,
        eval_every=args.eval_every,
        eval_episodes_curve=args.curve_eval_episodes,
    )
    print(f"Saved multi-seed summary to: {summary['summary_path']}")
    print(
        "Mean eval reward across seeds: "
        f"{summary['mean_eval_reward_across_seeds']:.3f}"
    )
    print(
        "Mean holdout reward across seeds: "
        f"{summary['mean_holdout_reward_across_seeds']:.3f}"
    )
    print(f"Best seed by holdout: {summary['best_seed']}")
    print(f"Best policy: {summary['best_policy_path']}")


if __name__ == "__main__":
    _main()
