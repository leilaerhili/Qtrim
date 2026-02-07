"""
Curriculum training for QuantumOptEnv.

Trains PPO across increasingly difficult phases, then evaluates on
unseen seeded challenge circuits.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
import shutil
from statistics import mean, pstdev
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from core.accelerator import format_device_resolution, resolve_training_device
from core.circuits_baseline import get_builder, make_seeded_challenge_builder
from core.env_quantum_opt import EnvConfig, QuantumOptEnv


POLICY_STORE = Path(__file__).resolve().parent / "policy_store"

# Staged action sets to reduce sparse/no-op exploration early in training.
ACTION_STAGE_EASY: Tuple[int, ...] = tuple(range(0, 20))
ACTION_STAGE_MEDIUM: Tuple[int, ...] = tuple(range(0, 34))
ACTION_STAGE_FULL: Tuple[int, ...] = tuple(range(0, 41))


@dataclass(frozen=True)
class CurriculumPhase:
    name: str
    train_baselines: Tuple[str, ...]
    timesteps: int
    pad_level_min: int
    pad_level_max: int
    allowed_action_ids: Tuple[int, ...]
    challenge_seed_start: Optional[int] = None
    challenge_seed_count: int = 0


def _phase_builder(
    phase: CurriculumPhase,
    rng: random.Random,
) -> Callable[[int], "QuantumCircuit"]:
    if phase.challenge_seed_start is None:
        builders = [get_builder(name) for name in phase.train_baselines]

        def _builder(pad_level: int):
            return rng.choice(builders)(pad_level)
    else:
        seed_low = int(phase.challenge_seed_start)
        seed_high = seed_low + max(1, int(phase.challenge_seed_count)) - 1

        def _builder(pad_level: int):
            seed = rng.randint(seed_low, seed_high)
            builder = make_seeded_challenge_builder(seed=seed, num_qubits=6, depth=36)
            return builder(pad_level)

    return _builder


def _make_phase_env(
    phase: CurriculumPhase,
    constraint_profile: str,
    seed: int,
) -> QuantumOptEnv:
    rng = random.Random(seed * 9973 + 17)
    pad_level = rng.randint(int(phase.pad_level_min), int(phase.pad_level_max))
    builder = _phase_builder(phase, rng)
    config = EnvConfig(
        constraint_profile=constraint_profile,
        max_steps=40,
        stall_patience=10,
        normalize_obs=True,
        allowed_action_ids=phase.allowed_action_ids,
    )
    env = QuantumOptEnv(circuit_builder=builder, pad_level=pad_level, config=config, seed=seed)
    return Monitor(env)


def _make_holdout_env(
    holdout_seed: int,
    constraint_profile: str,
    pad_level: int = 3,
) -> QuantumOptEnv:
    builder = make_seeded_challenge_builder(seed=holdout_seed, num_qubits=6, depth=40)
    config = EnvConfig(
        constraint_profile=constraint_profile,
        max_steps=45,
        stall_patience=10,
        normalize_obs=True,
        allowed_action_ids=ACTION_STAGE_FULL,
    )
    return Monitor(QuantumOptEnv(circuit_builder=builder, pad_level=pad_level, config=config, seed=holdout_seed))


def default_curriculum(total_timesteps: int) -> List[CurriculumPhase]:
    t = int(total_timesteps)
    t1 = max(2_048, int(t * 0.25))
    t2 = max(2_048, int(t * 0.35))
    t3 = max(2_048, t - t1 - t2)
    return [
        CurriculumPhase(
            name="easy",
            train_baselines=("toy", "parity"),
            timesteps=t1,
            pad_level_min=1,
            pad_level_max=2,
            allowed_action_ids=ACTION_STAGE_EASY,
        ),
        CurriculumPhase(
            name="medium",
            train_baselines=("half_adder", "line", "majority"),
            timesteps=t2,
            pad_level_min=2,
            pad_level_max=3,
            allowed_action_ids=ACTION_STAGE_MEDIUM,
        ),
        CurriculumPhase(
            name="hard",
            train_baselines=tuple(),
            timesteps=t3,
            pad_level_min=3,
            pad_level_max=4,
            allowed_action_ids=ACTION_STAGE_FULL,
            challenge_seed_start=2000,
            challenge_seed_count=200,
        ),
    ]


def parse_seed_list(raw: str) -> List[int]:
    parts = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not parts:
        raise ValueError("seeds must contain at least one integer.")
    return [int(x) for x in parts]


def train_curriculum_for_seed(
    *,
    seed: int,
    total_timesteps: int,
    constraint_profile: str,
    ent_coef: float = 0.01,
    learning_rate: float = 3e-4,
    n_steps: int = 1024,
    batch_size: int = 256,
    device: str = "auto",
    strict_device: bool = False,
    device_resolution_out: Optional[Dict[str, Any]] = None,
) -> PPO:
    resolved_device, device_resolution = resolve_training_device(
        requested=device,
        strict=bool(strict_device),
    )
    print(format_device_resolution(device_resolution))
    if device_resolution_out is not None:
        device_resolution_out.clear()
        device_resolution_out.update(device_resolution.to_dict())

    phases = default_curriculum(total_timesteps)
    first_env = DummyVecEnv(
        [lambda: _make_phase_env(phases[0], constraint_profile=constraint_profile, seed=seed)]
    )
    model = PPO(
        "MlpPolicy",
        first_env,
        verbose=1,
        seed=seed,
        ent_coef=float(ent_coef),
        learning_rate=float(learning_rate),
        n_steps=int(n_steps),
        batch_size=int(batch_size),
        device=resolved_device,
    )

    for i, phase in enumerate(phases):
        env = DummyVecEnv(
            [lambda p=phase, s=seed + i * 101: _make_phase_env(p, constraint_profile, s)]
        )
        model.set_env(env)
        model.learn(total_timesteps=int(phase.timesteps), reset_num_timesteps=False)
    return model


def evaluate_holdout(
    model: PPO,
    *,
    holdout_seed_start: int,
    holdout_count: int,
    constraint_profile: str,
    n_eval_episodes: int,
) -> Tuple[float, float]:
    scores: List[float] = []
    for hs in range(int(holdout_seed_start), int(holdout_seed_start) + int(holdout_count)):
        env = DummyVecEnv([lambda seed=hs: _make_holdout_env(seed, constraint_profile)])
        m, _ = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=True)
        scores.append(float(m))
    return float(mean(scores)), float(pstdev(scores)) if len(scores) > 1 else 0.0


def run_multi_seed(
    *,
    seeds: Sequence[int],
    total_timesteps: int,
    constraint_profile: str,
    save_name: str,
    holdout_seed_start: int,
    holdout_count: int,
    eval_episodes: int,
    ent_coef: float,
    learning_rate: float,
    n_steps: int,
    batch_size: int,
    device: str,
    strict_device: bool,
) -> Dict[str, object]:
    POLICY_STORE.mkdir(parents=True, exist_ok=True)
    runs: List[Dict[str, object]] = []

    for s in seeds:
        device_resolution_data: Dict[str, Any] = {}
        model = train_curriculum_for_seed(
            seed=int(s),
            total_timesteps=total_timesteps,
            constraint_profile=constraint_profile,
            ent_coef=ent_coef,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            device=device,
            strict_device=strict_device,
            device_resolution_out=device_resolution_data,
        )
        model_path = POLICY_STORE / f"{save_name}_seed{s}.zip"
        model.save(model_path)
        hold_mean, hold_std = evaluate_holdout(
            model,
            holdout_seed_start=holdout_seed_start,
            holdout_count=holdout_count,
            constraint_profile=constraint_profile,
            n_eval_episodes=eval_episodes,
        )
        runs.append(
            {
                "seed": int(s),
                "policy_path": str(model_path),
                "holdout_mean_reward": float(hold_mean),
                "holdout_std_reward": float(hold_std),
                "device_resolution": dict(device_resolution_data),
            }
        )

    summary: Dict[str, object] = {
        "seeds": [int(s) for s in seeds],
        "total_timesteps": int(total_timesteps),
        "constraint_profile": constraint_profile,
        "ppo_hparams": {
            "ent_coef": float(ent_coef),
            "learning_rate": float(learning_rate),
            "n_steps": int(n_steps),
            "batch_size": int(batch_size),
            "device": str(device),
            "strict_device": bool(strict_device),
        },
        "holdout_seed_start": int(holdout_seed_start),
        "holdout_count": int(holdout_count),
        "runs": runs,
        "mean_holdout_reward_across_seeds": float(mean([r["holdout_mean_reward"] for r in runs])),
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
    out_path = POLICY_STORE / f"{save_name}_curriculum_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(out_path)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Curriculum PPO training for QuantumOptEnv.")
    parser.add_argument("--timesteps", type=int, default=60_000)
    parser.add_argument("--constraint-profile", default="balanced")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--holdout-seed-start", type=int, default=10_000)
    parser.add_argument("--holdout-count", type=int, default=12)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--save-name", default="ppo_quantum_curriculum")
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--device",
        default="auto",
        help="Device preference: auto, npu, directml, cuda, xpu, mps, cpu.",
    )
    parser.add_argument(
        "--strict-device",
        action="store_true",
        help="Fail immediately if the requested device is unavailable.",
    )
    return parser


def _main() -> None:
    args = build_arg_parser().parse_args()
    seeds = parse_seed_list(args.seeds)
    summary = run_multi_seed(
        seeds=seeds,
        total_timesteps=args.timesteps,
        constraint_profile=args.constraint_profile,
        save_name=args.save_name,
        holdout_seed_start=args.holdout_seed_start,
        holdout_count=args.holdout_count,
        eval_episodes=args.eval_episodes,
        ent_coef=args.ent_coef,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        device=args.device,
        strict_device=args.strict_device,
    )
    print(f"Saved summary to: {summary['summary_path']}")
    print(
        "Mean holdout reward across seeds: "
        f"{summary['mean_holdout_reward_across_seeds']:.3f}"
    )
    print(f"Best seed by holdout: {summary['best_seed']}")
    print(f"Best policy: {summary['best_policy_path']}")


if __name__ == "__main__":
    _main()
