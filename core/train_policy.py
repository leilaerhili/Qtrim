"""
Train PPO policy and save to core/policy_store/.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from core.circuits_baseline import get_builder
from core.env_quantum_opt import EnvConfig, QuantumOptEnv


POLICY_STORE = Path(__file__).resolve().parent / "policy_store"


def _make_env(
    baseline: str,
    pad_level: int,
    constraint_profile: str,
    seed: Optional[int] = None,
    monitor: bool = False,
) -> QuantumOptEnv:
    builder = get_builder(baseline)
    config = EnvConfig(constraint_profile=constraint_profile)
    env = QuantumOptEnv(circuit_builder=builder, pad_level=pad_level, config=config, seed=seed)
    if monitor:
        return Monitor(env)
    return env


def train_policy(
    baseline: str = "toy",
    pad_level: int = 2,
    constraint_profile: str = "balanced",
    total_timesteps: int = 50_000,
    seed: Optional[int] = 0,
) -> PPO:
    env = DummyVecEnv([lambda: _make_env(baseline, pad_level, constraint_profile, seed, monitor=True)])
    model = PPO("MlpPolicy", env, verbose=1, seed=seed)
    model.learn(total_timesteps=total_timesteps)
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


def _main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO policy for QuantumOptEnv.")
    parser.add_argument("--baseline", default="toy", choices=["toy", "ghz", "line"])
    parser.add_argument("--pad-level", type=int, default=2)
    parser.add_argument("--constraint-profile", default="balanced")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-name", default="ppo_quantum_opt")
    parser.add_argument("--eval-episodes", type=int, default=5)
    args = parser.parse_args()

    model = train_policy(
        baseline=args.baseline,
        pad_level=args.pad_level,
        constraint_profile=args.constraint_profile,
        total_timesteps=args.timesteps,
        seed=args.seed,
    )

    path = save_policy(model, args.save_name)
    mean_reward, std_reward = evaluate(
        model,
        baseline=args.baseline,
        pad_level=args.pad_level,
        constraint_profile=args.constraint_profile,
        n_eval_episodes=args.eval_episodes,
    )

    print(f"Saved policy to: {path}")
    print(f"Eval mean reward: {mean_reward:.3f} +/- {std_reward:.3f}")


if __name__ == "__main__":
    _main()
