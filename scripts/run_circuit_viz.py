"""
Run circuit optimization episodes and visualize results.

Usage examples:
  python scripts/run_circuit_viz.py --circuit parity --policy core/policy_store/ppo_curriculum_tiny_seed0.zip
  python scripts/run_circuit_viz.py --circuit challenge_medium --seed 42 --max-steps 50
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import sys

import matplotlib.pyplot as plt

from qiskit.circuit import QuantumCircuit

# Ensure repository root is importable when executed as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.circuits_baseline import (
    BASELINE_BUILDERS,
    get_builder,
    make_seeded_challenge_builder,
)
from core.env_quantum_opt import EnvConfig, QuantumOptEnv


try:
    from stable_baselines3 import PPO
except Exception:
    PPO = None  # type: ignore[assignment]


@dataclass
class StepRecord:
    step: int
    action_id: int
    action_name: str
    changed: bool
    reward: float
    cost: float
    gate_count: int
    depth: int
    cx_count: int


def build_circuit_factory(name: str, seed: int) -> Callable[[int], QuantumCircuit]:
    key = name.strip().lower()
    if key in BASELINE_BUILDERS:
        return get_builder(key)
    if key == "challenge_easy":
        return make_seeded_challenge_builder(seed=seed, num_qubits=5, depth=22)
    if key == "challenge_medium":
        return make_seeded_challenge_builder(seed=seed, num_qubits=6, depth=34)
    if key == "challenge_hard":
        return make_seeded_challenge_builder(seed=seed, num_qubits=7, depth=46)
    raise KeyError(
        f"Unknown circuit '{name}'. "
        f"Options: {sorted(BASELINE_BUILDERS)} + challenge_easy/challenge_medium/challenge_hard"
    )


def save_circuit_text(circ: QuantumCircuit, path: Path) -> None:
    path.write_text(str(circ.draw(output="text")), encoding="utf-8")


def try_save_circuit_png(circ: QuantumCircuit, path: Path) -> bool:
    try:
        fig = circ.draw(output="mpl")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception:
        return False


def print_safe_terminal(text: str) -> None:
    enc = sys.stdout.encoding or "utf-8"
    safe = text.encode(enc, errors="replace").decode(enc, errors="replace")
    print(safe)


def format_step_table(rows: List[StepRecord]) -> str:
    header = "step | action_id | action_name                      | changed | reward   | cost     | gates | depth | cx"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"{r.step:>4} | {r.action_id:>9} | {r.action_name:<32} | "
            f"{str(r.changed):<7} | {r.reward:>8.3f} | {r.cost:>8.3f} | "
            f"{r.gate_count:>5} | {r.depth:>5} | {r.cx_count:>2}"
        )
    return "\n".join(lines)


def run_episode(
    env: QuantumOptEnv,
    policy_path: Optional[Path],
    deterministic: bool,
) -> Tuple[QuantumCircuit, QuantumCircuit, List[StepRecord]]:
    model = None
    if policy_path is not None:
        if PPO is None:
            raise RuntimeError("stable-baselines3 is not available. Install requirements first.")
        model = PPO.load(str(policy_path))

    obs, info = env.reset()
    start_circuit = env.get_circuit()

    rows: List[StepRecord] = []
    done = False
    truncated = False
    step = 0

    while not (done or truncated):
        if model is None:
            action = int(env.action_space.sample())
        else:
            predicted, _ = model.predict(obs, deterministic=deterministic)
            action = int(predicted)

        obs, reward, done, truncated, info = env.step(action)
        step += 1
        la = info.get("last_action", {})
        m = info["metrics"]
        rows.append(
            StepRecord(
                step=step,
                action_id=int(la.get("action_id", action)),
                action_name=str(la.get("action_name", f"action_{action}")),
                changed=bool(la.get("changed", False)),
                reward=float(reward),
                cost=float(m["cost"]),
                gate_count=int(m["gate_count"]),
                depth=int(m["depth"]),
                cx_count=int(m["cx_count"]),
            )
        )

    end_circuit = env.get_circuit()
    return start_circuit, end_circuit, rows


def save_trajectory_plot(rows: List[StepRecord], path: Path) -> None:
    if not rows:
        return
    steps = [r.step for r in rows]
    rewards = [r.reward for r in rows]
    costs = [r.cost for r in rows]
    gates = [r.gate_count for r in rows]
    depths = [r.depth for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(steps, rewards, marker="o", linewidth=1.4, label="reward")
    axes[0].plot(steps, costs, marker=".", linewidth=1.0, label="cost")
    axes[0].set_ylabel("Reward / Cost")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(steps, gates, marker="o", linewidth=1.4, label="gate_count")
    axes[1].plot(steps, depths, marker=".", linewidth=1.0, label="depth")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Circuit Size")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run and visualize optimized circuit episodes.")
    parser.add_argument(
        "--circuit",
        default="parity",
        help="Circuit name: baseline keys or challenge_easy/challenge_medium/challenge_hard",
    )
    parser.add_argument("--policy", default=None, help="Path to SB3 PPO .zip model. Omit for random actions.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for challenge circuit generators.")
    parser.add_argument("--pad-level", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--constraint-profile", default="balanced")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic model actions.")
    parser.add_argument(
        "--out-dir",
        default="core/policy_store/viz_runs",
        help="Directory for saved plots and circuit diagrams.",
    )
    parser.add_argument(
        "--no-terminal-print",
        action="store_true",
        help="Do not print circuit diagrams and step table in terminal.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    builder = build_circuit_factory(args.circuit, seed=args.seed)
    env = QuantumOptEnv(
        circuit_builder=builder,
        pad_level=args.pad_level,
        config=EnvConfig(
            max_steps=args.max_steps,
            constraint_profile=args.constraint_profile,
        ),
        seed=args.seed,
    )

    policy_path = Path(args.policy).resolve() if args.policy else None
    start_circuit, end_circuit, rows = run_episode(
        env=env,
        policy_path=policy_path,
        deterministic=bool(args.deterministic),
    )

    prefix = f"{args.circuit}_pad{args.pad_level}_seed{args.seed}"
    start_txt = out_dir / f"{prefix}_start.txt"
    end_txt = out_dir / f"{prefix}_end.txt"
    table_txt = out_dir / f"{prefix}_steps.txt"
    traj_png = out_dir / f"{prefix}_trajectory.png"
    start_png = out_dir / f"{prefix}_start.png"
    end_png = out_dir / f"{prefix}_end.png"

    save_circuit_text(start_circuit, start_txt)
    save_circuit_text(end_circuit, end_txt)
    step_table = format_step_table(rows)
    table_txt.write_text(step_table, encoding="utf-8")
    save_trajectory_plot(rows, traj_png)
    start_ok = try_save_circuit_png(start_circuit, start_png)
    end_ok = try_save_circuit_png(end_circuit, end_png)

    if rows:
        first = rows[0]
        last = rows[-1]
        total_reward = sum(r.reward for r in rows)
        print(f"Circuit: {args.circuit}")
        print(f"Steps: {len(rows)}")
        print(
            "Start metrics: "
            f"cost={first.cost:.3f}, gates={first.gate_count}, depth={first.depth}, cx={first.cx_count}"
        )
        print(
            "End metrics:   "
            f"cost={last.cost:.3f}, gates={last.gate_count}, depth={last.depth}, cx={last.cx_count}"
        )
        print(f"Total episode reward: {total_reward:.3f}")
    else:
        print("No steps were executed.")

    if not args.no_terminal_print:
        print("\n=== START CIRCUIT ===")
        print_safe_terminal(start_txt.read_text(encoding="utf-8"))
        print("\n=== END CIRCUIT ===")
        print_safe_terminal(end_txt.read_text(encoding="utf-8"))
        print("\n=== STEP TABLE ===")
        print_safe_terminal(step_table)

    print(f"Saved: {start_txt}")
    print(f"Saved: {end_txt}")
    print(f"Saved: {table_txt}")
    print(f"Saved: {traj_png}")
    if start_ok and end_ok:
        print(f"Saved: {start_png}")
        print(f"Saved: {end_png}")
    else:
        print("Note: PNG circuit rendering skipped (text diagrams were saved).")


if __name__ == "__main__":
    main()
