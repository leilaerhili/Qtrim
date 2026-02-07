"""
Train RL policy and save to core/policy_store/.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from statistics import mean, pstdev
import subprocess
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch as th

from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.base_class import BaseAlgorithm

from core.accelerator import format_device_resolution, resolve_training_device
from core.circuits_baseline import (
    BASELINE_BUILDERS,
    get_builder,
    get_mixed_training_weights,
    make_seeded_challenge_builder,
)
from core.env_quantum_opt import EnvConfig, QuantumOptEnv


POLICY_STORE = Path(__file__).resolve().parent / "policy_store"

_ORT_PROVIDER_MAP = {
    "qnn": "QNNExecutionProvider",
    "directml": "DmlExecutionProvider",
    "dml": "DmlExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "cpu": "CPUExecutionProvider",
}


@dataclass(frozen=True)
class OnnxProviderResolution:
    requested: str
    resolved: str
    provider_name: str
    reason: str
    available_providers: Tuple[str, ...]
    strict: bool
    fallback_used: bool
    enabled: bool
    import_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requested": self.requested,
            "resolved": self.resolved,
            "provider_name": self.provider_name,
            "reason": self.reason,
            "available_providers": list(self.available_providers),
            "strict": bool(self.strict),
            "fallback_used": bool(self.fallback_used),
            "enabled": bool(self.enabled),
            "import_error": self.import_error,
        }


@dataclass(frozen=True)
class NexaProbe:
    enabled: bool
    reason: str
    python_package_version: Optional[str] = None
    sdk_version: Optional[str] = None
    plugins: Tuple[str, ...] = tuple()
    devices: Optional[Dict[str, List[str]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "reason": self.reason,
            "python_package_version": self.python_package_version,
            "sdk_version": self.sdk_version,
            "plugins": list(self.plugins),
            "devices": dict(self.devices or {}),
        }


def format_onnx_provider_resolution(resolution: OnnxProviderResolution) -> str:
    providers = ", ".join(resolution.available_providers) if resolution.available_providers else "none"
    return (
        "[onnx] "
        f"requested={resolution.requested} "
        f"resolved={resolution.resolved} "
        f"provider={resolution.provider_name or 'none'} "
        f"enabled={int(resolution.enabled)} "
        f"strict={int(resolution.strict)} "
        f"fallback={int(resolution.fallback_used)} "
        f"| {resolution.reason} "
        f"| available_providers: {providers}"
    )


def format_nexa_probe(probe: NexaProbe) -> str:
    if not probe.enabled:
        return f"[nexa] enabled=0 | {probe.reason}"
    plugin_text = ",".join(probe.plugins) if probe.plugins else "none"
    return (
        "[nexa] "
        f"enabled=1 "
        f"py={probe.python_package_version or 'unknown'} "
        f"sdk={probe.sdk_version or 'unknown'} "
        f"plugins={plugin_text} "
        f"| {probe.reason}"
    )


def probe_nexa_sdk(enabled: bool) -> NexaProbe:
    if not enabled:
        return NexaProbe(enabled=False, reason="Nexa SDK probe disabled by CLI.")
    try:
        import nexaai  # type: ignore
    except Exception as exc:
        return NexaProbe(
            enabled=False,
            reason=f"nexaai import failed: {exc}",
        )

    plugins: Tuple[str, ...] = tuple()
    devices: Dict[str, List[str]] = {}
    reason = "Nexa SDK initialized."
    try:
        raw_plugins = nexaai.get_plugin_list()
        plugins = tuple(str(x) for x in raw_plugins)
        for plugin_id in plugins:
            try:
                _, names = nexaai.get_device_list(plugin_id)
                devices[plugin_id] = [str(n) for n in names]
            except Exception as device_exc:
                devices[plugin_id] = [f"<device query failed: {device_exc}>"]
        if plugins:
            reason = "Nexa SDK initialized and plugins were enumerated."
        else:
            reason = "Nexa SDK initialized, but no plugins were reported."
    except Exception as exc:
        reason = f"Nexa SDK initialized, but plugin enumeration failed: {exc}"

    py_ver = None
    sdk_ver = None
    try:
        py_ver = str(nexaai.version())
    except Exception:
        pass
    try:
        sdk_ver = str(nexaai.nexa_version())
    except Exception:
        pass
    return NexaProbe(
        enabled=True,
        reason=reason,
        python_package_version=py_ver,
        sdk_version=sdk_ver,
        plugins=plugins,
        devices=devices,
    )


def _resolve_onnx_provider(requested: str = "auto", strict: bool = False) -> OnnxProviderResolution:
    normalized = str(requested).strip().lower().replace("_", "-")
    valid = {"auto", "qnn", "directml", "dml", "cuda", "cpu"}
    if normalized not in valid:
        choices = ", ".join(sorted(valid))
        raise ValueError(f"Unknown ONNX provider request '{requested}'. Valid options: {choices}")

    try:
        import onnxruntime as ort  # type: ignore
    except Exception as exc:
        if strict:
            raise RuntimeError(
                "onnxruntime could not be imported; cannot use ONNX/QNN inference backend."
            ) from exc
        return OnnxProviderResolution(
            requested=normalized,
            resolved="disabled",
            provider_name="",
            reason="onnxruntime import failed; using torch inference path.",
            available_providers=tuple(),
            strict=bool(strict),
            fallback_used=False,
            enabled=False,
            import_error=str(exc),
        )

    available = tuple(str(p) for p in ort.get_available_providers())
    fallback_used = False
    selected_key: Optional[str] = None
    if normalized == "auto":
        for key in ("qnn", "directml", "cuda", "cpu"):
            provider_name = _ORT_PROVIDER_MAP[key]
            if provider_name in available:
                selected_key = key
                break
    else:
        key = "directml" if normalized == "dml" else normalized
        provider_name = _ORT_PROVIDER_MAP[key]
        if provider_name in available:
            selected_key = key

    if selected_key is None:
        if strict:
            raise RuntimeError(
                f"Requested ONNX provider '{normalized}' is not available. "
                f"Detected providers: {available}"
            )
        for key in ("qnn", "directml", "cuda", "cpu"):
            provider_name = _ORT_PROVIDER_MAP[key]
            if provider_name in available:
                selected_key = key
                fallback_used = True
                break

    if selected_key is None:
        return OnnxProviderResolution(
            requested=normalized,
            resolved="disabled",
            provider_name="",
            reason="No usable ONNX Runtime execution provider was found; using torch inference path.",
            available_providers=available,
            strict=bool(strict),
            fallback_used=True,
            enabled=False,
            import_error=None,
        )

    resolved = selected_key
    provider_name = _ORT_PROVIDER_MAP[selected_key]
    if normalized == "auto":
        reason = (
            "Auto-selected ONNX Runtime provider with priority order "
            "qnn > directml > cuda > cpu."
        )
    elif fallback_used and normalized != resolved:
        reason = f"Requested provider '{normalized}' was unavailable; fell back to '{resolved}'."
    else:
        reason = f"Requested provider '{normalized}' is available."

    return OnnxProviderResolution(
        requested=normalized,
        resolved=resolved,
        provider_name=provider_name,
        reason=reason,
        available_providers=available,
        strict=bool(strict),
        fallback_used=bool(fallback_used),
        enabled=True,
        import_error=None,
    )


class OnnxQValueSession:
    def __init__(
        self,
        model_path: Path,
        resolution: OnnxProviderResolution,
        qnn_backend_path: Optional[str] = None,
    ) -> None:
        import onnxruntime as ort  # type: ignore

        if not resolution.enabled or not resolution.provider_name:
            raise RuntimeError("Cannot create ONNX session for a disabled provider resolution.")
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        providers: List[str] = [resolution.provider_name]
        provider_options: List[Dict[str, str]] = [{}]
        if resolution.provider_name == "QNNExecutionProvider":
            qnn_options: Dict[str, str] = {}
            if qnn_backend_path:
                qnn_options["backend_path"] = str(qnn_backend_path)
            provider_options = [qnn_options]
        if resolution.provider_name != "CPUExecutionProvider":
            providers.append("CPUExecutionProvider")
            provider_options.append({})

        self._session = ort.InferenceSession(
            str(model_path),
            providers=providers,
            provider_options=provider_options,
        )
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if not inputs or not outputs:
            raise RuntimeError("ONNX session has no inputs or outputs.")
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        self.active_providers = tuple(str(p) for p in self._session.get_providers())

    def predict_actions(self, observation_batch: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation_batch, dtype=np.float32)
        q_values = self._session.run([self.output_name], {self.input_name: obs})[0]
        if q_values.ndim == 1:
            q_values = q_values.reshape(1, -1)
        return np.asarray(np.argmax(q_values, axis=1), dtype=np.int64)


class QnnInferenceDQN(DQN):
    def __init__(
        self,
        *args,
        onnx_provider_request: str = "auto",
        onnx_provider_strict: bool = False,
        onnx_sync_interval: int = 2000,
        onnx_export_path: Optional[Path] = None,
        onnx_qnn_backend_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        self._onnx_provider_request = str(onnx_provider_request)
        self._onnx_provider_strict = bool(onnx_provider_strict)
        self._onnx_sync_interval = max(1, int(onnx_sync_interval))
        self._onnx_export_path = (
            Path(onnx_export_path)
            if onnx_export_path is not None
            else POLICY_STORE / "dqn_live_qnet.onnx"
        )
        self._onnx_qnn_backend_path = (
            str(onnx_qnn_backend_path) if onnx_qnn_backend_path else None
        )
        self._onnx_runner: Optional[OnnxQValueSession] = None
        self._onnx_resolution: Optional[OnnxProviderResolution] = None
        self._onnx_last_sync_step = -1
        self._onnx_sync_count = 0
        self._onnx_fail_count = 0
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        self._onnx_resolution = _resolve_onnx_provider(
            requested=self._onnx_provider_request,
            strict=self._onnx_provider_strict,
        )
        print(format_onnx_provider_resolution(self._onnx_resolution), flush=True)
        if not self._onnx_resolution.enabled:
            return
        self._sync_onnx_policy(force=True)

    def _predict_actions_with_onnx(self, observation: np.ndarray) -> np.ndarray:
        if self._onnx_runner is None:
            raise RuntimeError("ONNX inference session is not initialized.")
        obs_array = np.asarray(observation, dtype=np.float32)
        vectorized = self.policy.is_vectorized_observation(obs_array)
        if vectorized:
            batch = obs_array.reshape((-1, *self.observation_space.shape)).astype(np.float32, copy=False)
        else:
            batch = obs_array.reshape((1, *self.observation_space.shape)).astype(np.float32, copy=False)
        actions = self._onnx_runner.predict_actions(batch)
        if vectorized:
            return actions
        return np.asarray(actions[0])

    def _export_q_net_onnx(self) -> None:
        if not hasattr(self.observation_space, "shape") or self.observation_space.shape is None:
            raise RuntimeError("Only Box observation spaces are supported for ONNX inference DQN.")
        _export_q_network_onnx(
            q_net=self.q_net,
            observation_shape=tuple(int(x) for x in self.observation_space.shape),
            output_path=self._onnx_export_path,
            dynamic_batch=True,
            batch_size=1,
        )

    def _sync_onnx_policy(self, force: bool) -> None:
        if self._onnx_resolution is None or not self._onnx_resolution.enabled:
            return
        if not force and self.num_timesteps < self.learning_starts:
            return
        if not force and self._onnx_last_sync_step >= 0:
            if int(self.num_timesteps) - int(self._onnx_last_sync_step) < self._onnx_sync_interval:
                return
        try:
            self._export_q_net_onnx()
            self._onnx_runner = OnnxQValueSession(
                model_path=self._onnx_export_path,
                resolution=self._onnx_resolution,
                qnn_backend_path=self._onnx_qnn_backend_path,
            )
            self._onnx_last_sync_step = int(self.num_timesteps)
            self._onnx_sync_count += 1
            providers = ",".join(self._onnx_runner.active_providers)
            print(
                "[onnx] sync "
                f"step={int(self.num_timesteps)} "
                f"count={int(self._onnx_sync_count)} "
                f"providers={providers}",
                flush=True,
            )
        except Exception as exc:
            self._onnx_runner = None
            self._onnx_fail_count += 1
            if self._onnx_provider_strict:
                raise RuntimeError("Failed to sync ONNX Runtime inference path.") from exc
            if self._onnx_fail_count <= 3:
                print(
                    "[onnx] disabled "
                    f"reason=sync_failed error={exc} "
                    "falling back to torch policy inference.",
                    flush=True,
                )

    def _on_step(self) -> None:
        super()._on_step()
        self._sync_onnx_policy(force=False)

    def _excluded_save_params(self) -> List[str]:
        excluded = list(super()._excluded_save_params())
        excluded.append("_onnx_runner")
        return excluded

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        if not deterministic and np.random.rand() < self.exploration_rate:
            if self.policy.is_vectorized_observation(observation):
                if isinstance(observation, dict):
                    n_batch = observation[next(iter(observation.keys()))].shape[0]
                else:
                    n_batch = observation.shape[0]
                action = np.array([self.action_space.sample() for _ in range(n_batch)])
            else:
                action = np.array(self.action_space.sample())
            return action, state

        if isinstance(observation, dict):
            return self.policy.predict(observation, state, episode_start, deterministic)
        if self._onnx_runner is not None:
            try:
                action = self._predict_actions_with_onnx(observation)
                return action, state
            except Exception as exc:
                self._onnx_fail_count += 1
                if self._onnx_provider_strict:
                    raise RuntimeError("ONNX Runtime predict path failed.") from exc
                if self._onnx_fail_count <= 3:
                    print(
                        "[onnx] predict_fallback "
                        f"reason=runtime_error error={exc} "
                        "using torch inference for this step.",
                        flush=True,
                    )

        return self.policy.predict(observation, state, episode_start, deterministic)

    def get_onnx_runtime_info(self) -> Dict[str, Any]:
        resolution = self._onnx_resolution.to_dict() if self._onnx_resolution else None
        providers = (
            list(self._onnx_runner.active_providers)
            if self._onnx_runner is not None
            else []
        )
        return {
            "resolution": resolution,
            "export_path": str(self._onnx_export_path),
            "sync_interval": int(self._onnx_sync_interval),
            "sync_count": int(self._onnx_sync_count),
            "fail_count": int(self._onnx_fail_count),
            "active_providers": providers,
        }


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

    all_names = sorted(BASELINE_BUILDERS.keys())
    weights = get_mixed_training_weights()
    weighted_names = [(name, float(weights.get(name, 1.0))) for name in all_names]
    weighted_names = [(name, w) for name, w in weighted_names if w > 0.0]
    if not weighted_names and all_names:
        weighted_names = [(name, 1.0) for name in all_names]
    names = [name for name, _ in weighted_names]
    probs = np.array([w for _, w in weighted_names], dtype=np.float64)
    if not names:
        raise RuntimeError("No baseline builders are registered.")
    probs = probs / float(np.sum(probs))
    lo = int(min(pad_level_min, pad_level_max))
    hi = int(max(pad_level_min, pad_level_max))
    rng = np.random.default_rng(seed)

    def _builder(_: int) -> QuantumCircuit:
        name = names[int(rng.choice(len(names), p=probs))]
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
    use_applicability_mask: bool = False,
    priority_config: Optional[Dict[str, Any]] = None,
) -> QuantumOptEnv:
    cfg_extra = dict(priority_config or {})
    config = EnvConfig(
        constraint_profile=constraint_profile,
        use_applicability_mask=bool(use_applicability_mask),
        **cfg_extra,
    )
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
    use_applicability_mask: bool = False,
    priority_config: Optional[Dict[str, Any]] = None,
) -> QuantumOptEnv:
    return _make_env_with_builder(
        circuit_builder=get_builder(baseline),
        pad_level=pad_level,
        constraint_profile=constraint_profile,
        seed=seed,
        monitor=monitor,
        use_applicability_mask=use_applicability_mask,
        priority_config=priority_config,
    )


def _make_holdout_env(
    seed: int,
    pad_level: int,
    constraint_profile: str,
    monitor: bool = False,
    use_applicability_mask: bool = False,
    priority_config: Optional[Dict[str, Any]] = None,
) -> QuantumOptEnv:
    builder = make_seeded_challenge_builder(seed=seed, num_qubits=6, depth=36)
    return _make_env_with_builder(
        circuit_builder=builder,
        pad_level=pad_level,
        constraint_profile=constraint_profile,
        seed=seed,
        monitor=monitor,
        use_applicability_mask=use_applicability_mask,
        priority_config=priority_config,
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
    use_applicability_mask: bool,
    priority_config: Optional[Dict[str, Any]],
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
                    use_applicability_mask=use_applicability_mask,
                    priority_config=priority_config,
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
                    use_applicability_mask=use_applicability_mask,
                    priority_config=priority_config,
                )
            )
    return DummyVecEnv(env_fns)


def _evaluate_training_curve_point(
    model: BaseAlgorithm,
    *,
    baseline: str,
    pad_level: int,
    constraint_profile: str,
    train_mode: str,
    seed: Optional[int],
    n_eval_episodes: int,
    mixed_pad_level_min: int,
    mixed_pad_level_max: int,
    priority_config: Optional[Dict[str, Any]],
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
                    use_applicability_mask=False,
                    priority_config=priority_config,
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
        priority_config=priority_config,
    )


def _effective_batch_size(batch_size: int, n_steps: int, n_envs: int) -> int:
    rollout = max(1, int(n_steps) * max(1, int(n_envs)))
    target = max(1, int(batch_size))
    if target > rollout:
        return rollout
    if rollout % target == 0:
        return target
    # Pick the largest divisor <= target to avoid truncated minibatches.
    for d in range(target, 0, -1):
        if rollout % d == 0:
            return d
    return 1


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


def _coerce_priority_weights(weights: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    if not isinstance(weights, dict):
        return None
    out: Dict[str, float] = {}
    for k, v in weights.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            continue
    return out if out else None


def _build_priority_config(
    *,
    priority_profile_id: str,
    priority_weights: Optional[Dict[str, Any]],
    max_depth_budget: Optional[int],
    max_latency_ms: Optional[float],
    max_shots: Optional[int],
    queue_level: str,
    noise_level: str,
    backend_condition: str,
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "priority_profile_id": str(priority_profile_id),
        "context_queue_level": str(queue_level),
        "context_noise_level": str(noise_level),
        "context_backend": str(backend_condition),
    }
    weights = _coerce_priority_weights(priority_weights)
    if weights is not None:
        cfg["priority_weights"] = weights
    if max_depth_budget is not None:
        cfg["max_depth_budget"] = int(max_depth_budget)
    if max_latency_ms is not None:
        cfg["max_latency_ms"] = float(max_latency_ms)
    if max_shots is not None:
        cfg["max_shots"] = int(max_shots)
    return cfg


def _export_q_network_onnx(
    *,
    q_net: th.nn.Module,
    observation_shape: Tuple[int, ...],
    output_path: Path,
    dynamic_batch: bool,
    batch_size: int = 1,
) -> None:
    from copy import deepcopy

    output_path.parent.mkdir(parents=True, exist_ok=True)
    q_net_cpu = deepcopy(q_net).cpu().eval()
    effective_batch = max(1, int(batch_size))
    dummy_obs = th.zeros((effective_batch, *observation_shape), dtype=th.float32)
    export_kwargs: Dict[str, Any] = {
        "input_names": ["obs"],
        "output_names": ["q_values"],
        "opset_version": 17,
    }
    if dynamic_batch:
        export_kwargs["dynamic_axes"] = {
            "obs": {0: "batch"},
            "q_values": {0: "batch"},
        }
    try:
        import inspect

        sig = inspect.signature(th.onnx.export)
        if "dynamo" in sig.parameters:
            export_kwargs["dynamo"] = False
    except Exception:
        pass

    with th.no_grad():
        th.onnx.export(
            q_net_cpu,
            dummy_obs,
            str(output_path),
            **export_kwargs,
        )


def _validate_onnx_model_cpu(model_path: Path, batch_size: int, obs_dim: int) -> Dict[str, Any]:
    result: Dict[str, Any] = {"ok": False, "error": None}
    try:
        import onnx
        import onnxruntime as ort  # type: ignore

        onnx_model = onnx.load(str(model_path))
        onnx.checker.check_model(onnx_model)
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        x = np.zeros((max(1, int(batch_size)), int(obs_dim)), dtype=np.float32)
        _ = session.run(None, {session.get_inputs()[0].name: x})[0]
        result["ok"] = True
        return result
    except Exception as exc:
        result["error"] = str(exc)
        return result


def _validate_onnx_model_qnn_strict(
    model_path: Path,
    batch_size: int,
    obs_dim: int,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "ok": False,
        "error": None,
        "providers": [],
    }
    validator_code = (
        "import json, sys\n"
        "import numpy as np\n"
        "import onnxruntime as ort\n"
        "out = {'ok': False, 'error': None, 'providers': []}\n"
        "try:\n"
        "    model_path = sys.argv[1]\n"
        "    batch = max(1, int(sys.argv[2]))\n"
        "    obs_dim = int(sys.argv[3])\n"
        "    so = ort.SessionOptions()\n"
        "    so.add_session_config_entry('session.disable_cpu_ep_fallback', '1')\n"
        "    session = ort.InferenceSession(model_path, sess_options=so, providers=['QNNExecutionProvider'])\n"
        "    x = np.zeros((batch, obs_dim), dtype=np.float32)\n"
        "    _ = session.run(None, {session.get_inputs()[0].name: x})[0]\n"
        "    out['ok'] = True\n"
        "    out['providers'] = [str(p) for p in session.get_providers()]\n"
        "except Exception as exc:\n"
        "    out['error'] = str(exc)\n"
        "print(json.dumps(out))\n"
    )
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                validator_code,
                str(model_path),
                str(max(1, int(batch_size))),
                str(int(obs_dim)),
            ],
            capture_output=True,
            timeout=180,
            check=False,
        )
    except Exception as exc:
        result["error"] = f"Failed to launch strict QNN validator subprocess: {exc}"
        return result

    stdout_text = proc.stdout.decode("utf-8", errors="replace").replace("\x00", "").strip()
    stderr_text = proc.stderr.decode("utf-8", errors="replace").replace("\x00", "").strip()

    parsed: Optional[Dict[str, Any]] = None
    for line in reversed([ln.strip() for ln in stdout_text.splitlines() if ln.strip()]):
        if line.startswith("{") and line.endswith("}"):
            try:
                maybe = json.loads(line)
                if isinstance(maybe, dict):
                    parsed = maybe
                    break
            except Exception:
                pass

    if proc.returncode != 0:
        result["error"] = f"Strict QNN validator crashed (exit={proc.returncode})."
        if stderr_text:
            result["stderr_tail"] = stderr_text[-800:]
        if stdout_text:
            result["stdout_tail"] = stdout_text[-800:]
        if parsed is not None:
            result.update(parsed)
        return result

    if parsed is not None:
        result.update(parsed)
    else:
        result["error"] = "Strict QNN validator produced no JSON output."
        if stdout_text:
            result["stdout_tail"] = stdout_text[-800:]
        if stderr_text:
            result["stderr_tail"] = stderr_text[-800:]
    return result


def export_android_onnx_artifacts(
    *,
    model: BaseAlgorithm,
    run_name: str,
    static_batch_size: int,
    export_int8: bool,
    strict_qnn_check: bool,
) -> Dict[str, Any]:
    details: Dict[str, Any] = {
        "enabled": True,
        "run_name": str(run_name),
        "fp32_path": None,
        "int8_path": None,
        "cpu_validation": None,
        "qnn_validation_fp32": None,
        "qnn_validation_int8": None,
        "notes": [],
    }
    if not hasattr(model, "q_net") or not hasattr(model, "observation_space"):
        details["enabled"] = False
        details["notes"].append("Model has no q_net; Android ONNX export currently supports DQN only.")
        return details

    obs_shape = getattr(model.observation_space, "shape", None)
    if obs_shape is None or len(obs_shape) != 1:
        details["enabled"] = False
        details["notes"].append("Only 1D Box observation spaces are currently supported.")
        return details
    obs_dim = int(obs_shape[0])
    fp32_path = POLICY_STORE / f"{run_name}_android_fp32_bs{max(1, int(static_batch_size))}.onnx"

    try:
        _export_q_network_onnx(
            q_net=model.q_net,  # type: ignore[attr-defined]
            observation_shape=tuple(int(x) for x in obs_shape),
            output_path=fp32_path,
            dynamic_batch=False,
            batch_size=max(1, int(static_batch_size)),
        )
        details["fp32_path"] = str(fp32_path)
        print(
            "[android] export_fp32 "
            f"path={fp32_path} static_batch={max(1, int(static_batch_size))}",
            flush=True,
        )
    except Exception as exc:
        details["enabled"] = False
        details["notes"].append(f"FP32 export failed: {exc}")
        print(f"[android] export_failed error={exc}", flush=True)
        return details

    details["cpu_validation"] = _validate_onnx_model_cpu(
        fp32_path,
        batch_size=max(1, int(static_batch_size)),
        obs_dim=obs_dim,
    )
    if strict_qnn_check:
        details["qnn_validation_fp32"] = _validate_onnx_model_qnn_strict(
            fp32_path,
            batch_size=max(1, int(static_batch_size)),
            obs_dim=obs_dim,
        )

    if export_int8:
        int8_path = POLICY_STORE / f"{run_name}_android_int8_bs{max(1, int(static_batch_size))}.onnx"
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic  # type: ignore

            quantize_dynamic(
                model_input=str(fp32_path),
                model_output=str(int8_path),
                weight_type=QuantType.QInt8,
            )
            details["int8_path"] = str(int8_path)
            print(f"[android] export_int8 path={int8_path}", flush=True)
            if strict_qnn_check:
                details["qnn_validation_int8"] = _validate_onnx_model_qnn_strict(
                    int8_path,
                    batch_size=max(1, int(static_batch_size)),
                    obs_dim=obs_dim,
                )
        except Exception as exc:
            details["notes"].append(f"INT8 export failed: {exc}")
            print(f"[android] export_int8_failed error={exc}", flush=True)
    return details


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
    algo: str = "dqn",
    device: str = "directml",
    strict_device: bool = False,
    inference_backend: str = "ort-qnn",
    strict_inference_backend: bool = False,
    onnx_sync_interval: int = 2000,
    onnx_qnn_backend_path: Optional[str] = None,
    onnx_export_path: Optional[Path] = None,
    use_nexa_sdk: bool = True,
    priority_profile_id: str = "auto",
    priority_weights: Optional[Dict[str, Any]] = None,
    max_depth_budget: Optional[int] = None,
    max_latency_ms: Optional[float] = None,
    max_shots: Optional[int] = None,
    queue_level: str = "normal",
    noise_level: str = "normal",
    backend_condition: str = "unknown",
    train_mode: str = "fixed",
    mixed_pad_level_min: int = 1,
    mixed_pad_level_max: int = 3,
    eval_every: int = 0,
    eval_episodes_curve: int = 5,
    curve_json_path: Optional[Path] = None,
    curve_png_path: Optional[Path] = None,
    device_resolution_out: Optional[Dict[str, Any]] = None,
    inference_resolution_out: Optional[Dict[str, Any]] = None,
    nexa_probe_out: Optional[Dict[str, Any]] = None,
    use_applicability_mask: bool = False,
) -> BaseAlgorithm:
    resolved_device, device_resolution = resolve_training_device(
        requested=device,
        strict=bool(strict_device),
    )
    print(format_device_resolution(device_resolution))
    if device_resolution_out is not None:
        device_resolution_out.clear()
        device_resolution_out.update(device_resolution.to_dict())
    nexa_probe = probe_nexa_sdk(enabled=bool(use_nexa_sdk))
    print(format_nexa_probe(nexa_probe), flush=True)
    if nexa_probe_out is not None:
        nexa_probe_out.clear()
        nexa_probe_out.update(nexa_probe.to_dict())
    resolved_priority_profile = (
        constraint_profile
        if str(priority_profile_id).strip().lower() == "auto"
        else str(priority_profile_id)
    )
    priority_config = _build_priority_config(
        priority_profile_id=resolved_priority_profile,
        priority_weights=priority_weights,
        max_depth_budget=max_depth_budget,
        max_latency_ms=max_latency_ms,
        max_shots=max_shots,
        queue_level=queue_level,
        noise_level=noise_level,
        backend_condition=backend_condition,
    )
    print(
        "[priority] "
        f"profile={priority_config.get('priority_profile_id')} "
        f"weights_override={'yes' if 'priority_weights' in priority_config else 'no'} "
        f"budgets(depth={priority_config.get('max_depth_budget')},"
        f"latency_ms={priority_config.get('max_latency_ms')},"
        f"shots={priority_config.get('max_shots')}) "
        f"context(queue={priority_config.get('context_queue_level')},"
        f"noise={priority_config.get('context_noise_level')},"
        f"backend={priority_config.get('context_backend')})",
        flush=True,
    )

    env = _make_train_vec_env(
        baseline=baseline,
        pad_level=pad_level,
        constraint_profile=constraint_profile,
        seed=seed,
        train_mode=train_mode,
        n_envs=n_envs,
        mixed_pad_level_min=mixed_pad_level_min,
        mixed_pad_level_max=mixed_pad_level_max,
        use_applicability_mask=use_applicability_mask,
        priority_config=priority_config,
    )
    effective_batch = _effective_batch_size(
        batch_size=int(batch_size),
        n_steps=int(n_steps),
        n_envs=int(n_envs),
    )
    if effective_batch != int(batch_size):
        print(
            "[train] adjusted_batch_size "
            f"requested={int(batch_size)} effective={effective_batch} "
            f"(n_steps*n_envs={int(n_steps)*max(1, int(n_envs))})",
            flush=True,
    )
    algo_name = str(algo).strip().lower()
    inference_choice = str(inference_backend).strip().lower().replace("_", "-")
    if algo_name == "ppo":
        if inference_choice != "torch":
            print(
                "[onnx] note algo=ppo uses torch inference path; "
                "ONNX/QNN inference backend is only implemented for DQN.",
                flush=True,
            )
        model: BaseAlgorithm = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            seed=seed,
            ent_coef=float(ent_coef),
            learning_rate=float(learning_rate),
            n_steps=int(n_steps),
            batch_size=effective_batch,
            device=resolved_device,
        )
        if inference_resolution_out is not None:
            inference_resolution_out.clear()
            inference_resolution_out.update(
                {
                    "backend": "torch",
                    "reason": "PPO path currently uses torch policy inference.",
                }
            )
    elif algo_name == "dqn":
        dqn_kwargs = dict(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            seed=seed,
            learning_rate=float(learning_rate),
            batch_size=effective_batch,
            buffer_size=max(20_000, 100 * effective_batch),
            learning_starts=max(200, 4 * effective_batch),
            train_freq=4,
            gradient_steps=1,
            target_update_interval=500,
            exploration_fraction=0.25,
            device=resolved_device,
        )
        ort_provider_map = {
            "ort-qnn": "qnn",
            "ort-auto": "auto",
            "ort-cpu": "cpu",
            "ort-dml": "directml",
            "ort-directml": "directml",
            "ort-cuda": "cuda",
        }
        if inference_choice == "torch":
            model = DQN(**dqn_kwargs)
            if inference_resolution_out is not None:
                inference_resolution_out.clear()
                inference_resolution_out.update(
                    {
                        "backend": "torch",
                        "reason": "Requested torch inference backend.",
                    }
                )
        elif inference_choice in ort_provider_map:
            model = QnnInferenceDQN(
                **dqn_kwargs,
                onnx_provider_request=ort_provider_map[inference_choice],
                onnx_provider_strict=bool(strict_inference_backend),
                onnx_sync_interval=int(onnx_sync_interval),
                onnx_export_path=onnx_export_path,
                onnx_qnn_backend_path=onnx_qnn_backend_path,
            )
            if inference_resolution_out is not None:
                inference_resolution_out.clear()
                inference_resolution_out.update(model.get_onnx_runtime_info())
        else:
            raise ValueError(
                "Unsupported inference backend "
                f"'{inference_backend}'. Choose from: "
                "torch, ort-qnn, ort-auto, ort-dml, ort-cuda, ort-cpu."
            )
    else:
        raise ValueError(f"Unsupported algo '{algo}'. Choose from: ppo, dqn.")
    print(
        "[train] start "
        f"timesteps={int(total_timesteps)} "
        f"eval_every={int(eval_every)} "
        f"train_mode={train_mode} "
        f"n_envs={int(n_envs)}",
        flush=True,
    )
    if int(eval_every) > 0:
        history = {"timesteps": [], "mean_reward": [], "std_reward": []}
        steps_done = 0
        while steps_done < int(total_timesteps):
            chunk = min(int(eval_every), int(total_timesteps) - steps_done)
            print(
                "[train] chunk_begin "
                f"done={steps_done}/{int(total_timesteps)} "
                f"chunk={chunk}",
                flush=True,
            )
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
                priority_config=priority_config,
            )
            history["timesteps"].append(int(steps_done))
            history["mean_reward"].append(float(mean_r))
            history["std_reward"].append(float(std_r))
            print(
                "[train] chunk_end "
                f"done={steps_done}/{int(total_timesteps)} "
                f"eval_mean={float(mean_r):.3f} "
                f"eval_std={float(std_r):.3f}",
                flush=True,
            )
        if curve_json_path is not None:
            _save_training_curve(
                history,
                curve_json_path=curve_json_path,
                curve_png_path=curve_png_path,
            )
    else:
        print("[train] single_learn_begin", flush=True)
        model.learn(total_timesteps=total_timesteps)
        print("[train] single_learn_end", flush=True)
    env.close()
    if inference_resolution_out is not None and isinstance(model, QnnInferenceDQN):
        inference_resolution_out.clear()
        inference_resolution_out.update(model.get_onnx_runtime_info())
    print("[train] complete", flush=True)
    return model


def save_policy(model: BaseAlgorithm, name: str) -> Path:
    POLICY_STORE.mkdir(parents=True, exist_ok=True)
    path = POLICY_STORE / f"{name}.zip"
    model.save(path)
    return path


def load_policy(
    name: str,
    baseline: str = "toy",
    pad_level: int = 2,
    constraint_profile: str = "balanced",
) -> BaseAlgorithm:
    env = DummyVecEnv(
        [lambda: _make_env(baseline, pad_level, constraint_profile, seed=0, monitor=True)]
    )
    path = POLICY_STORE / f"{name}.zip"
    try:
        return PPO.load(path, env=env)
    except Exception:
        return DQN.load(path, env=env)


def get_action(model: BaseAlgorithm, obs: np.ndarray, deterministic: bool = True) -> int:
    action, _ = model.predict(obs, deterministic=deterministic)
    return int(action)


def evaluate(
    model: BaseAlgorithm,
    baseline: str,
    pad_level: int,
    constraint_profile: str,
    n_eval_episodes: int = 5,
    priority_config: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float]:
    env = DummyVecEnv(
        [
            lambda: _make_env(
                baseline,
                pad_level,
                constraint_profile,
                seed=123,
                monitor=True,
                priority_config=priority_config,
            )
        ]
    )
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
    )
    return float(mean_reward), float(std_reward)


def evaluate_holdout(
    model: BaseAlgorithm,
    holdout_seed_start: int = 10_000,
    holdout_count: int = 8,
    pad_level: int = 3,
    constraint_profile: str = "balanced",
    n_eval_episodes: int = 2,
    priority_config: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float]:
    scores: List[float] = []
    for s in range(int(holdout_seed_start), int(holdout_seed_start) + int(holdout_count)):
        env = DummyVecEnv(
            [
                lambda seed=s: _make_holdout_env(
                    seed,
                    pad_level,
                    constraint_profile,
                    monitor=True,
                    priority_config=priority_config,
                )
            ]
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


def parse_optional_json_object(raw: Optional[str], field_name: str) -> Optional[Dict[str, Any]]:
    if raw is None:
        return None
    txt = str(raw).strip()
    if not txt:
        return None
    try:
        parsed = json.loads(txt)
    except Exception as exc:
        raise ValueError(f"{field_name} must be valid JSON object text.") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{field_name} must decode to a JSON object.")
    return dict(parsed)


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
    algo: str,
    device: str,
    strict_device: bool,
    inference_backend: str,
    strict_inference_backend: bool,
    onnx_sync_interval: int,
    onnx_qnn_backend_path: Optional[str],
    use_nexa_sdk: bool,
    export_android_onnx: bool,
    android_onnx_static_batch: int,
    android_onnx_int8: bool,
    android_qnn_strict_check: bool,
    priority_profile_id: str,
    priority_weights: Optional[Dict[str, Any]],
    max_depth_budget: Optional[int],
    max_latency_ms: Optional[float],
    max_shots: Optional[int],
    queue_level: str,
    noise_level: str,
    backend_condition: str,
    train_mode: str,
    mixed_pad_level_min: int,
    mixed_pad_level_max: int,
    eval_every: int,
    eval_episodes_curve: int,
    use_applicability_mask: bool,
) -> Dict[str, object]:
    POLICY_STORE.mkdir(parents=True, exist_ok=True)
    runs: List[Dict[str, object]] = []
    resolved_priority_profile = (
        constraint_profile
        if str(priority_profile_id).strip().lower() == "auto"
        else str(priority_profile_id)
    )
    priority_config = _build_priority_config(
        priority_profile_id=resolved_priority_profile,
        priority_weights=priority_weights,
        max_depth_budget=max_depth_budget,
        max_latency_ms=max_latency_ms,
        max_shots=max_shots,
        queue_level=queue_level,
        noise_level=noise_level,
        backend_condition=backend_condition,
    )

    for seed in seeds:
        run_name = f"{save_name}_seed{seed}"
        print(f"[run] seed_begin seed={int(seed)} name={run_name}", flush=True)
        curve_json = POLICY_STORE / f"{run_name}_curve.json"
        curve_png = POLICY_STORE / f"{run_name}_curve.png"
        onnx_model_path = POLICY_STORE / f"{run_name}_qnet.onnx"
        device_resolution_data: Dict[str, Any] = {}
        inference_resolution_data: Dict[str, Any] = {}
        nexa_probe_data: Dict[str, Any] = {}
        android_onnx_data: Dict[str, Any] = {}
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
            algo=algo,
            device=device,
            strict_device=strict_device,
            inference_backend=inference_backend,
            strict_inference_backend=strict_inference_backend,
            onnx_sync_interval=onnx_sync_interval,
            onnx_qnn_backend_path=onnx_qnn_backend_path,
            onnx_export_path=onnx_model_path,
            use_nexa_sdk=use_nexa_sdk,
            priority_profile_id=resolved_priority_profile,
            priority_weights=priority_weights,
            max_depth_budget=max_depth_budget,
            max_latency_ms=max_latency_ms,
            max_shots=max_shots,
            queue_level=queue_level,
            noise_level=noise_level,
            backend_condition=backend_condition,
            train_mode=train_mode,
            mixed_pad_level_min=mixed_pad_level_min,
            mixed_pad_level_max=mixed_pad_level_max,
            eval_every=eval_every,
            eval_episodes_curve=eval_episodes_curve,
            curve_json_path=curve_json if eval_every > 0 else None,
            curve_png_path=curve_png if eval_every > 0 else None,
            device_resolution_out=device_resolution_data,
            inference_resolution_out=inference_resolution_data,
            nexa_probe_out=nexa_probe_data,
            use_applicability_mask=use_applicability_mask,
        )
        if bool(export_android_onnx):
            android_onnx_data = export_android_onnx_artifacts(
                model=model,
                run_name=run_name,
                static_batch_size=max(1, int(android_onnx_static_batch)),
                export_int8=bool(android_onnx_int8),
                strict_qnn_check=bool(android_qnn_strict_check),
            )
        else:
            android_onnx_data = {"enabled": False, "notes": ["Android ONNX export disabled by CLI."]}
        path = save_policy(model, run_name)
        eval_mean, eval_std = evaluate(
            model,
            baseline=baseline,
            pad_level=pad_level,
            constraint_profile=constraint_profile,
            n_eval_episodes=eval_episodes,
            priority_config=priority_config,
        )
        holdout_mean, holdout_std = evaluate_holdout(
            model,
            holdout_seed_start=holdout_seed_start,
            holdout_count=holdout_count,
            pad_level=max(3, pad_level),
            constraint_profile=constraint_profile,
            n_eval_episodes=max(1, eval_episodes // 2),
            priority_config=priority_config,
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
                "device_resolution": dict(device_resolution_data),
                "inference_resolution": dict(inference_resolution_data),
                "nexa_probe": dict(nexa_probe_data),
                "onnx_model_path": str(onnx_model_path),
                "priority_config": dict(priority_config),
                "android_onnx": dict(android_onnx_data),
            }
        )
        print(
            "[run] seed_end "
            f"seed={int(seed)} "
            f"eval_mean={float(eval_mean):.3f} "
            f"holdout_mean={float(holdout_mean):.3f}",
            flush=True,
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
            "algo": str(algo),
            "device": str(device),
            "strict_device": bool(strict_device),
            "inference_backend": str(inference_backend),
            "strict_inference_backend": bool(strict_inference_backend),
            "onnx_sync_interval": int(onnx_sync_interval),
            "onnx_qnn_backend_path": str(onnx_qnn_backend_path) if onnx_qnn_backend_path else None,
            "use_nexa_sdk": bool(use_nexa_sdk),
            "export_android_onnx": bool(export_android_onnx),
            "android_onnx_static_batch": int(android_onnx_static_batch),
            "android_onnx_int8": bool(android_onnx_int8),
            "android_qnn_strict_check": bool(android_qnn_strict_check),
            "priority_profile_id": str(resolved_priority_profile),
            "priority_weights": dict(priority_weights) if isinstance(priority_weights, dict) else None,
            "max_depth_budget": int(max_depth_budget) if max_depth_budget is not None else None,
            "max_latency_ms": float(max_latency_ms) if max_latency_ms is not None else None,
            "max_shots": int(max_shots) if max_shots is not None else None,
            "queue_level": str(queue_level),
            "noise_level": str(noise_level),
            "backend_condition": str(backend_condition),
            "train_mode": str(train_mode),
            "mixed_pad_level_min": int(mixed_pad_level_min),
            "mixed_pad_level_max": int(mixed_pad_level_max),
            "use_applicability_mask": bool(use_applicability_mask),
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
    parser = argparse.ArgumentParser(description="Train RL policy for QuantumOptEnv.")
    baseline_choices = sorted(BASELINE_BUILDERS.keys())
    default_baseline = "toy" if "toy" in BASELINE_BUILDERS else baseline_choices[0]
    parser.add_argument("--baseline", default=default_baseline, choices=baseline_choices)
    parser.add_argument("--pad-level", type=int, default=2)
    parser.add_argument("--constraint-profile", default="balanced")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-name", default="dqn_directml_quantum_opt")
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
    parser.add_argument("--algo", choices=["ppo", "dqn"], default="dqn")
    parser.add_argument(
        "--device",
        default="directml",
        help="Device preference: auto, npu, directml, cuda, xpu, mps, cpu.",
    )
    parser.add_argument(
        "--strict-device",
        action="store_true",
        help="Fail immediately if the requested device is unavailable.",
    )
    parser.add_argument(
        "--inference-backend",
        default="ort-qnn",
        choices=["torch", "ort-qnn", "ort-auto", "ort-dml", "ort-cuda", "ort-cpu"],
        help=(
            "Inference backend for action selection during rollout. "
            "Use ort-qnn to target ONNX Runtime QNN (NPU) when available."
        ),
    )
    parser.add_argument(
        "--strict-inference-backend",
        action="store_true",
        help="Fail if the requested ONNX Runtime provider is unavailable.",
    )
    parser.add_argument(
        "--onnx-sync-interval",
        type=int,
        default=2000,
        help="How often (in env steps) to re-export and reload DQN q_net into ONNX Runtime.",
    )
    parser.add_argument(
        "--onnx-qnn-backend-path",
        default=None,
        help=(
            "Optional full path to QNN backend library (for example QnnHtp.dll). "
            "Used only when inference backend resolves to QNNExecutionProvider."
        ),
    )
    parser.add_argument(
        "--use-nexa-sdk",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Nexa SDK probe/telemetry during training startup.",
    )
    parser.add_argument(
        "--export-android-onnx",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Export Android-friendly ONNX artifacts after each run "
            "(fixed-shape FP32 plus optional INT8 model)."
        ),
    )
    parser.add_argument(
        "--android-onnx-static-batch",
        type=int,
        default=1,
        help="Static batch size for Android ONNX export.",
    )
    parser.add_argument(
        "--android-onnx-int8",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also emit a dynamic-quantized INT8 ONNX variant for Android.",
    )
    parser.add_argument(
        "--android-qnn-strict-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run strict QNN-only validation on Android ONNX artifacts "
            "(disables CPU fallback in the validation session)."
        ),
    )
    parser.add_argument(
        "--priority-profile-id",
        default="auto",
        choices=["auto", "balanced", "high_fidelity", "low_latency", "low_cost"],
        help="Phone-selected objective preset for this run.",
    )
    parser.add_argument(
        "--priority-weights-json",
        default=None,
        help=(
            "Optional JSON object for profile weight overrides, for example "
            "'{\"two_qubit_gates\":0.5,\"depth\":0.3,\"total_gates\":0.1,\"swap_gates\":0.1}'."
        ),
    )
    parser.add_argument("--max-depth-budget", type=int, default=None)
    parser.add_argument("--max-latency-ms", type=float, default=None)
    parser.add_argument("--max-shots", type=int, default=None)
    parser.add_argument("--queue-level", default="normal", choices=["low", "normal", "high"])
    parser.add_argument("--noise-level", default="normal", choices=["low", "normal", "high"])
    parser.add_argument("--backend-condition", default="unknown")
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
    parser.add_argument(
        "--use-applicability-mask",
        action="store_true",
        help=(
            "Enable per-step applicability mask recomputation. "
            "This is slower; disabled by default for faster training."
        ),
    )
    return parser


def _main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    seeds = parse_seed_list(args.seeds)
    try:
        priority_weights = parse_optional_json_object(
            args.priority_weights_json,
            field_name="priority-weights-json",
        )
    except ValueError as exc:
        parser.error(str(exc))
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
        algo=args.algo,
        device=args.device,
        strict_device=args.strict_device,
        inference_backend=args.inference_backend,
        strict_inference_backend=args.strict_inference_backend,
        onnx_sync_interval=args.onnx_sync_interval,
        onnx_qnn_backend_path=args.onnx_qnn_backend_path,
        use_nexa_sdk=args.use_nexa_sdk,
        export_android_onnx=args.export_android_onnx,
        android_onnx_static_batch=args.android_onnx_static_batch,
        android_onnx_int8=args.android_onnx_int8,
        android_qnn_strict_check=args.android_qnn_strict_check,
        priority_profile_id=args.priority_profile_id,
        priority_weights=priority_weights,
        max_depth_budget=args.max_depth_budget,
        max_latency_ms=args.max_latency_ms,
        max_shots=args.max_shots,
        queue_level=args.queue_level,
        noise_level=args.noise_level,
        backend_condition=args.backend_condition,
        train_mode=args.train_mode,
        mixed_pad_level_min=args.mixed_pad_min,
        mixed_pad_level_max=args.mixed_pad_max,
        eval_every=args.eval_every,
        eval_episodes_curve=args.curve_eval_episodes,
        use_applicability_mask=args.use_applicability_mask,
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
