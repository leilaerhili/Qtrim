"""
Accelerator selection helpers for training/inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch as th


DeviceArg = Union[str, th.device]

_ORDERED_BACKENDS = ("npu", "directml", "cuda", "xpu", "mps", "cpu")
_VALID_REQUESTS = {"auto", "cpu", "cuda", "mps", "xpu", "npu", "directml", "dml"}
_BACKEND_LABELS = {
    "cpu": "torch",
    "cuda": "torch",
    "mps": "torch",
    "xpu": "torch",
    "npu": "torch_npu",
    "directml": "torch_directml",
}


@dataclass(frozen=True)
class _ProbeResult:
    availability: Dict[str, bool]
    directml_device: Optional[th.device]


@dataclass(frozen=True)
class DeviceResolution:
    requested: str
    resolved: str
    backend: str
    reason: str
    availability: Dict[str, bool]
    strict: bool
    fallback_used: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requested": self.requested,
            "resolved": self.resolved,
            "backend": self.backend,
            "reason": self.reason,
            "availability": dict(self.availability),
            "strict": bool(self.strict),
            "fallback_used": bool(self.fallback_used),
        }


def _safe_available(checker) -> bool:
    try:
        return bool(checker())
    except Exception:
        return False


def _load_directml_device() -> Optional[th.device]:
    try:
        import torch_directml
    except Exception:
        return None
    try:
        return torch_directml.device()
    except Exception:
        return None


def _probe_accelerators() -> _ProbeResult:
    directml_device = _load_directml_device()
    availability = {
        "npu": _safe_available(lambda: hasattr(th, "npu") and th.npu.is_available()),
        "directml": directml_device is not None,
        "cuda": _safe_available(lambda: th.cuda.is_available()),
        "xpu": _safe_available(lambda: hasattr(th, "xpu") and th.xpu.is_available()),
        "mps": _safe_available(
            lambda: hasattr(th.backends, "mps") and th.backends.mps.is_available()
        ),
        "cpu": True,
    }
    return _ProbeResult(availability=availability, directml_device=directml_device)


def _pick_available(order: Tuple[str, ...], availability: Dict[str, bool]) -> Optional[str]:
    for candidate in order:
        if availability.get(candidate, False):
            return candidate
    return None


def _fallback_order_for(requested: str) -> Tuple[str, ...]:
    if requested == "npu":
        return ("directml", "cuda", "xpu", "mps", "cpu")
    if requested in {"directml", "dml"}:
        return ("npu", "cuda", "xpu", "mps", "cpu")
    if requested == "cuda":
        return ("xpu", "mps", "cpu")
    if requested == "xpu":
        return ("cuda", "mps", "cpu")
    if requested == "mps":
        return ("cuda", "xpu", "cpu")
    if requested == "cpu":
        return tuple()
    return _ORDERED_BACKENDS


def resolve_training_device(
    requested: str = "auto",
    strict: bool = False,
) -> Tuple[DeviceArg, DeviceResolution]:
    normalized = str(requested).strip().lower()
    if normalized not in _VALID_REQUESTS:
        choices = ", ".join(sorted(_VALID_REQUESTS))
        raise ValueError(f"Unknown device '{requested}'. Valid options: {choices}")
    if normalized == "dml":
        normalized = "directml"

    probe = _probe_accelerators()
    availability = {name: bool(probe.availability.get(name, False)) for name in _ORDERED_BACKENDS}
    availability["cpu"] = True

    if normalized == "auto":
        selected = _pick_available(_ORDERED_BACKENDS, availability)
    else:
        selected = normalized if availability.get(normalized, False) else None

    fallback_used = False
    if selected is None:
        if strict:
            raise RuntimeError(
                f"Requested device '{normalized}' is not available. "
                f"Detected availability: {availability}"
            )
        selected = _pick_available(_fallback_order_for(normalized), availability)
        fallback_used = True
    if selected is None:
        selected = "cpu"
        fallback_used = True

    if selected == "directml":
        if probe.directml_device is None:
            if strict:
                raise RuntimeError(
                    "torch_directml was selected but the DirectML device could not be created."
                )
            selected = "cpu"
            fallback_used = True
            device: DeviceArg = "cpu"
        else:
            device = probe.directml_device
    else:
        device = selected

    if normalized == "auto":
        reason = (
            "Auto-selected accelerator using preference order "
            f"{' > '.join(_ORDERED_BACKENDS)}."
        )
    elif fallback_used and selected != normalized:
        reason = f"Requested '{normalized}' was unavailable; fell back to '{selected}'."
    else:
        reason = f"Requested '{normalized}' is available."

    resolved = str(device) if selected == "directml" else str(selected)
    resolution = DeviceResolution(
        requested=normalized,
        resolved=resolved,
        backend=_BACKEND_LABELS[selected],
        reason=reason,
        availability=availability,
        strict=bool(strict),
        fallback_used=bool(fallback_used),
    )
    return device, resolution


def format_device_resolution(resolution: DeviceResolution) -> str:
    avail = ", ".join(
        f"{name}={'yes' if resolution.availability.get(name, False) else 'no'}"
        for name in _ORDERED_BACKENDS
    )
    return (
        "[device] "
        f"requested={resolution.requested} "
        f"resolved={resolution.resolved} "
        f"backend={resolution.backend} "
        f"strict={int(resolution.strict)} "
        f"fallback={int(resolution.fallback_used)} "
        f"| {resolution.reason} "
        f"| availability: {avail}"
    )
