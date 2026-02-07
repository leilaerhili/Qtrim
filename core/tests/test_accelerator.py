import pytest

from core import accelerator as accel


class _FakeDevice:
    def __str__(self) -> str:
        return "privateuseone:0"


def _fake_probe(
    *,
    npu: bool = False,
    directml: bool = False,
    cuda: bool = False,
    xpu: bool = False,
    mps: bool = False,
    directml_device=None,
) -> accel._ProbeResult:
    availability = {
        "npu": npu,
        "directml": directml,
        "cuda": cuda,
        "xpu": xpu,
        "mps": mps,
        "cpu": True,
    }
    return accel._ProbeResult(availability=availability, directml_device=directml_device)


def test_auto_device_picks_cpu_when_no_accelerator(monkeypatch):
    monkeypatch.setattr(accel, "_probe_accelerators", lambda: _fake_probe())
    device, resolution = accel.resolve_training_device(requested="auto", strict=False)
    assert device == "cpu"
    assert resolution.resolved == "cpu"
    assert resolution.backend == "torch"
    assert resolution.fallback_used is False


def test_npu_request_falls_back_to_directml(monkeypatch):
    fake_device = _FakeDevice()
    monkeypatch.setattr(
        accel,
        "_probe_accelerators",
        lambda: _fake_probe(directml=True, directml_device=fake_device),
    )
    device, resolution = accel.resolve_training_device(requested="npu", strict=False)
    assert device is fake_device
    assert resolution.backend == "torch_directml"
    assert resolution.resolved == "privateuseone:0"
    assert resolution.fallback_used is True


def test_npu_request_strict_raises_if_unavailable(monkeypatch):
    monkeypatch.setattr(accel, "_probe_accelerators", lambda: _fake_probe())
    with pytest.raises(RuntimeError):
        accel.resolve_training_device(requested="npu", strict=True)


def test_unknown_device_name_raises():
    with pytest.raises(ValueError):
        accel.resolve_training_device(requested="banana", strict=False)
