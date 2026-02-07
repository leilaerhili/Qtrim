from core.circuits_baseline import BASELINE_BUILDERS
from core.train_policy import build_arg_parser


def test_train_policy_baseline_choices_match_builders():
    parser = build_arg_parser()
    baseline_arg = next(a for a in parser._actions if a.dest == "baseline")
    assert sorted(baseline_arg.choices) == sorted(BASELINE_BUILDERS.keys())
    assert baseline_arg.default in BASELINE_BUILDERS

    ent_coef_arg = next(a for a in parser._actions if a.dest == "ent_coef")
    n_steps_arg = next(a for a in parser._actions if a.dest == "n_steps")
    n_envs_arg = next(a for a in parser._actions if a.dest == "n_envs")
    algo_arg = next(a for a in parser._actions if a.dest == "algo")
    device_arg = next(a for a in parser._actions if a.dest == "device")
    strict_device_arg = next(a for a in parser._actions if a.dest == "strict_device")
    inference_backend_arg = next(a for a in parser._actions if a.dest == "inference_backend")
    strict_inference_backend_arg = next(
        a for a in parser._actions if a.dest == "strict_inference_backend"
    )
    onnx_sync_interval_arg = next(a for a in parser._actions if a.dest == "onnx_sync_interval")
    use_nexa_sdk_arg = next(a for a in parser._actions if a.dest == "use_nexa_sdk")
    export_android_onnx_arg = next(a for a in parser._actions if a.dest == "export_android_onnx")
    android_onnx_static_batch_arg = next(
        a for a in parser._actions if a.dest == "android_onnx_static_batch"
    )
    android_onnx_int8_arg = next(a for a in parser._actions if a.dest == "android_onnx_int8")
    android_qnn_strict_check_arg = next(
        a for a in parser._actions if a.dest == "android_qnn_strict_check"
    )
    priority_profile_id_arg = next(a for a in parser._actions if a.dest == "priority_profile_id")
    queue_level_arg = next(a for a in parser._actions if a.dest == "queue_level")
    noise_level_arg = next(a for a in parser._actions if a.dest == "noise_level")
    train_mode_arg = next(a for a in parser._actions if a.dest == "train_mode")
    assert ent_coef_arg.default == 0.01
    assert n_steps_arg.default == 1024
    assert n_envs_arg.default == 1
    assert algo_arg.default == "dqn"
    assert device_arg.default == "directml"
    assert strict_device_arg.default is False
    assert inference_backend_arg.default == "ort-qnn"
    assert strict_inference_backend_arg.default is False
    assert onnx_sync_interval_arg.default == 2000
    assert use_nexa_sdk_arg.default is True
    assert export_android_onnx_arg.default is True
    assert android_onnx_static_batch_arg.default == 1
    assert android_onnx_int8_arg.default is True
    assert android_qnn_strict_check_arg.default is True
    assert priority_profile_id_arg.default == "auto"
    assert queue_level_arg.default == "normal"
    assert noise_level_arg.default == "normal"
    assert sorted(train_mode_arg.choices) == ["fixed", "mixed"]
