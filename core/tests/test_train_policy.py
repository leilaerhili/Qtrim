from core.circuits_baseline import BASELINE_BUILDERS
from core.train_policy import build_arg_parser


def test_train_policy_baseline_choices_match_builders():
    parser = build_arg_parser()
    baseline_arg = next(a for a in parser._actions if a.dest == "baseline")
    assert sorted(baseline_arg.choices) == sorted(BASELINE_BUILDERS.keys())
    assert baseline_arg.default in BASELINE_BUILDERS

    ent_coef_arg = next(a for a in parser._actions if a.dest == "ent_coef")
    n_steps_arg = next(a for a in parser._actions if a.dest == "n_steps")
    assert ent_coef_arg.default == 0.01
    assert n_steps_arg.default == 1024
