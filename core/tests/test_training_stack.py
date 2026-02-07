from core.circuits_baseline import make_seeded_challenge_builder
from core.env_quantum_opt import EnvConfig, QuantumOptEnv
from core.rewrites import ACTION_CANCEL_DOUBLE_CX, ACTION_CANCEL_INVERSE_RZ
from core.train_curriculum import default_curriculum, parse_seed_list as parse_curriculum_seed_list
from core.train_policy import make_random_mixed_builder, parse_seed_list as parse_policy_seed_list


def test_seeded_challenge_builder_is_deterministic():
    builder_a = make_seeded_challenge_builder(seed=123, num_qubits=5, depth=20)
    builder_b = make_seeded_challenge_builder(seed=123, num_qubits=5, depth=20)
    qc_a = builder_a(2)
    qc_b = builder_b(2)
    sig_a = [(ci.operation.name, tuple(ci.operation.params), tuple(qc_a.find_bit(q).index for q in ci.qubits)) for ci in qc_a.data]
    sig_b = [(ci.operation.name, tuple(ci.operation.params), tuple(qc_b.find_bit(q).index for q in ci.qubits)) for ci in qc_b.data]
    assert sig_a == sig_b


def test_env_can_expose_action_subset():
    builder = make_seeded_challenge_builder(seed=10, num_qubits=4, depth=12)
    cfg = EnvConfig(allowed_action_ids=(ACTION_CANCEL_DOUBLE_CX, ACTION_CANCEL_INVERSE_RZ))
    env = QuantumOptEnv(circuit_builder=builder, pad_level=1, config=cfg)
    obs, info = env.reset()
    assert env.action_space.n == 41
    assert len(info["available_actions"]) == 2
    assert sum(info["action_mask"]) <= 2
    assert obs.shape == (6,)


def test_seed_parsers():
    assert parse_policy_seed_list("0,1,2") == [0, 1, 2]
    assert parse_curriculum_seed_list("3, 4 ,5") == [3, 4, 5]


def test_curriculum_allocates_timesteps():
    phases = default_curriculum(20_000)
    assert len(phases) == 3
    assert sum(p.timesteps for p in phases) == 20_000


def test_mixed_builder_produces_circuits():
    builder = make_random_mixed_builder(seed=7, pad_level_min=1, pad_level_max=2)
    qc1 = builder(1)
    qc2 = builder(1)
    assert qc1.num_qubits >= 1
    assert len(qc1.data) > 0
    assert qc2.num_qubits >= 1
    assert len(qc2.data) > 0
