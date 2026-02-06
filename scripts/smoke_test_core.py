from qiskit.circuit import QuantumRegister, QuantumCircuit

from core.rewrites import list_actions, apply_action
from core.metrics import compute_metrics, weights_for_profile
from core.env_quantum_opt import QuantumOptEnv, EnvConfig

def builder(pad_level: int) -> QuantumCircuit:
    qr = QuantumRegister(3, "q")
    qc = QuantumCircuit(qr)
    qc.cx(qr[0], qr[1])
    qc.cx(qr[0], qr[1])
    qc.rz(0.7, qr[2])
    qc.rz(-0.7, qr[2])
    for _ in range(max(0, pad_level - 1)):
        qc.rz(0.1, qr[2])
        qc.rz(0.2, qr[2])
    return qc

def main():
    print("=== REWRITES + METRICS SMOKE TEST ===")
    qc = builder(2)
    w = weights_for_profile("balanced")
    m0 = compute_metrics(qc, weights=w)
    print("Baseline:", m0)

    for aid, name in list_actions():
        qc2, res = apply_action(qc, aid)
        if res.changed:
            m1 = compute_metrics(qc2, weights=w)
            print(f"\nApplied {aid} {name}: {res.message}")
            print("After:", m1)
            break

    print("\n=== ENV SMOKE TEST ===")
    env = QuantumOptEnv(builder, pad_level=2, config=EnvConfig(max_steps=10))
    obs, info = env.reset()
    print("Reset metrics:", info["metrics"])
    for t in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"t={t} action={info.get('last_action', {}).get('action_name')} reward={reward:.3f} metrics={info['metrics']}")
        if terminated or truncated:
            print("Episode ended.")
            break

if __name__ == "__main__":
    main()
