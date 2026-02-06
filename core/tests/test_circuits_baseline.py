from qiskit.circuit import QuantumCircuit
from core.circuits_baseline import BASELINE_BUILDERS, get_builder

def test_get_builder_and_outputs():
    for name in BASELINE_BUILDERS.keys():
        builder = get_builder(name)
        qc = builder(2)
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits >= 1
        assert len(qc.data) > 0
