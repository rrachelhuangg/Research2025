from qiskit.circuit.library import QuantumVolume
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler import generate_preset_pass_manager


service = QiskitRuntimeService()
backend = service.backend("ibm_torino")
pm = generate_preset_pass_manager(backend=backend, optimization_level=3)


def qv_circuit(n):
    qv_circuit_whole = QuantumVolume(num_qubits=n)
    # qv_circuit_transpiled = pm.run(qv_circuit_whole)
    # print(qv_circuit_transpiled.draw(output='text'))
    print(qv_circuit_whole.draw(output='text'))
    print(backend.basis_gates)
    return qv_circuit_whole
