"""
Qiskit documentation for the QAOA algorithm. 
Approach: treat gate angles as a parameter to be optimized.
Link: https://quantum.cloud.ibm.com/docs/en/tutorials/quantum-approximate-optimization-algorithm
"""

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Operator
from qiskit.circuit.library import QAOAAnsatz

circuit = QuantumCircuit(2, 2)
circuit.x(1)
basis_gates = ['x', 'y', 'z', 'cx']


def circuit_to_paulis(circuit):
    """
    Pauli gates are just X, Y, Z, etc.
    Note: sparse_pauli_op is the cost function hamiltonian; i.e. the quantum
    definition of the problem. 
    """
    transpiled = transpile(circuit, basis_gates=basis_gates, optimization_level=0)
    operator = Operator.from_circuit(transpiled)
    sparse_pauli_op = SparsePauliOp.from_operator(operator)
    return sparse_pauli_op


def retrieve_qaoa_sample_parameters(cost_hamiltonian):
    created_circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=2)
    created_circuit.measure_all()
    parameters = created_circuit.parameters
    return parameters
    

cost_hamiltonian = circuit_to_paulis(circuit)
retrieve_qaoa_sample_parameters(cost_hamiltonian)

