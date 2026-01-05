"""
Qiskit documentation for the QAOA algorithm. 
Link: https://quantum.cloud.ibm.com/docs/en/tutorials/quantum-approximate-optimization-algorithm
Approach: treat gate angles as a parameter to be optimized.
"""

import numpy as np
from scipy.optimize import minimize
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Operator
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer.primitives import EstimatorV2 as Estimator

#example circuit
circuit = QuantumCircuit(2, 2)
circuit.x(1)

#for transpilation
basis_gates = ['x', 'y', 'z', 'cx']

#arbitrary parameters to start with
initial_gamma = np.pi
initial_beta = np.pi / 2
init_params = [initial_beta, initial_beta, initial_gamma, initial_gamma]


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
    """
    Parameters that the QAOA layer will try to optimize.
    """
    created_circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=2)
    created_circuit.measure_all()
    parameters = created_circuit.parameters
    return created_circuit, parameters


def transpile_circuit(circuit):
    """
    Converts the abstraction of the quantum algorithm into a tangible representation.
    """
    simulator = AerSimulator()
    compiled_circuit = transpile(circuit, simulator)
    return compiled_circuit


objective_func_vals = []
def cost_func_estimator(params, ansatz, hamiltonian, estimator):
    isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)
    pub = (ansatz, isa_hamiltonian, params)
    job = estimator.run([pub])
    results = job.result()[0]
    cost = results.data.evs
    objective_func_vals.append(cost)
    return cost


def find_optimal_parameters(candidate_circuit, cost_hamiltonian):
    estimator = Estimator()
    result = minimize(
        cost_func_estimator,
        init_params,
        args=(candidate_circuit, cost_hamiltonian, estimator),
        method="COBYLA",
        tol=1e-2,
    )
    return result


def optimize_angles(circuit):
    cost_hamiltonian = circuit_to_paulis(circuit)
    created_circuit, parameters = retrieve_qaoa_sample_parameters(cost_hamiltonian)
    candidate_circuit = transpile_circuit(created_circuit)
    result = find_optimal_parameters(candidate_circuit, cost_hamiltonian)
    optimized_circuit = candidate_circuit.assign_parameters(result.x)
    return optimized_circuit


def test_circuit(optimized_circuit):
    simulator = AerSimulator()
    optimized_circuit.measure_all()
    compiled_circuit = transpile(optimized_circuit, simulator)
    job = simulator.run(optimized_circuit, shots=1000)
    result = job.result()
    counts = result.get_counts(compiled_circuit)
    return counts

optimized_circuit = optimize_angles(circuit)
print(test_circuit(optimized_circuit))

