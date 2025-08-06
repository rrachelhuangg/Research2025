import matplotlib
import matplotlib.pyplot as plt
import rustworkx as rx
from rustworkx.visualization import mpl_draw as draw_graph
import numpy as np
from scipy.optimize import minimize
from collections import defaultdict
from typing import Sequence
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_aer import AerSimulator

n = 5

graph = rx.PyGraph()
graph.add_nodes_from(np.arange(0, n, 1))
edge_list = [
    (0, 1, 1.0), #all edges have weight of 1.0
    (0, 2, 1.0),
    (0, 4, 1.0),
    (1, 2, 1.0),
    (2, 3, 1.0),
    (3, 4, 1.0)
]
graph.add_edges_from(edge_list)
# fig = draw_graph(graph, node_size=600, with_labels=True)
# fig.savefig('qaoa/graph.png')

def build_max_cut_paulis(graph: rx.PyGraph) -> list[tuple[str, float]]:
    """Convert the graph to Pauli list"""
    pauli_list = []
    for edge in list(graph.edge_list()):
        weight = graph.get_edge_data(edge[0], edge[1])
        pauli_list.append(("ZZ", [edge[0], edge[1]], weight))
    return pauli_list

max_cut_paulis = build_max_cut_paulis(graph)
cost_hamiltonian = SparsePauliOp.from_sparse_list(max_cut_paulis, n)
print("Cost function hamiltonian: ", cost_hamiltonian)

circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps = 2)
circuit.measure_all()
print(circuit.draw(output='text'))
print("PARAMETERS: ", circuit.parameters)

service = QiskitRuntimeService()
backend = service.backend("ibm_brisbane")
opt_level = 0
layout_method = ''
routing_method = ''
translation_method = ''
pass_manager = generate_preset_pass_manager(
    optimization_level=opt_level, backend=backend, layout_method=layout_method, routing_method=routing_method, translation_method=translation_method
)
transpiled = pass_manager.run(circuit)
# print("TRANSPILED: ", transpiled.draw(output='text'))
objective_func_vals = []

initial_gamma = np.pi
initial_beta = np.pi/2
init_params = [initial_beta, initial_beta, initial_gamma, initial_gamma]

def cost_func_estimator(params, ansatz, hamiltonian, estimator):
    isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)
    pub = (ansatz, isa_hamiltonian, params)
    job = estimator.run([pub])
    results = job.result()[0]
    cost = results.data.evs
    objective_func_vals.append(cost)
    return cost

bound_circuit = transpiled.assign_parameters(init_params)
simulator = AerSimulator(method="matrix_product_state")
result = simulator.run(bound_circuit, shots=1000).result().get_counts()
print("RESULT: ", result)

# estimator = Estimator(mode=backend)
# #error suppression/mitigation options
# estimator.options.default_shots = 1000
# estimator.options.dynamical_decoupling.enable = True
# estimator.options.dynamical_decoupling.sequence_type = "XY4"
# estimator.options.twirling.enable_gates = True
# estimator.options.twirling.num_randomizations = "auto"
# result = minimize(cost_func_estimator, init_params, args=(transpiled, cost_hamiltonian, estimator), method="COBYLA", tol=1e-2)
# print("RESULT: ", result)

