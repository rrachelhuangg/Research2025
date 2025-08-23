import matplotlib.pyplot as plt
import rustworkx as rx
from rustworkx.visualization import mpl_draw as draw_graph
import numpy as np
from typing import Sequence
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit import QuantumCircuit, transpile
from qiskit.primitives import Sampler
import time, sys
from qiskit.qasm2 import dumps
import pyzx as zx
import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler import generate_preset_pass_manager, PassManager
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.qasm2 import dumps
from qiskit.circuit import library as lib
from qiskit.transpiler.passes import (
    ALAPScheduleAnalysis,
    InverseCancellation,
    PadDynamicalDecoupling,
)

def apply_zx_calc(oracle, n: int=3,):
    with open(f"qaoa/{n}.qasm", "w") as f:
        basis_circuit = transpile(oracle)
        qasm_circuit = dumps(basis_circuit)
        f.write(qasm_circuit)
    zx_qasm_circuit = zx.Circuit.load(f"qaoa/{n}.qasm")
    graph = zx_qasm_circuit.to_graph(compress_rows=True)
    print("ZX-Calculus Reduction Steps:")
    print("----------------------------")
    zx.full_reduce(graph, quiet=False)
    print("\n")
    graph.normalize()
    optimized_circuit = zx.extract_circuit(graph.copy())
    #for post optimization validation
    g = zx_qasm_circuit
    p = optimized_circuit
    #converting back to qiskit circuit for testing and drawing
    opt_qasm = optimized_circuit.to_qasm()
    opt_circuit = QuantumCircuit.from_qasm_str(opt_qasm)
    stats_circuit('Pre ZX-calculus optimized stats: ', g)
    stats_circuit('Post ZX-calculus optimized stats: ', p)
    return opt_circuit

def output_circuit(message, circuit):
    print(message)
    print(circuit)
    print('\n')


def stats_circuit(message, circuit):
    print(message)
    print(circuit.to_basic_gates().stats())
    print('\n')

def custom_pass_stages(backend, opt:int=1):
    if opt==1:
        dd_sequence = [lib.XGate(), lib.XGate()]
        scheduling_pm = PassManager(
            [
                ALAPScheduleAnalysis(target=backend.target),
                PadDynamicalDecoupling(target=backend.target, dd_sequence=dd_sequence),
            ]
        )
        inverse_gate_list = [
            lib.CXGate(),
            lib.HGate(),
            (lib.RXGate(np.pi / 4), lib.RXGate(-np.pi / 4)),
            (lib.PhaseGate(np.pi / 4), lib.PhaseGate(-np.pi / 4)),
            (lib.TGate(), lib.TdgGate()),
        ]
        logical_opt = PassManager([InverseCancellation(inverse_gate_list)])
        return logical_opt, scheduling_pm

def build_max_cut_paulis(graph: rx.PyGraph) -> list[tuple[str, float]]:
    """Convert the graph to Pauli list"""
    pauli_list = []
    for edge in list(graph.edge_list()):
        weight = graph.get_edge_data(edge[0], edge[1])
        pauli_list.append(("ZZ", [edge[0], edge[1]], weight))
    return pauli_list

def plot_result(G, x):
    colors = ["tab:grey" if i == 0 else "tab:purple" for i in x]
    pos, _default_axes = rx.spring_layout(G), plt.axes(frameon=True)
    return rx.visualization.mpl_draw(
        G, node_color=colors, node_size=100, alpha=0.8, pos=pos
    )

def evaluate_sample(x: Sequence[int], graph: rx.PyGraph) -> float:
    assert len(x) == len(list(graph.nodes())), "The length of x must coincide with the number of nodes in the graph."
    return sum(x[u]*(1-x[v])+x[v]*(1-x[u]) for u, v in list(graph.edge_list()))

#auxiliary function to sample most likely bitstring (that represents the max-cut problem)
def to_bitstring(integer, num_bits):
    result = np.binary_repr(integer, width=num_bits)
    return [int(digit) for digit in result]

def controller(n: int=15, hardware: str='s', opt_method:str=''):
    start_time, end_time = None, None
    initial_gamma = np.pi
    initial_beta = np.pi/2
    init_params = [initial_beta, initial_beta, initial_gamma, initial_gamma]
    if hardware=='r':
        service = QiskitRuntimeService()
        backend = service.backend("ibm_torino")
        node_range = np.arange(0, n, 1)
        elist = []
        for edge in backend.coupling_map:
            if edge[0] < n and edge[1] < n:
                elist.append((edge[0], edge[1], 1.0))
        for i in range(3):  #need to make sure that trials are run on the same randomly initialized max-cut problem graph
            if opt_method=='':
                message=f'Unoptimized circuit: '
            else:
                message=f'Pre {opt_method} optimized circuit: '
            opt_level = 0
            layout_method = ''
            routing_method = ''
            translation_method = ''
            graph = rx.PyGraph()
            graph.add_nodes_from(node_range)
            graph.add_edges_from(elist)
            large_fig = draw_graph(graph, node_size=200, with_labels=True)
            large_fig.savefig('qaoa/pre_split_graph.png')
            max_cut_paulis = build_max_cut_paulis(graph)
            cost_hamiltonian = SparsePauliOp.from_sparse_list(max_cut_paulis, n)

            pre_bound_params_circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps = 2)
            circuit = pre_bound_params_circuit.assign_parameters(init_params)
            if opt_method=='ZX':
                circuit = apply_zx_calc(circuit, n)
                message='Post ZX-calculus optimized circuit: '
            elif opt_method=='plugin-combo':
                message = f'Post {opt_method} optimized circuit: '
                opt_level=3
                layout_method='sabre'
                routing_method='stochastic'
                translation_method='synthesis'
            oracle_viz = circuit.draw(output='text')
            output_circuit(message, oracle_viz)
            circuit.measure_all()
            # print(circuit.draw(output='text'))

            pass_manager = generate_preset_pass_manager(
                optimization_level=opt_level, backend=backend, layout_method=layout_method, routing_method=routing_method, translation_method=translation_method
            )
            if opt_method=='custom-pass':
                custom_stage_1, custom_stage_2 = custom_pass_stages(backend, 1)
                pass_manager.pre_layout = custom_stage_1
                pass_manager.scheduling = custom_stage_2
            transpiled = pass_manager.run(circuit)
            print('Transpiled circuit stats: ', transpiled.count_ops())

            sampler = Sampler(mode=backend)
            #simple error supression/mitigation options
            sampler.options.dynamical_decoupling.enable = True
            sampler.options.dynamical_decoupling.sequence_type = "XY4"
            sampler.options.twirling.enable_gates = True
            sampler.options.twirling.num_randomizations = "auto"
            #run the circuit
            start_time = time.time()
            job = sampler.run([(transpiled,)])
            end_time = time.time()
            counts_int = job.result()[0].data.meas.get_int_counts()
            shots = sum(counts_int.values())
            final_distribution_int = {key: val/shots for key, val in counts_int.items()}

            keys = list(final_distribution_int.keys())
            values = list(final_distribution_int.values())
            most_likely = keys[np.argmax(np.abs(values))]
            most_likely_bitstring = to_bitstring(most_likely, len(graph))
            most_likely_bitstring.reverse()
            print("RESULT BIT STRING: ", most_likely_bitstring)
            split_fig = plot_result(graph, most_likely_bitstring)
            split_fig.savefig('qaoa/post_split_graph.png')
            cut_value = evaluate_sample(most_likely_bitstring, graph)
            print("The value of the cut is: ", cut_value)
            print(f"Time taken: {end_time-start_time}\n")

if __name__ == '__main__':
    n = int(sys.argv[1])
    hardware = sys.argv[2]
    if len(sys.argv)>3:
        opt_method = sys.argv[3]
        controller(n, hardware, opt_method)
    else:
        controller(n, hardware, '')