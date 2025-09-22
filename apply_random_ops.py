import pyzx as zx
import random
from qiskit.qasm2 import dumps
from qiskit import QuantumCircuit, qpy
from mutate_circuit import mutate_circuit
from generate_random_circuit import generate_random

def circuit_to_graph(circuit, n):
    with open(f"temp_for_zx/{n}.txt", "w") as f:
            qasm_circuit = str(dumps(circuit))
            f.write(qasm_circuit)
    with open(f"temp_for_zx/{n}.txt", "r") as f:
        lines = f.read().split('\n')
        format_lines = ""
        for l in lines:
            if 'creg' not in l and 'measure' not in l and 'barrier' not in l:
                format_lines += (l+'\n')
    qasm_file_name = f"temp_for_zx/{n}.qasm"
    with open(qasm_file_name, "w") as f:
        qasm_circuit = QuantumCircuit.from_qasm_str(format_lines)
        formatted = dumps(qasm_circuit)
        f.write(formatted)
    zx_qasm_circuit = zx.Circuit.load(qasm_file_name)
    graph = zx_qasm_circuit.to_graph(compress_rows=True)
    return graph

#not including full_reduce
zx_operations = ["bialg_simp", "clifford_simp", "gadget_simp", "id_simp", "lcomp_simp", "phase_free_simp", "pivot_boundary_simp", "pivot_simp", "spider_simp", "supplementarity_simp"]
def apply_random_ops(circuit):
     num_ops = random.randint(50,100)
     curr_op = 0

     graph = circuit_to_graph(circuit, 5)
     initial_circuit = zx.extract_circuit(graph.copy())
     initial_qasm = initial_circuit.to_qasm()
     initial_circuit_viz = QuantumCircuit.from_qasm_str(initial_qasm)
     comparison_graph = graph.copy()
     print('Initial circuit stats: ', initial_circuit_viz.count_ops())
     print("INITIAL CIRCUIT: ", initial_circuit_viz)
     print("Initial number of ops: ", len(initial_circuit_viz.data))
     print("Random Application of ZX-Calculus Rules: ")

     print("NUM OPS: ", num_ops)
     while curr_op < num_ops:
          rand_idx = random.randint(0, len(zx_operations)-1)
          operation = zx_operations[rand_idx]
          if operation == "bialg_simp":
               print("bialg_simp")
               zx.bialg_simp(graph, quiet=True)
          elif operation == "clifford_simp":
               print("clifford_simp")
               zx.clifford_simp(graph, quiet=True)
          elif operation == "gadget_simp":
               print("gadget_simp")
               zx.gadget_simp(graph, quiet=True)
          elif operation == "id_simp":
               print("id_simp")
               zx.id_simp(graph, quiet=True)
          elif operation == "lcomp_simp":
               print("lcomp_simp")
               zx.lcomp_simp(graph, quiet=True)
          elif operation == "phase_free_simp":
               print("phase_free_simp")
               zx.phase_free_simp(graph, quiet=True)
          elif operation == "pivot_boundary_simp":
               print("pivot_boundary_simp")
               zx.pivot_boundary_simp(graph, quiet=True)
          elif operation == "pivot_simp":
               print("pivot_simp")
               zx.pivot_simp(graph, quiet=True)
          elif operation == "spider_simp":
               print("spider_simp")
               zx.spider_simp(graph, quiet=True)
          elif operation == "supplementarity_simp":
               print("supplementarity_simp")
               zx.supplementarity_simp(graph, quiet=True)
          curr_op += 1
     print("\n")
     graph.normalize()

     print("Full ZX-Calculus Method Application Stats: ", comparison_graph)
     zx.full_reduce(comparison_graph, quiet=False)
     comparison_circuit = zx.extract_circuit(comparison_graph.copy())
     comparison_qasm = comparison_circuit.to_qasm()
     comparison_circuit = QuantumCircuit.from_qasm_str(comparison_qasm)
     print("FULL_REDUCE OPTIMIZED CIRCUIT: ", comparison_circuit)
     print('Post full_reduce optimization stats: ', comparison_circuit.count_ops())
     print("Post full_reduce number of ops: ", len(comparison_circuit.data))

     optimized_circuit = zx.extract_circuit(graph.copy())
     opt_qasm = optimized_circuit.to_qasm()
     opt_circuit = QuantumCircuit.from_qasm_str(opt_qasm)
     return opt_circuit

with open("circuit_population/0.qpy", "rb") as file:
    circuit = qpy.load(file)[0]
    mutated_circuit = mutate_circuit(circuit)
    optimized_circuit = apply_random_ops(mutated_circuit)
    print("RANDOMLY OPTIMIZED CIRCUIT: ", optimized_circuit)
    print('Post random optimization stats: ', optimized_circuit.count_ops())
    print("Post random optimization number of ops: ", len(optimized_circuit.data))

