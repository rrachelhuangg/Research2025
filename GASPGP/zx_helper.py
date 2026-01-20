"""
Implementation of various helper functions relating to ZX-calculus.
"""


import pyzx as zx
from qiskit.qasm2 import dumps
from qiskit import QuantumCircuit


def apply_zx_calc(circuit, n: int=3):
    """
    Apply PYZX's ZX-Calculus implementation to a QuantumCircuit.
    """
    #convert to qasm intermediary format
    # print("PRE OPT:\n")
    # print(circuit.draw(output='text'))
    with open(f"gasp/{n}.txt", "w") as f:
        qasm_circuit = str(dumps(circuit))
        f.write(qasm_circuit)
    with open(f"gasp/{n}.txt", "r") as f:
        lines = f.read().split('\n')
        format_lines = ""
        for l in lines:
            if 'creg' not in l and 'measure' not in l and 'barrier' not in l:
                format_lines += (l+'\n')
    qasm_file_name = f"gasp/{n}.qasm"
    with open(qasm_file_name, "w") as f:
        qasm_circuit = QuantumCircuit.from_qasm_str(format_lines)
        formatted = dumps(qasm_circuit)
        f.write(formatted)
    #apply zx calculus
    zx_qasm_circuit = zx.Circuit.load(qasm_file_name)
    graph = zx_qasm_circuit.to_graph(compress_rows=True)
    zx.full_reduce(graph, quiet=True)
    # print("\n")
    graph.normalize()
    optimized_circuit = zx.extract_circuit(graph.copy())
    g = zx_qasm_circuit
    p = optimized_circuit
    opt_qasm = optimized_circuit.to_qasm()
    opt_circuit = QuantumCircuit.from_qasm_str(opt_qasm)
    # print("POST OPT: \n")
    # print(opt_circuit.draw(output='text'))
    # print('Pre ZX-calculus optimized stats: ')
    # print(g.to_basic_gates().stats())
    # print('Post ZX-calculus optimized stats: ')
    # print(p.to_basic_gates().stats())
    # print("optimization validation check: ", zx.compare_tensors(g, p))
    # print("optimization validation check: ", zx_qasm_circuit.verify_equality(optimized_circuit, up_to_swaps=False, up_to_global_phase=True))
    return opt_circuit, g.to_basic_gates().stats(), p.to_basic_gates().stats()


def parse_circuit_stats(values):
    gate_types = {"T-count":0, "Cliffords":0, "2-qubit":0, "Hadamard":0}
    lines = values.split("\n")[1:]
    stripped_lines = []
    for line in lines:
        stripped_lines.append(line.strip().split(" "))
    gate_types["T-count"] = int(stripped_lines[0][0])
    gate_types["Cliffords"] = int(stripped_lines[1][0])
    gate_types["2-qubit"] = int(stripped_lines[2][0])
    gate_types["Hadamard"] = int(stripped_lines[3][0])
    return gate_types


def assign_zx_value(circuit):
    optimized_circuit, g_vals, p_vals = apply_zx_calc(circuit)
    pre_zx_stats = parse_circuit_stats(g_vals)
    post_zx_stats = parse_circuit_stats(p_vals)
    differences = []
    for key in pre_zx_stats:
        diff = pre_zx_stats[key] - post_zx_stats[key]
        if pre_zx_stats[key] != 0:
            divide = diff/pre_zx_stats[key]
            differences.append(divide*100)
        else:
            differences.append(0)
    average_decrease = sum(differences)/len(differences)
    return average_decrease
