import pyzx as zx
from qiskit.qasm2 import dumps
from qiskit import QuantumCircuit


def visualize(circuit):
    """
    Helps to visualize a circuit by displaying its gates and qubits in the terminal. 
    """
    print(circuit.draw(output='text'))


def stats_circuit(message, circuit):
    """
    Output the gate statistics for a circuit.
    """
    print(message)
    print(circuit.to_basic_gates().stats())
    print('\n')


def apply_zx_calc(circuit, n: int=3):
    """
    Apply PYZX's ZX-Calculus implementation to a QuantumCircuit.
    """
    #convert to qasm intermediary format
    print("PRE OPT:\n")
    visualize(circuit)
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
    #print("ZX-Calculus Reduction Steps:")
    #print("----------------------------")
    # zx.full_reduce(graph, quiet=False)
    zx.full_reduce(graph, quiet=True)
    print("\n")
    graph.normalize()
    optimized_circuit = zx.extract_circuit(graph.copy())
    g = zx_qasm_circuit
    p = optimized_circuit
    opt_qasm = optimized_circuit.to_qasm()
    opt_circuit = QuantumCircuit.from_qasm_str(opt_qasm)
    print("POST OPT: \n")
    visualize(opt_circuit)
    stats_circuit('Pre ZX-calculus optimized stats: ', g)
    stats_circuit('Post ZX-calculus optimized stats: ', p)
    # print("optimization validation check: ", zx.compare_tensors(g, p))
    # print("optimization validation check: ", zx_qasm_circuit.verify_equality(optimized_circuit, up_to_swaps=False, up_to_global_phase=True))
    return opt_circuit
