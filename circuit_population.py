from generate_random_circuit import generate_random
from qiskit import QuantumCircuit, qpy
from qiskit.qasm2 import dumps
import pyzx as zx

def apply_zx_calc(circuit, n: int=3):
    #convert to qasm intermediary format
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
    #apply zx calculus
    zx_qasm_circuit = zx.Circuit.load(qasm_file_name)
    graph = zx_qasm_circuit.to_graph(compress_rows=True)
    print("ZX-Calculus Reduction Steps:")
    print("----------------------------")
    zx.full_reduce(graph, quiet=False)
    print("\n")
    graph.normalize()
    optimized_circuit = zx.extract_circuit(graph.copy())
    g = zx_qasm_circuit
    p = optimized_circuit
    opt_qasm = optimized_circuit.to_qasm()
    opt_circuit = QuantumCircuit.from_qasm_str(opt_qasm)
    return opt_circuit

for i in range(10):
    circuit = generate_random(5)
    post_opt_circuit = apply_zx_calc(circuit) #circuit expands after optimization
    with open(f"circuit_population/visualization_{i}.txt", "w") as f:
        f.write((str(post_opt_circuit))) 
    with open(f"circuit_population/{i}.qpy", "wb") as f:
        qpy.dump(post_opt_circuit, f)
