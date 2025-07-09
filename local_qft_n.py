from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit_algorithms import AmplificationProblem, Grover
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import ZGate
from qiskit_aer import AerSimulator
from qiskit.primitives import BackendSampler
import time, sys, math
from qiskit.qasm2 import dumps
import pyzx as zx
from qiskit.circuit.library import QFT

def input_state(circuit, q, n): #q = QuantumRegister(n), c = ClassicalRegister(3)
    """n-qubit state for QFT that produces output 1"""
    for j in range(n):
        circuit.h(q[j])
        circuit.p(-math.pi/float(2**(j)), q[j])

def qft(circuit, q, n):
    """n-qubit QFT on q in circuit"""
    for j in range(n):
        for k in range(j):
            circuit.cp(math.pi/float(2**(j-k)), q[j], q[k])
        circuit.h(q[j])

def controller(n: int=3):
    quantum_bits = QuantumRegister(n)
    classical_bits = ClassicalRegister(n)
    qft_circuit = QuantumCircuit(quantum_bits, classical_bits)
    input_state(qft_circuit, quantum_bits, n)
    qft(qft_circuit, quantum_bits, n)
    for i in range(n):
        qft_circuit.measure(quantum_bits[i], classical_bits[i])
    # with open(f"qft/qft_circuit_{n}.qasm", "w") as f:
    #     qasm_circuit = dumps(qft_circuit)
    #     f.write(qasm_circuit)
    with open(f"qft/pre_zx_{n}_circuit.txt", "w") as file:
        file.write(str(qft_circuit))
    simulator = AerSimulator(method="matrix_product_state")
    pre_start_time = time.time()
    result = simulator.run(qft_circuit, shots=1000).result()
    counts = result.get_counts()
    pre_end_time = time.time()
    print("PRE OPT RUN TIME: ", round(pre_end_time-pre_start_time, 6))
    print("RESULT: ", counts)
    zx_qasm_circuit = zx.Circuit.load(f"qft/qft_circuit_{n}.qasm")
    graph = zx_qasm_circuit.to_graph(compress_rows=True)
    print("INITIAL GRAPH STATS: ", graph.stats())
    print("INITIAL STATS 2: ", zx_qasm_circuit.to_basic_gates().stats())
    zx.full_reduce(graph, quiet=False)
    graph.normalize() #zx_full_reduce_init
    print("POST GRAPH STATS: ", graph.stats())
    optimized_circuit = zx.extract_circuit(graph.copy()) #zx_full_reduce_init_extracted_circuit
    #for post optimization validation
    g = zx_qasm_circuit
    p = optimized_circuit
    #converting back to qiskit circuit for testing and drawing
    opt_qasm = optimized_circuit.to_qasm()
    opt_circuit = QuantumCircuit.from_qasm_str(opt_qasm)
    print("POST OPTIMIZATION STATS: ", p.to_basic_gates().stats())
    opt_circuit_viz = opt_circuit.draw(output='text') #use pyzx validation too HERE
    with open(f"qft/post_zx_{n}circuit.txt", "w") as file:
        file.write(str(opt_circuit_viz))
    opt_circuit.add_register(ClassicalRegister(n))
    opt_circuit.measure(range(n), range(n))
    simulator = AerSimulator()
    post_start_time = time.time()
    counts = simulator.run(opt_circuit, shots=1000).result().get_counts()
    print("COUNTS: ", counts)
    post_end_time = time.time()
    print("POST OPT RUN TIME: ", round(post_end_time-post_start_time, 6))
    print("optimization validation check: ", zx.compare_tensors(g, p))
    print("optimization validation check: ", zx_qasm_circuit.verify_equality(optimized_circuit, up_to_swaps=False, up_to_global_phase=True))

if __name__ == '__main__':
    n = int(sys.argv[1])
    controller(n)


