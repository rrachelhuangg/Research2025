from qiskit import QuantumCircuit, transpile
from qiskit_algorithms import AmplificationProblem, Grover
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import ZGate
from qiskit_aer import AerSimulator
from qiskit.primitives import BackendSampler
import time, sys
from qiskit.qasm2 import dumps
import pyzx as zx

"""
- Statevector makes it easier to mark a state when number of qubits increases - 
just have to specify the marked state and then it makes the oracle circuit accordingly. Way slower though. 
- the statevector is internally mapped to a quantum circuit
- OR can mark the desired state by applying gates to a circuit of n fresh qubits (H superposition)
- this second method seems faster and the circuit drawing is way clearer.
- HXH gate sandwich in this second method's drawing is equivalent to a Z gate
- multi-controlled Z Gate (normal Z-gate flips the phase of the 1 component of a qubit's superposition)
- mcz flips the 1 component of the target qubit's state (last qubit in this case) when the state of all of the previous qubits is 1
- some stats so far:
  - ran out of memory error w method q for 50 qubits
  - q takes ~0.9 s for 15 qubits, but time went up to ~63s after upping to 20 qubits
  - for min 15 qubits, method s already takes ~>10s
"""

def controller(n: int=15, m: str='q'):
    start_time = time.time()
    if m == 's': #statevector method
        oracle = Statevector.from_label('1'*n)
    elif m == 'q': #fresh circuit method (H superposition initialization)
        oracle = QuantumCircuit(n)
        mcz_gate = ZGate().control(num_ctrl_qubits=(n-1),ctrl_state='1'*(n-1))
        oracle.append(mcz_gate, list(range(n)))

    problem = AmplificationProblem(oracle, is_good_state=['1'*n])
    oracle_viz = problem.grover_operator.oracle.decompose().draw(output='text')
    
    mps_backend = AerSimulator(method="matrix_product_state")
    sampler = BackendSampler(backend=mps_backend)

    pre_start_time = time.time()
    grover = Grover(sampler=sampler)
    result = grover.amplify(problem)
    end_time = time.time()
    with open(f"grover_drawings/pre_{m}_{n}.txt", "w") as file:
        file.write(f"Total elapsed time: {round(end_time-start_time, 3)}\n\nTop measurement:{result.top_measurement}\n\n{str(oracle_viz)}\n")
    pre_end_time = time.time()
    print("PRE OPT RUN TIME: ", round(pre_end_time-pre_start_time, 6))

    with open(f"grover_drawings/{m}_{n}.qasm", "w") as f:
        basis_circuit = transpile(oracle)
        qasm_circuit = dumps(basis_circuit)
        f.write(qasm_circuit)
    
    zx_qasm_circuit = zx.Circuit.load(f"grover_drawings/{m}_{n}.qasm")
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
    post_start_time = time.time()
    grover = Grover(sampler=sampler)
    problem_2 = AmplificationProblem(opt_circuit, is_good_state=['1'*n])
    result = grover.amplify(problem_2)
    with open(f"grover_drawings/pre_{m}_{n}.txt", "w") as file:
        file.write(f"Total elapsed time: {round(end_time-start_time, 3)}\n\nTop measurement:{result.top_measurement}\n\n{str(oracle_viz)}\n")
    post_end_time = time.time()
    print("POST OPT RUN TIME: ", round(post_end_time-post_start_time, 6))
    print("optimization validation check: ", zx.compare_tensors(g, p))
    print("optimization validation check: ", zx_qasm_circuit.verify_equality(optimized_circuit, up_to_swaps=False, up_to_global_phase=True))
    
if __name__ == '__main__':
    n = int(sys.argv[1])
    m = sys.argv[2]
    controller(n, m)


