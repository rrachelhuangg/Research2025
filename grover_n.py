from qiskit import QuantumCircuit
from qiskit_algorithms import AmplificationProblem, Grover
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import ZGate
from qiskit_aer import AerSimulator
from qiskit.primitives import BackendSampler
import time, sys

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

    grover = Grover(sampler=sampler)
    result = grover.amplify(problem)

    end_time = time.time()
    with open(f"grover_drawings/{m}_{n}.txt", "w") as file:
        file.write(f"Total elapsed time: {round(end_time-start_time, 3)}\n\nTop measurement:{result.top_measurement}\n\n{str(oracle_viz)}\n")

if __name__ == '__main__':
    n = int(sys.argv[1])
    m = sys.argv[2]
    controller(n, m)


