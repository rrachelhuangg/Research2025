from qiskit import *
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import pyzx as zx
from qiskit.qasm2 import dumps

#3 qubit teleportation circuit
circuit = QuantumCircuit(3,3)
circuit.x(0)
circuit.barrier()
#essentially entangling all of the qubits with each other
circuit.h(1)
circuit.cx(1,2)
circuit.cx(0,1)
circuit.h(0)
circuit.barrier()
#measure qubits 0 and 1 into classical bits 0 and 1
circuit.measure([0,1],[0,1])
#based on the classical bits measured/received, unitary x/z gates are applied to entangled qubit to get state idential to initial state
circuit.barrier()
circuit.cx(1,2)
circuit.cz(0,2)
#running classical simulation to compare with post optimization simulation
circuit.measure([0,1],[0,1])
circuit.barrier()
circuit.cx(1,2)
circuit.cz(0,2)
simulator = AerSimulator(method="matrix_product_state")
circuit.measure(2,2)
result = simulator.run(circuit, shots=1000).result()
counts = result.get_counts()
fig = plot_histogram(counts)
fig.savefig('teleportation/pre_opt_circuit_results.png')
# intermediate step to convert qiskit circuit to pyzx graph (convert to qasm format)
# with open("teleportation/qasm_init.qasm", "w") as f:
#     qasm_circuit = dumps(circuit)
#     f.write(qasm_circuit)
# have to go through and remove non-unitary qasm ops like creg, measure, and barrier from file first
#loading qasm circuit and converting to pyzx zx-calc graph form
zx_qasm_circuit = zx.Circuit.load("teleportation/qasm_init.qasm")
graph = zx_qasm_circuit.to_graph(compress_rows=True) #init_zx_graph
#optimization operation with zx calculus
zx.full_reduce(graph, quiet=False)
graph.normalize() #zx_full_reduce_init
optimized_circuit = zx.extract_circuit(graph.copy()) #zx_full_reduce_init_extracted_circuit
#for post optimization validation
g = zx_qasm_circuit
p = optimized_circuit
#converting back to qiskit circuit for testing and drawing
opt_qasm = optimized_circuit.to_qasm()
opt_circuit = QuantumCircuit.from_qasm_str(opt_qasm)
opt_circuit_viz = opt_circuit.draw(output='text') #use pyzx validation too HERE
with open(f"teleportation/post_zx_qiskit_circuit.txt", "w") as file:
    file.write(str(opt_circuit_viz))
#verifying post optimization testing by comparing results to pre optimization
opt_circuit.measure_all()
opt_circuit.barrier()
opt_circuit.cx(1,2)
opt_circuit.cz(0,2)
simulator = AerSimulator(method="matrix_product_state")
opt_circuit.measure(2,2)
result = simulator.run(opt_circuit, shots=1000).result()
counts = result.get_counts()
fig = plot_histogram(counts)
fig.savefig('teleportation/opt_circuit_results.png')
#validating with pyzx's <10 qubit direct linear map comparison
print("optimization validation check: ", zx.compare_tensors(g, p))


