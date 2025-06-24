from qiskit import *
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

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
#
simulator = AerSimulator(method="matrix_product_state")
circuit.measure(2,2)
result = simulator.run(circuit, shots=1000).result()
counts = result.get_counts()
fig = plot_histogram(counts)
fig.savefig('teleportation/init_histogram.png')

circuit_viz = circuit.draw(output='text')
with open(f"teleportation/init.txt", "w") as file:
    file.write(str(circuit_viz))

