from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import CDKMRippleCarryAdder
from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit_ibm_runtime import QiskitRuntimeService
from GASP_steps import individual_to_circuit, run_circuit

adder = CDKMRippleCarryAdder(3, 'full', 'Full Adder')

operand1 = QuantumRegister(3, 'o1')
operand2 = QuantumRegister(3, 'o2')
anc = QuantumRegister(2, 'a')
cr = ClassicalRegister(4)

circ = QuantumCircuit(operand1, operand2, anc, cr)

circ.x([operand1[0], operand1[1]])
circ.x([operand2[0], operand2[2]])

circ.append(adder, [anc[0]] + operand1[0:3] + operand2[0:3] + [anc[1]])

circ.measure(operand2[0:3] + [anc[1]], cr)
print(circ.draw(output='text'))

service = QiskitRuntimeService()
backend = service.backend("ibm_torino")
simulator = AerSimulator.from_backend(backend)
tr_circ = transpile(circ, simulator)
result = simulator.run(tr_circ).result()
counts = result.get_counts()
print("COUNTS: ", counts)
