from qiskit import QuantumCircuit
from qiskit.circuit.library import HGate, MCXGate
from qiskit.circuit.library import zz_feature_map 
from qiskit.circuit.quantumcircuit import QuantumCircuit

mcx_gate = MCXGate(3)
hadamard_gate = HGate()
 
qc = QuantumCircuit(4)
qc.append(hadamard_gate, [0])
qc.append(mcx_gate, [0, 1, 2, 3])
qc.draw(filename="circuit_drawings/normal_circuit.txt")

###
features = [0.2, 0.4, 0.8]
feature_map = zz_feature_map(feature_dimension=len(features))
 
encoded = feature_map.assign_parameters(features)
encoded.draw(filename="circuit_drawings/angle_encoding.txt")

###
qc = QuantumCircuit(2)
qc.cx(0, 1)
qc.cx(0, 1)
qc.draw("circuit_drawings/template_circuit.txt")