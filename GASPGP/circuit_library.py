from qiskit.circuit.library import QuantumVolume, CDKMRippleCarryAdder, RGQFTMultiplier, WeightedAdder
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.random import random_circuit

service = QiskitRuntimeService()
backend = service.backend("ibm_torino")
# pm = generate_preset_pass_manager(backend=backend, optimization_level=3)


def qv_circuit(n):
    qv_circuit_whole = QuantumVolume(num_qubits=n)
    # qv_circuit_transpiled = pm.run(qv_circuit_whole)
    # print(qv_circuit_transpiled.draw(output='text'))
    print(qv_circuit_whole.draw(output='text'))
    print(backend.basis_gates)
    return qv_circuit_whole


def adder_circuit(n):
    adder_circuit_whole = CDKMRippleCarryAdder(n)
    print(adder_circuit_whole.draw(output='text'))
    return adder_circuit_whole
    # operand1 = QuantumRegister(3, 'o1')
    # operand2 = QuantumRegister(3, 'o2')
    # anc = QuantumRegister(2, 'a')
    # cr = ClassicalRegister(4)
    # circuit = QuantumCircuit(operand1, operand2, anc, cr)
    # print(circuit.draw(output='text'))
    # return circuit


def mult_circuit(n):
    # operand1 = QuantumRegister(3, 'o1')
    # operand2 = QuantumRegister(3, 'o2')
    # anc = QuantumRegister(6, 'p')
    # cr = ClassicalRegister(6)
    # circuit = QuantumCircuit(operand1, operand2, anc, cr)
    # return circuit
    return RGQFTMultiplier(3)


def weight_circuit(n):
    #n should be 3
    return WeightedAdder(n)