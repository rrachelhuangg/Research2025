import random
import itertools
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import CDKMRippleCarryAdder, RGQFTMultiplier, WeightedAdder
from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()
backend = service.backend("ibm_torino")
simulator = AerSimulator.from_backend(backend)


def bits_to_val(bits):
    return int(bits, 2)


def generate_n_inputs():
    """
    enumerate the full state-space for all 3-bit numbers (2 operands)
    """
    vals = ['000', '001', '010', '011', '100', '101', '110', '111']
    product_iterator = itertools.product(vals, vals)
    combinations_list = list(product_iterator)
    selected = random.sample(combinations_list, 10)
    return selected


def generate_weight_inputs():
    """
    enumerate a state-space with fixed weights [4,5,6] and random 3-bit input values
    """
    vals = ['000', '001', '010', '011', '100', '101', '110', '111']
    fixed_weights = [4, 5, 6]
    selected = random.sample(vals, 10)
    return [(val, fixed_weights) for val in selected]


def operands_to_gates(operand_1, operand_2, circuit, operand1, operand2):
    if operand_1 == '000':
        pass
    elif operand_1 == '001':
        circuit.x([operand1[0]])
    elif operand_1 == '010':
        circuit.x([operand1[1]])
    elif operand_1 == '011':
        circuit.x([operand1[0], operand1[1]])
    elif operand_1 == '100':
        circuit.x([operand1[2]])
    elif operand_1 == '101':
        circuit.x([operand1[0], operand1[2]])
    elif operand_1 == '110':
        circuit.x([operand1[1], operand1[2]])
    elif operand_1 == '111':
        circuit.x([operand1[0], operand1[1], operand1[2]])

    if operand_2 == '000':
        pass
    elif operand_2 == '001':
        circuit.x([operand2[0]])
    elif operand_2 == '010':
        circuit.x([operand2[1]])
    elif operand_2 == '011':
        circuit.x([operand2[0], operand2[1]])
    elif operand_2 == '100':
        circuit.x([operand2[2]])
    elif operand_2 == '101':
        circuit.x([operand2[0], operand2[2]])
    elif operand_2 == '110':
        circuit.x([operand2[1], operand2[2]])
    elif operand_2 == '111':
        circuit.x([operand2[0], operand2[1], operand2[2]])
    return circuit


def create_circuits(inputs):
    val_circuits = []
    for expression in inputs:
        operand1 = QuantumRegister(3, 'o1')
        operand2 = QuantumRegister(3, 'o2')
        anc = QuantumRegister(2, 'a')
        cr = ClassicalRegister(4)
        circ = QuantumCircuit(operand1, operand2, anc, cr)
        formed_circuit = operands_to_gates(expression[0], expression[1], circ, operand1, operand2)
        val_circuits += [formed_circuit]
    return val_circuits


def create_mult_circuits(inputs):
    val_circuits = []
    for expression in inputs:
        operand1 = QuantumRegister(3, 'o1')
        operand2 = QuantumRegister(3, 'o2')
        anc = QuantumRegister(6, 'p')
        cr = ClassicalRegister(6)
        circ = QuantumCircuit(operand1, operand2, anc, cr)
        formed_circuit = operands_to_gates(expression[0], expression[1], circ, operand1, operand2)
        val_circuits += [formed_circuit]
    return val_circuits


def create_weight_circuits(inputs):
    val_circuits = []
    for expression in inputs:
        bitstring = expression[0]
        adder = WeightedAdder(3, expression[1])
        circuit = QuantumCircuit(11, adder.num_sum_qubits)
        for i, bit in enumerate(bitstring):
            if bit == '1':
                circuit.x(i)
        circuit.append(adder, range(adder.num_qubits))
        val_circuits += [circuit]
    return val_circuits


def pair_up():
    inputs = generate_n_inputs()
    circuits = create_circuits(inputs)
    pairs = []
    for i in range(len(inputs)):
        pairs += [(inputs[i], circuits[i])]
    return pairs


def pair_mult_up():
    inputs = generate_n_inputs()
    circuits = create_mult_circuits(inputs)
    pairs = []
    for i in range(len(inputs)):
        pairs += [(inputs[i], circuits[i])]
    return pairs


def pair_weight_up():
    inputs = generate_weight_inputs()
    circuits = create_weight_circuits(inputs)
    pairs = []
    for i in range(len(inputs)):
        pairs += [(inputs[i], circuits[i])]
    return pairs


def add_operands(circuit):
    """
    output: returns most likely measured bitstring in little-endian format
    """
    circuit = circuit.copy()
    adder = CDKMRippleCarryAdder(3, 'full', 'Full Adder')

    operand1, operand2, anc = circuit.qregs
    creg = circuit.cregs[0]

    circuit.append(adder, [anc[0]] + list(operand1) + list(operand2) + [anc[1]])

    circuit.measure(list(operand2) + [anc[1]], creg)
    tr_circ = transpile(circuit, simulator)
    result = simulator.run(tr_circ).result()
    counts = result.get_counts()
    maximum = max(counts, key=counts.get)
    return maximum


def multiply_operands(circuit):
    """
    output: returns most likely measured bitstring in little-endian format
    """
    circuit = circuit.copy()
    multiplier = RGQFTMultiplier(3)

    operand1, operand2, anc = circuit.qregs
    creg = circuit.cregs[0]

    circuit.append(multiplier, operand1[:]+operand2[:]+anc[:])

    circuit.measure(anc, creg)
    tr_circ = transpile(circuit, simulator)
    result = simulator.run(tr_circ).result()
    counts = result.get_counts()
    maximum = max(counts, key=counts.get)
    return maximum


def weight_operands(circuit):
    """
    output: returns most likely measured bitstring in little-endian format
    """
    circuit = circuit.copy()
    num_sums = circuit.num_clbits
    sum_qubits = list(range(3, 3 + num_sums))
    circuit.measure(sum_qubits, range(num_sums))
    tr_circ = transpile(circuit, simulator)
    result = simulator.run(tr_circ).result()
    counts = result.get_counts()
    maximum = max(counts, key=counts.get)
    return maximum
