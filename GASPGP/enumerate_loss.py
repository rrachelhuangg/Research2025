import random
import itertools
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import CDKMRippleCarryAdder
from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit_ibm_runtime import QiskitRuntimeService


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


def operands_to_gates(operand_1, operand_2, circuit, operand1, operand2):
    match operand_1:
        case '000':
            pass
        case '001':
            circuit.x([operand1[2]])
        case '010':
            circuit.x([operand1[1]])
        case '011':
            circuit.x([operand1[1], operand1[2]])
        case '100':
            circuit.x([operand1[0]])
        case '101':
            circuit.x([operand1[0], operand1[2]])
        case '110':
            circuit.x([operand1[0], operand1[1]])
        case '111':
            circuit.x([operand1[0], operand1[1], operand1[2]])
    match operand_2:
        case '000':
            pass
        case '001':
            circuit.x([operand2[2]])
        case '010':
            circuit.x([operand2[1]])
        case '011':
            circuit.x([operand2[1], operand2[2]])
        case '100':
            circuit.x([operand2[0]])
        case '101':
            circuit.x([operand2[0], operand2[2]])
        case '110':
            circuit.x([operand2[0], operand2[1]])
        case '111':
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


def pair_up():
    inputs = [('100', '011'), ('111', '111'), ('001', '000'), ('011', '100'), ('111', '100'), ('000', '011'), ('010', '000'), ('001', '101'), ('001', '001'), ('100', '110')]
    circuits = create_circuits(inputs)
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

    service = QiskitRuntimeService()
    backend = service.backend("ibm_torino")
    simulator = AerSimulator.from_backend(backend)
    tr_circ = transpile(circuit, simulator)
    result = simulator.run(tr_circ).result()
    counts = result.get_counts()
    maximum = max(counts, key=counts.get)
    return maximum
