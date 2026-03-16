import itertools
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import CDKMRippleCarryAdder
from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit_ibm_runtime import QiskitRuntimeService


def generate_n_inputs():
    #enumerate the full state-space for all 3-bit numbers (2 operands)
    vals = ['000', '001', '010', '011', '100', '101', '110', '111']
    product_iterator = itertools.product(vals, vals)
    combinations_list = list(product_iterator)
    print("LIST: ", combinations_list)
    return combinations_list


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


def pair_up(inputs, circuits):
    pairs = []
    for i in range(len(inputs)):
        pairs += [(inputs[i], circuits[i])]
    print("PAIRS ELEMENT: ", pairs[5][0], pairs[5][1].draw(output='text'))
    return pairs


inputs = generate_n_inputs()
circuits = create_circuits(inputs)
pair_up(inputs, circuits)


# first_option = generate_n_inputs()[1]
# print("FIRST OPTION: ", first_option)


# operand1 = QuantumRegister(3, 'o1')
# operand2 = QuantumRegister(3, 'o2')
# anc = QuantumRegister(2, 'a')
# cr = ClassicalRegister(4)

# circ = QuantumCircuit(operand1, operand2, anc, cr)

# modded_circuit = operands_to_gates(first_option[0], first_option[1], circ, operand1, operand2)
# print("CIRCUIT: ", modded_circuit.draw(output='text'))


# circ = modded_circuit
# adder = CDKMRippleCarryAdder(3, 'full', 'Full Adder')
# circ.append(adder, [anc[0]] + operand1[0:3] + operand2[0:3] + [anc[1]])

# circ.measure(operand2[0:3] + [anc[1]], cr)
# print(circ.draw(output='text'))

# service = QiskitRuntimeService()
# backend = service.backend("ibm_torino")
# simulator = AerSimulator.from_backend(backend)
# tr_circ = transpile(circ, simulator)
# result = simulator.run(tr_circ).result()
# counts = result.get_counts()
# print("COUNTS: ", counts)
# print("MOST LIKELY RESULT: ", max(counts, key=counts.get))