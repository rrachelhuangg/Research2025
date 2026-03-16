from qiskit.circuit.library import QuantumVolume, CDKMRippleCarryAdder
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.random import random_circuit

service = QiskitRuntimeService()
backend = service.backend("ibm_torino")
pm = generate_preset_pass_manager(backend=backend, optimization_level=3)


def qv_circuit(n):
    qv_circuit_whole = QuantumVolume(num_qubits=n)
    # qv_circuit_transpiled = pm.run(qv_circuit_whole)
    # print(qv_circuit_transpiled.draw(output='text'))
    print(qv_circuit_whole.draw(output='text'))
    print(backend.basis_gates)
    return qv_circuit_whole


def adder_circuit(n):
    # adder_circuit_whole = CDKMRippleCarryAdder(n)
    # print(adder_circuit_whole.draw(output='text'))
    # return adder_circuit_whole
    operand1 = QuantumRegister(3, 'o1')
    operand2 = QuantumRegister(3, 'o2')
    anc = QuantumRegister(2, 'a')
    cr = ClassicalRegister(4)
    circuit = QuantumCircuit(operand1, operand2, anc, cr)
    # print(circuit.draw(output='text'))
    return circuit

def chem_circuit():
    driver = PySCFDriver(
        # atom='H .0 .0 .0; H .0 .0 0.735',
        atom='N 0.0 0.0 0.0; N 0.0 0.0 1.097',
        unit=DistanceUnit.ANGSTROM,
        basis='sto3g',
    )
    problem = driver.run()
    mapper = ParityMapper(num_particles=problem.num_particles)
    ansatz = UCCSD(
        problem.num_spatial_orbitals,
        problem.num_particles,
        mapper,
        initial_state=HartreeFock(
            problem.num_spatial_orbitals,
            problem.num_particles,
            mapper,
        ),
    )
    # print(ansatz.draw(output='text'))
    print(ansatz.num_parameters)
    return ansatz
