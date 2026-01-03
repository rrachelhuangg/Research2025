"""
In-progress implementation of the GASP experiment/algorithm from:
https://www.nature.com/articles/s41598-023-37767-w.
"""

import random
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

#biological structure
gates = {0:"R_X", 1:"R_Y", 2:"R_Z", 3:"CNOT"}

#experiment parameters
n = 6
init_pop_size = 1000

target_state_circuit = QuantumCircuit(n,n)
target_state_circuit.x(n-1)
target_state_vector = Statevector(target_state_circuit)


def run_circuit(circuit):
    """
    Simulate a run of the input quantum circuit.
    Note: Measures the circuit and thus collapses it.
    """
    circuit.measure(range(n), range(n))
    simulator = AerSimulator()
    compiled_circuit = transpile(circuit, simulator)
    job = simulator.run(compiled_circuit, shots=1000)
    result = job.result()
    counts = result.get_counts(compiled_circuit)
    return counts


def create_individual():
    individual = []
    for i in range(n):
        gene = [None, None, None, None]
        gate = gates[random.randint(0,3)]
        if gate == "CNOT":
            control_bit = i
            while control_bit == i:
                control_bit = random.randint(0, n-1)
            gene[2] = control_bit
        else:
            gene[2] = None
        gene[0] = i
        gene[1] = gate
        gene[3] = 0
        individual += [gene]
    return individual


def create_population():
    population = []
    for i in range(init_pop_size):
        population += [create_individual()]
    return population


def individual_to_circuit(individual):
    circuit = QuantumCircuit(n, n)
    for gene in individual:
        if gene[1] == "R_X":
            circuit.rx(gene[3], gene[0])
        elif gene[1] == "R_Y":
            circuit.ry(gene[3], gene[0])
        elif gene[1] == "R_Z":
            circuit.rz(gene[3], gene[0])
        elif gene[1] == "CNOT":
            circuit.cx(gene[2], gene[0])
    return circuit


def calculate_fitness(circuit):
    individual_statevector = Statevector(circuit)
    inner_product = individual_statevector.inner(target_state_vector)
    fitness = abs(inner_product)**2
    return fitness


if __name__ == '__main__':
    population = create_population()
    for individual in population:
        circuit = individual_to_circuit(individual)
        calculate_fitness(circuit)