"""
In-progress implementation of the GASP experiment/algorithm from:
https://www.nature.com/articles/s41598-023-37767-w.
"""

import copy
import random
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
import numpy as np

#gene = [q_ti, G_i, q_ci, theta]
gene_gates = ["R_X", "R_Y", "R_Z", "CNOT"]
#temp testing value
init_pop_size = 10
#n determines the W-state (target) and size of individuals
n = 6 


def create_W_state(n):
    """
    Equal superposition of all states with one qubit in |1> and the rest in |0>.
    """
    q_reg = QuantumRegister(n, 'q')
    c_reg = ClassicalRegister(n, 'c')
    circuit = QuantumCircuit(q_reg, c_reg)
    circuit.x(q_reg[0])
    for i in range(n-1):
        theta = 2*np.arccos(np.sqrt((n-i-1)/(n-i)))
        circuit.ry(theta, q_reg[i+1])
        circuit.cx(q_reg[i+1], q_reg[i])
        circuit.x(q_reg[i+1])
    return circuit
    

def generate_init_pop():
    """
    Each individual P_i = [gene_1, gene_2, ..., gene_n]
    """
    init_pop = []
    for i in range(init_pop_size):
        individual = QuantumCircuit(n, n)
        j = 0
        while j < n:
            g = gene_gates[random.randint(0, 3)]
            if g == "R_X":
                individual.rx(1.5, j)
                j += 1
            elif g == "R_Y":
                individual.ry(1.5, j)
                j += 1
            elif g == "R_Z":
                individual.rz(1.5, j)
                j += 1
            elif g == "CNOT":
                if j+1 == n:
                    continue
                else:
                    individual.cx(j, j+1)
                    j += 2
        init_pop += [individual]
    return init_pop


def get_circuit_state(circuit):
    """
    Measure the circuit with 1 shot to get a single measurement for its state.
    """
    simulator = AerSimulator()
    circuit_copy = copy.deepcopy(circuit)
    circuit_copy.measure_all()
    result = simulator.run(circuit_copy, shots=1).result().get_counts()
    return result


def calculate_fitness(W_state, individual_state):
    """
    Calculating the fitness of an individual:
    1) Calculate the inner product of the W_state and the individual's state vector
    2) Square the magnitude of this inner product
    """
    state_a = Statevector.from_label(W_state)
    state_b = Statevector.from_label(individual_state)
    inner_product = state_a.inner(state_b)
    return abs(inner_product)**2


def run_generation(W_state, generation):
    """
    Calculate the fitness for each individual in a generation.
    """
    evaluations = []
    bitstring_b = next(iter(W_state.keys())).replace(" ", "")
    for individual in generation:
        individual_state = get_circuit_state(individual)
        bitstring_a = next(iter(individual_state.keys())).replace(" ", "")
        evaluations += [[individual, calculate_fitness(bitstring_a, bitstring_b).item()]]
    return evaluations


if __name__=='__main__':
    W_state = get_circuit_state(create_W_state(n))
    init_population = generate_init_pop()
    run_generation(W_state, init_population)
    