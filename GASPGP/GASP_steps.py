"""
Implementation of the GASP experiment steps, where each step is a function (or two).
"""

import random
import numpy as np
import numpy.random as npr
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler import generate_preset_pass_manager
from circuit_library import qv_circuit
from qiskit.circuit.random import random_circuit

#biological structure
gates = {0:"R_X", 1:"R_Y", 2:"R_Z", 3:"CNOT"}

#experiment parameters
n = 3

# target_state_circuit = QuantumCircuit(n,n)
# target_state_circuit.x(n-1)
# target_state_circuit.h(n-2)
# target_state_circuit.y(n-3)
target_state_circuit = qv_circuit(n)
target_state_vector = Statevector(target_state_circuit)
service = QiskitRuntimeService()
backend = service.backend("ibm_torino")
pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
target_state_circuit = pm.run(target_state_circuit)


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


def select_gene(i):
    gene = [None, None, None, None]
    gate = gates[random.randint(0,3)]
    if gate == "CNOT":
        control_bit = i
        while control_bit == i:
            control_bit = random.randint(0, n-1)
        gene[2] = control_bit
        gene[3] = 0
    else:
        gene[2] = None
        gene[3] = random.uniform(0, 2*np.pi)
    gene[0] = i
    gene[1] = gate
    return gene

def create_individual():
    """
    O: gene format
    """
    individual = []
    for i in range(n):
        gene = select_gene(i)
        individual += [gene]
    return individual

def create_random_individual():
    """
    O: gene format
    """
    rando_circuit = random_circuit(n, max_operands=2, depth=3)
    to_gene = circuit_to_individual(rando_circuit)
    return to_gene


def create_population(init_pop_size):
    population = []
    for i in range(init_pop_size):
        # population += [create_individual()]
        population += [create_random_individual()]
    return population


def individual_to_circuit(individual):
    """
    I: gene format
    O: circuit format
    """
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


def circuit_to_individual(individual):
    """
    I: circuit format
    O: gene format
    """
    gene_format = []
    for gate in individual.data:
        gene = [None, None, None, None]
        if gate.operation.name != "cx":
            gene[0] = gate.qubits[0]._index
            if gate.operation.name == "rx":
                gene[1] = "R_X"
            elif gate.operation.name == "ry":
                gene[1] = "R_Y"
            elif gate.operation.name == "rz":
                gene[1] = "R_Z"
            gene[2] = None
            gene[3] = gate.operation.params[0]
        elif gate.operation.name == "cx":
            gene[0] = gate.qubits[1]._index
            gene[2] = gate.qubits[0]._index
            gene[1] = "CNOT"
            gene[3] = 0
        gene_format += [gene]
    return gene_format


def calculate_fitness(circuit):
    """
    I: circuit (non-parameterized)
    O: float
    """
    individual_statevector = Statevector(circuit)
    inner_product = individual_statevector.inner(target_state_vector)
    fitness = abs(inner_product)**2
    return fitness


def crossover(ind_1, ind_2):
    """
    (k=1)-point crossover as defined by the GASP algorithm. 
    I/O: individuals are in gene format
    """
    half_1_idx = len(ind_1)//2
    half_2_idx = len(ind_2)//2
    child = ind_1[:half_1_idx] + ind_2[half_2_idx:]
    return child


def mutate(individual):
    """
    I/O: individuals are in gene format
    """
    idx = random.randint(0, n-1)
    new_gene = individual[idx]
    while new_gene == individual[idx]:
        new_gene = select_gene(idx)
    mutated_individual = individual[:idx] + [new_gene] + individual[idx+1:]
    return mutated_individual


def roulette_wheel_select_single(population, max_fitness, selection_probs):
    """
    I: population of individuals in circuit format
    O: selected individual in gene format
    Note: This function selects a single individual from the population. 
    Prioritizes a higher fitness.
    """
    if max_fitness == 0:
        selected_individual = random.choice(population)
    else:
        #selection_probs = [calculate_fitness(c)/max_fitness for c in population]
        selected_individual = population[npr.choice(len(population), p=selection_probs)]
    return circuit_to_individual(selected_individual)


def roulette_wheel_selection(population, survival_rate):
    """
    I/O: population with individuals in gene format
    """
    selected_individuals = []
    to_survive = int(len(population)*survival_rate)

    circuit_population = []
    for individual in population:
        circuit_population += [individual_to_circuit(individual)]

    max_fitness = sum([calculate_fitness(c) for c in circuit_population])
    selection_probs = [calculate_fitness(c)/max_fitness for c in circuit_population]

    while len(selected_individuals) < to_survive:
        selected_individual = roulette_wheel_select_single(circuit_population, max_fitness, selection_probs)
        selected_individuals += [selected_individual]
    
    return selected_individuals

def breed_to_minimum(population, target_size):
    """
    Prevent population from running to size 0.
    """
    if len(population) >= target_size:
        return population
    current_population = population.copy()
    while len(current_population) < target_size:
        parent1 = random.choice(population)
        parent2 = random.choice(population)
        child = crossover(parent1, parent2)
        current_population.append(child)
    return current_population
