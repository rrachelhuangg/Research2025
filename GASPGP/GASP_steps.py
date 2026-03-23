"""
Implementation of the GASP experiment steps, where each step is a function (or two).
"""

import random
import numpy as np
import numpy.random as npr
from tqdm import tqdm
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler import generate_preset_pass_manager
from circuit_library import qv_circuit, adder_circuit, chem_circuit, mult_circuit
from qiskit.circuit.random import random_circuit
from enumerate_loss import add_operands, bits_to_val, pair_up, multiply_operands, pair_mult_up

#biological structure
gates = {0:"R_X", 1:"R_Y", 2:"R_Z", 3:"CNOT"}

#experiment parameters
n = 12

# target_state_circuit = qv_circuit(5)
# target_state_circuit = adder_circuit(5)
target_state_circuit = mult_circuit(5)
# params = list(target_state_circuit.parameters)
# target_state_circuit = target_state_circuit.assign_parameters({params[0]:0, params[1]:0, params[2]:0})
target_state_vector = Statevector(target_state_circuit)
# service = QiskitRuntimeService()
# backend = service.backend("ibm_torino") #pass manager configured for chosen QPU
# pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
# target_state_circuit = pm.run(target_state_circuit)


def run_circuit(circuit):
    """
    Simulate a run of the input quantum circuit.
    Note: Measures the circuit and thus collapses it.
    """
    circuit.measure(range(n), range(n))
    service = QiskitRuntimeService()
    backend = service.backend("ibm_torino")
    simulator = AerSimulator().from_backend(backend)
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

def create_random_individual(depth):
    """
    O: gene format
    """
    rando_circuit = random_circuit(n, max_operands=2, depth=depth)
    to_gene = circuit_to_individual(rando_circuit)
    return to_gene


def create_random_adder_individual(depth):
    operand1 = QuantumRegister(3, 'o1')
    operand2 = QuantumRegister(3, 'o2')
    anc = QuantumRegister(2, 'a')
    cr = ClassicalRegister(4)
    circuit = QuantumCircuit(operand1, operand2, anc, cr)
    random_part = random_circuit(8, max_operands=2, depth=depth)
    circuit.compose(random_part, inplace=True)
    to_gene = circuit_to_individual(circuit)
    return to_gene

def create_random_mult_individual(depth):
    operand1 = QuantumRegister(3, 'o1')
    operand2 = QuantumRegister(3, 'o2')
    anc = QuantumRegister(6, 'p')
    cr = ClassicalRegister(6)
    circuit = QuantumCircuit(operand1, operand2, anc, cr)
    random_part = random_circuit(8, max_operands=2, depth=depth)
    circuit.compose(random_part, inplace=True)
    to_gene = circuit_to_individual(circuit)
    return to_gene


def create_population(init_pop_size, depth):
    population = []
    print("Creating population")
    for i in tqdm(range(init_pop_size)):
        # population += [create_individual()]
        # population += [create_random_individual(depth)]
        # population += [create_random_adder_individual(depth)]
        population += [create_random_mult_individual(depth)]
    return population


def individual_to_circuit(individual):
    """
    I: gene format
    O: circuit format
    """
    # circuit = QuantumCircuit(n, n)
    # operand1 = QuantumRegister(3, 'o1')                                                                                                
    # operand2 = QuantumRegister(3, 'o2')                                                                                                
    # anc = QuantumRegister(2, 'a')                                                                                                      
    # cr = ClassicalRegister(4)      
    circuit = QuantumCircuit(n, n)
    operand1 = QuantumRegister(3, 'o1')                                                                                                
    operand2 = QuantumRegister(3, 'o2')                                                                                                
    anc = QuantumRegister(6, 'p')                                                                                                      
    cr = ClassicalRegister(6)                                                                                                
    circuit = QuantumCircuit(operand1, operand2, anc, cr)
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
        if gate.operation.name == "rx":
            gene[0] = individual.find_bit(gate.qubits[0]).index
            gene[1] = "R_X"
            gene[2] = None
            gene[3] = gate.operation.params[0]
        elif gate.operation.name == "ry":
            gene[0] = individual.find_bit(gate.qubits[0]).index
            gene[1] = "R_Y"
            gene[2] = None
            gene[3] = gate.operation.params[0]
        elif gate.operation.name == "rz":
            gene[0] = individual.find_bit(gate.qubits[0]).index
            gene[1] = "R_Z"
            gene[2] = None
            gene[3] = gate.operation.params[0]
        elif gate.operation.name == "cx":
            gene[0] = individual.find_bit(gate.qubits[1]).index
            gene[2] = individual.find_bit(gate.qubits[0]).index
            gene[1] = "CNOT"
            gene[3] = 0
        else:
            continue  # skip unsupported gates (e.g. h, x, s, t)
        gene_format += [gene]
    return gene_format


def calculate_fitness(circuit, target_circuit):
    """
    I: circuit (non-parameterized)
    O: float
    """
    individual_statevector = Statevector(circuit)
    if target_circuit == None:
        tsv = target_state_vector
    else:
        tsv = Statevector(target_circuit)
    inner_product = individual_statevector.inner(tsv)
    fitness = abs(inner_product)**2
    return fitness


def compare_measurements(circuit, target_circuit):
    circuit_result = add_operands(circuit)
    target_result = add_operands(target_circuit)
    bits_diff = abs(bits_to_val(circuit_result) - bits_to_val(target_result))
    return bits_diff/100


def compare_mult_measurements(circuit, target_circuit):
    circuit_result = multiply_operands(circuit)
    target_result = multiply_operands(target_circuit)
    bits_diff = abs(bits_to_val(circuit_result) - bits_to_val(target_result))
    return bits_diff/100


def calculate_mod_fitness(circuit):
    avg_statevector_fitness = 0
    avg_measurement_comparisons = 0
    guidance_pairs = pair_up()
    i = 0
    for pair in guidance_pairs:
        print("I: ", i)
        avg_statevector_fitness += calculate_fitness(circuit, pair[1])
        avg_measurement_comparisons += compare_measurements(circuit, pair[1])
        i += 1
    avg_statevector_fitness /= len(guidance_pairs)
    avg_measurement_comparisons /= len(guidance_pairs)
    return avg_statevector_fitness + avg_measurement_comparisons


def calculate_mult_mod_fitness(circuit):
    avg_statevector_fitness = 0
    avg_measurement_comparisons = 0
    guidance_pairs = pair_mult_up()
    i = 0
    for pair in guidance_pairs:
        print("I: ", i)
        avg_statevector_fitness += calculate_fitness(circuit, pair[1])
        avg_measurement_comparisons += compare_mult_measurements(circuit, pair[1])
        i += 1
    avg_statevector_fitness /= len(guidance_pairs)
    avg_measurement_comparisons /= len(guidance_pairs)
    return avg_statevector_fitness + avg_measurement_comparisons


fitness_cache = {}

def get_fitness(individual):
    key = tuple(tuple(gene) for gene in individual)
    if key not in fitness_cache:
        circuit = individual_to_circuit(individual)
        fitness_cache[key] = calculate_mod_fitness(circuit)
    return fitness_cache[key]

mult_fitness_cache = {}

def get_mult_fitness(individual):
    key = tuple(tuple(gene) for gene in individual)
    if key not in mult_fitness_cache:
        circuit = individual_to_circuit(individual)
        mult_fitness_cache[key] = calculate_mult_mod_fitness(circuit)
    return mult_fitness_cache[key]


def crossover(ind_1, ind_2):
    """
    (k=1)-point crossover as defined by the GASP algorithm. 
    I/O: individuals are in gene format
    """
    half_1_idx = random.randint(0, len(ind_1))
    half_2_idx = random.randint(0, len(ind_2))
    child = ind_1[:half_1_idx] + ind_2[half_2_idx:]
    return child


def mutate(individual):
    """
    I/O: individuals are in gene format
    """
    if len(individual) == 0:
        return individual
    idx = random.randint(0, len(individual)-1)
    mutation_type = random.randint(0, 3)
    if mutation_type == 0:
        return mutate_replace(individual, idx)
    elif mutation_type == 1:
        return mutate_insert(individual, idx)
    elif mutation_type == 2:
        return mutate_swap(individual, idx)
    elif mutation_type == 3:
        return mutate_delete(individual, idx)


def mutate_replace(individual, idx):
    """
    I/O: individuals are in gene format
    """
    new_gene = individual[idx]
    while new_gene == individual[idx]:
        new_gene = select_gene(random.randint(0, n-1))
    mutated_individual = individual[:idx] + [new_gene] + individual[idx+1:]
    return mutated_individual


def mutate_insert(individual, idx):
    """
    I/O: individuals are in gene format
    """
    new_gene = select_gene(random.randint(0, n-1))
    return individual[:idx] + [new_gene] + individual[idx:]


def mutate_swap(individual, idx):
    """
    I/O: individuals are in gene format
    """
    if len(individual) < 2:
        return individual 
    other_idx = random.choice([i for i in range(len(individual)) if i != idx])
    copied_individual = individual.copy()
    copied_individual[idx], copied_individual[other_idx] = copied_individual[other_idx], copied_individual[idx]
    return copied_individual


def mutate_delete(individual, idx):
    """
    I/O: individuals are in gene format
    """
    MIN_LEN=1
    if len(individual)<=MIN_LEN:
        return individual
    return individual[:idx] + individual[idx+1:]


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
    return selected_individual


def roulette_wheel_selection(population, survival_rate):
    """
    I/O: population with individuals in gene format
    """
    selected_individuals = []
    to_survive = int(len(population)*survival_rate)

    # max_fitness = sum([get_fitness(ind) for ind in population])
    # selection_probs = [get_fitness(ind)/max_fitness for ind in population]

    max_fitness = sum([get_mult_fitness(ind) for ind in population])
    selection_probs = [get_mult_fitness(ind)/max_fitness for ind in population]

    while len(selected_individuals) < to_survive:
        selected_individual = roulette_wheel_select_single(population, max_fitness, selection_probs)
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
