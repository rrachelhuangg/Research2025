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
from collections import Counter
from helper import visualize

#gene = [q_ti, G_i, q_ci, theta]
gene_gates = ["R_X", "R_Y", "R_Z", "CNOT"]
#temp testing value
init_pop_size = 100000
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
    init_pop_circuits = []
    init_pop_genes = []
    for i in range(init_pop_size):
        individual = QuantumCircuit(n, n)
        individual_genes = []
        j = 0
        while j < n:
            g = gene_gates[random.randint(0, 3)]
            if g == "R_X":
                individual.rx(1.5, j)
                individual_genes += [g]
                j += 1
            elif g == "R_Y":
                individual.ry(1.5, j)
                individual_genes += [g]
                j += 1
            elif g == "R_Z":
                individual.rz(1.5, j)
                individual_genes += [g]
                j += 1
            elif g == "CNOT":
                if j+1 == n:
                    continue
                else:
                    individual.cx(j, j+1)
                    individual_genes += [g]
                    j += 2
        init_pop_circuits += [individual]
        init_pop_genes += [individual_genes]
    return init_pop_circuits, init_pop_genes

def create_circuits(population):
    """
    Turns the gene representations of population individuals into actual Qiskit circuits.
    """
    circuit_representations = []
    for circuit in population:
        individual = QuantumCircuit(n, n)
        for j in range(len(circuit)):
            if isinstance(circuit[j], list):
                for gate in circuit[j]:
                    if gate == "R_X":
                        individual.rx(1.5, j)
                    elif gate == "R_Y":
                        individual.ry(1.5, j)
                    elif gate == "R_Z":
                        individual.rz(1.5, j)
                    elif gate == "CNOT":
                        if j + 1 == n:
                            continue
                        else:
                            individual.cx(j, j+1)
            else:
                if circuit[j] == "R_X":
                    individual.rx(1.5, j)
                    j += 1
                elif circuit[j] == "R_Y":
                    individual.ry(1.5, j)
                    j += 1
                elif circuit[j] == "R_Z":
                    individual.rz(1.5, j)
                    j += 1
                elif circuit[j] == "CNOT":
                    if j+1 == n:
                        continue
                    else:
                        individual.cx(j, j+1)
                        j += 2
                        # j += 1
        circuit_representations += [individual]
    return circuit_representations, population


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

def one_point_crossover(parent_1, parent_2):
    """
    1-point crossover (50% crossover) 2-parent breeding.
    """
    half_1_idx = len(parent_1)//2
    half_2_idx = len(parent_2)//2
    child = parent_1[:half_1_idx] + parent_2[half_2_idx:]
    return child

def breed_population(population):
    """
    Apply the chosen crossover method to the entire population. Result population is the same size
    as the input population.
    """
    new_generation = []
    while len(new_generation) < len(population):
        parent_1_idx = random.randint(0, len(population)-1)
        if parent_1_idx + 1 < len(population):
            parent_2_idx = parent_1_idx+1
        else:
            parent_2_idx = parent_1_idx-1
        child = one_point_crossover(population[parent_1_idx], population[parent_2_idx])
        new_generation += [child]
    return new_generation

def additive_mutate(individual):
    """
    Mutates an individual by adding a gene randomly to it.
    """
    idx_to_mutate = random.randint(0, len(individual)-1)
    if idx_to_mutate != len(individual)-1:
        addition = gene_gates[random.randint(0, len(gene_gates)-1)]
    else:
        addition = gene_gates[random.randint(0, len(gene_gates)-2)]
    if isinstance(individual[idx_to_mutate], list):
        individual[idx_to_mutate].append(addition)
    else:
        individual[idx_to_mutate] = [individual[idx_to_mutate], addition]
    return individual

def removal_mutate(individual):
    """
    Mutates an individual by removing a gene randomly from it.
    """
    idx_to_mutate = random.randint(0, len(individual)-1)
    if isinstance(individual[idx_to_mutate], list):
        if len(individual[idx_to_mutate]) == 1:
            individual[idx_to_mutate]= []
        elif len(individual[idx_to_mutate]) > 1:
            individual[idx_to_mutate] = individual[idx_to_mutate][1:]
    else:
        individual[idx_to_mutate] = ""
    return individual

def mutate_individual(individual):
    """
    Mutates a random gene in the individual. Should probably make the number of mutated genes also variable.
    Currently replaces gates (may reduce number of gates when replacing a CNOT).
    """
    idx_to_mutate = random.randint(0, len(individual)-1)
    gate = individual[idx_to_mutate]
    if gate == "R_X" or gate == "R_Y" or gate == "R_Z":
        individual[idx_to_mutate] = gene_gates[random.randint(0, len(gene_gates)-2)]
    elif gate == "CNOT":
        individual[idx_to_mutate] = gene_gates[random.randint(0, len(gene_gates)-1)]
    return individual

def mutate_population(population):
    """
    The entire input population is mutated with probability mutation_rate.
    """
    mutation_rate = 0.05
    mutated_population = []
    n_mutate = int(len(population)*mutation_rate)
    idxs_mutate = []
    i = 0
    idx = 0
    while i < n_mutate:
        idx = random.randint(0, len(population)-1)
        if idx not in idxs_mutate:
            idxs_mutate += [idx]
            i += 1
    for i in range(len(population)):
        if idx in idxs_mutate:
            mutation_type = random.randint(0, 2)
            if mutation_type == 0:
                mutated_individual = mutate_individual(population[i])
            elif mutation_type == 1:
                mutated_individual = additive_mutate(population[i])
            elif mutation_type == 2:
                mutated_individual = removal_mutate(population[i])
            mutated_population += [mutated_individual]
        else:
            mutated_population += [population[i]]
    return mutated_population

def assign_fitness_weights(population, curr_genes):
    """
    Assigns a weight to each individual based on their fitness.
    """
    fitness_weights = []
    selected_genes = []
    average_fitness = 0
    for individual, fitness in population:
        average_fitness += fitness
    if len(population) == 0:
        return None, None, True
    else:
        average_fitness = average_fitness/len(population)
    if average_fitness == 0:
        average_fitness = 1
    i = 0
    for individual, fitness in population:
        if fitness >= 0 and fitness < 0.25*average_fitness:
            fitness_weights += [[individual, 1]]
            selected_genes += [curr_genes[i]]
        elif fitness >= 0.25*average_fitness and fitness < 0.5*average_fitness:
            fitness_weights += [[individual, 2]]
            selected_genes += [curr_genes[i]]
        elif fitness >= 0.5*average_fitness and fitness < 0.75*average_fitness:
            fitness_weights += [[individual, 3]]
            selected_genes += [curr_genes[i]]
        else:
            fitness_weights += [[individual, 4]]
            selected_genes += [curr_genes[i]]
        i += 1
    return fitness_weights, selected_genes, False

def roulette_wheel_selection(fitness_weights, genes):
    """
    Randomly selects individuals to exterminate based on their assigned fitness weight.
    """
    population = [x[0] for x in fitness_weights]
    fitnesses = [x[1] for x in fitness_weights]
    indexes = random.choices(range(len(population)), weights=fitnesses, k=int(0.75*len(fitness_weights)))
    selected_individuals = [population[i] for i in indexes]
    selected_genes = [genes[i] for i in indexes]
    return selected_individuals, selected_genes

if __name__=='__main__':
    W_state = get_circuit_state(create_W_state(n))
    init_population, init_pop_genes = generate_init_pop()
    for i in range(50):
        print(f"GENERATION {i}")
        bred = breed_population(init_pop_genes)
        mutated = mutate_population(bred)
        mutated_population, curr_genes = create_circuits(mutated)
        first_gen = run_generation(W_state, mutated_population)
        fitness_weights, selected_genes, halt = assign_fitness_weights(first_gen, curr_genes)
        if halt == True: #prob need to make this handling more robust
            break
        selected_individuals, init_pop_genes = roulette_wheel_selection(fitness_weights, selected_genes)
    second_entries = [sublist[1] for sublist in fitness_weights]
    counts = dict(Counter(second_entries))
    if 4 in counts:
        for ind in fitness_weights:
            if ind[1] == 4:
                visualize(ind[0])
    print("COUNTS: ", counts)