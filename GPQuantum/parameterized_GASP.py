"""
In-progress implementation of the GASP experiment/algorithm from:
https://www.nature.com/articles/s41598-023-37767-w.
"""

import os
import copy
import random
import argparse
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
import numpy as np
from collections import Counter
from helper import visualize, apply_zx_calc
from pathlib import Path
from qiskit.qasm2 import dumps

#gene = [q_ti, G_i, q_ci, theta]
gene_gates = ["R_X", "R_Y", "R_Z", "CNOT"]

# Default hyperparameters (can be overridden by command-line arguments)
init_pop_size = 1000
n = 6
mutation_rate = 0.10
survival_rate = 0.50
num_generations = 10
output_dir = None 


def create_W_state(n):
    """
    Equal superposition of all states with one qubit in |1> and the rest in |0>.
    """
    q_reg = QuantumRegister(n, 'q')
    circuit = QuantumCircuit(q_reg)
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
        individual = QuantumCircuit(n)
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
        individual = QuantumCircuit(n)
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
    # simulator = AerSimulator()
    # circuit_copy = copy.deepcopy(circuit)
    # circuit_copy.measure_all()
    # result = simulator.run(circuit_copy, shots=1).result().get_counts()
    # return result
    return Statevector.from_instruction(circuit)

def calculate_fitness(W_state, individual):
    """
    Calculating the fitness of an individual:
    1) Calculate the inner product of the W_state and the individual's state vector
    2) Square the magnitude of this inner product
    """
    # state_a = Statevector.from_label(W_state)
    # state_b = Statevector.from_label(individual_state)
    # inner_product = state_a.inner(state_b)
    # fitness = abs(inner_product)**2

    # if len(individual)>0:
    #     fitness += (1/len(individual))
    #add a gamma 10^-6 error --> average fitness over population, as more gens happen, put a lower weight on
    #length. use accuracy fitness as an indicator for how much to weigh the length penalty. 
    #only focus on accuracy on the beginning, and only take length into account eventually. 
    #length only plays a role when you're already in a very high fitness category. could increase n to have 
    #more initial diversity anyways. 
    statevec_target, statevec_candidate = W_state, individual
    if not isinstance(W_state, Statevector):
        statevec_target = Statevector(W_state)
    if not isinstance(individual, Statevector):
        statevec_candidate = Statevector(individual)
    fitness = statevec_target.inner(statevec_candidate)
    return float(abs(fitness)**2)


def run_generation(W_state, generation):
    """
    Calculate the fitness for each individual in a generation.
    """
    evaluations = []
    # bitstring_b = next(iter(W_state.keys())).replace(" ", "")
    # for individual in generation:
    #     individual_state = get_circuit_state(individual)
    #     bitstring_a = next(iter(individual_state.keys())).replace(" ", "")
    #     evaluations += [[individual, calculate_fitness(bitstring_a, bitstring_b, individual).item()]]
    if isinstance(W_state, QuantumCircuit):
        W_sv = Statevector.from_instruction(W_state)
    elif isinstance(W_state, Statevector):
        W_sv = W_state
    else:
        W_sv = Statevector(W_state)

    for individual in generation:
        indiv_sv = Statevector.from_instruction(individual)
        fitness = calculate_fitness(W_sv, indiv_sv)
        evaluations.append([individual, fitness])
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

def mutate_population(population, mutation_rate_param):
    """
    The entire input population is mutated with probability mutation_rate.
    """
    mutated_population = []
    n_mutate = int(len(population)*mutation_rate_param)
    idxs_mutate = []
    i = 0
    idx = 0
    while i < n_mutate:
        idx = random.randint(0, len(population)-1)
        # if idx not in idxs_mutate:
        idxs_mutate += [idx]
        i += 1
    for i in range(len(population)):
        # if idx in idxs_mutate:
        if i in idxs_mutate:
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
    print("AVERAGE FITNESS: ", average_fitness)
    for individual, fitness in population:
        if fitness >= 0 and fitness < 0.25*average_fitness:
            fitness_weights += [[individual, 1]]
            selected_genes += [curr_genes[i]]
        elif fitness >= 0.25*average_fitness and fitness < 0.5*average_fitness:
            fitness_weights += [[individual, 2]]
            selected_genes += [curr_genes[i]]
        elif fitness >= 0.5*average_fitness and fitness < 0.9*average_fitness:
            fitness_weights += [[individual, 3]]
            selected_genes += [curr_genes[i]]
        else:
            fitness_weights += [[individual, 4]]
            selected_genes += [curr_genes[i]]
        i += 1
    return fitness_weights, selected_genes, False

def roulette_wheel_selection(fitness_weights, genes, survival_rate_param):
    """
    Randomly selects individuals to exterminate based on their assigned fitness weight.
    """
    #only try keeping 10%...(less diversity...mutations will be main factor for diversity)
    #keeping more, main factor for diversity is recombination
    #hyperparameter search: write a script to evaluate metrics (diversity, accuracy). mass run stuff
    #come up with a thesis, try to support/disprove it in feedback loops
    population = [x[0] for x in fitness_weights]
    fitnesses = [x[1] for x in fitness_weights]
    indexes = random.choices(range(len(population)), weights=fitnesses, k=int(survival_rate_param*len(fitness_weights)))
    selected_individuals = [population[i] for i in indexes]
    selected_genes = [genes[i] for i in indexes]
    return selected_individuals, selected_genes

def run_experiment(init_pop_size_param, n_param, mutation_rate_param, survival_rate_param, num_generations_param, output_dir_param=None):
    """
    Run a GASP experiment with the specified parameters.
    """
    global init_pop_size, n, mutation_rate, survival_rate, num_generations, output_dir

    init_pop_size = init_pop_size_param
    n = n_param
    mutation_rate = mutation_rate_param
    survival_rate = survival_rate_param
    num_generations = num_generations_param
    output_dir = output_dir_param

    print(f"Starting GASP experiment with parameters:")
    print(f"  Population size: {init_pop_size}")
    print(f"  n (qubits): {n}")
    print(f"  Mutation rate: {mutation_rate}")
    print(f"  Survival rate: {survival_rate}")
    print(f"  Generations: {num_generations}")
    print()

    W_state = get_circuit_state(create_W_state(n))
    init_population, init_pop_genes = generate_init_pop()
    prev_fitness_weights = None

    for i in range(num_generations):
        print(f"GENERATION {i}")
        bred = breed_population(init_pop_genes)
        mutated = mutate_population(bred, mutation_rate)
        mutated_population, curr_genes = create_circuits(mutated)
        first_gen = run_generation(W_state, mutated_population)
        fitness_weights, selected_genes, halt = assign_fitness_weights(first_gen, curr_genes)
        if halt == True:
            print(f"Halting at generation {i} - saving data from previous generation")
            fitness_weights = prev_fitness_weights
            break
        prev_fitness_weights = fitness_weights
        selected_individuals, init_pop_genes = roulette_wheel_selection(fitness_weights, selected_genes, survival_rate)

    if fitness_weights is None or len(fitness_weights) == 0:
        counts = {}
    else:
        second_entries = [sublist[1] for sublist in fitness_weights]
        counts = dict(Counter(second_entries))

    if output_dir_param:
        base_dir = Path(output_dir_param)
        base_dir.mkdir(parents=True, exist_ok=True)
        new_dir = base_dir
    else:
        base_dir = Path(f'target_W_{n}/')
        base_dir.mkdir(parents=True, exist_ok=True)
        experiment_n = sum(1 for _ in base_dir.rglob('*') if _.is_dir())
        new_dir = base_dir / f'experiment{experiment_n+1}'
        os.makedirs(new_dir, exist_ok=True)

    circuit_count = 1
    if 4 in counts:
        for ind in fitness_weights:
            if ind[1] == 4:
                file_path = new_dir / f'circuit_{circuit_count}.qasm'
                qasm_circuit = str(dumps(ind[0]))
                with open(file_path, "w") as f:
                    f.write(qasm_circuit)
                print("MEASURED CIRCUIT: ", get_circuit_state(ind[0]), "=> TARGET STATE:", W_state, ind[1])
                apply_zx_calc(ind[0], n)
                circuit_count += 1

    print("COUNTS: ", counts)
    return counts


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Run GASP experiment with specified hyperparameters')
    parser.add_argument('--init_pop_size', type=int, default=1000, help='Initial population size')
    parser.add_argument('--n', type=int, default=6, help='Number of qubits (determines W-state target)')
    parser.add_argument('--mutation_rate', type=float, default=0.10, help='Mutation rate (0.0 to 1.0)')
    parser.add_argument('--survival_rate', type=float, default=0.50, help='Survival rate (0.0 to 1.0)')
    parser.add_argument('--num_generations', type=int, default=10, help='Number of generations')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for results')

    args = parser.parse_args()

    run_experiment(
        args.init_pop_size,
        args.n,
        args.mutation_rate,
        args.survival_rate,
        args.num_generations,
        args.output_dir
    )