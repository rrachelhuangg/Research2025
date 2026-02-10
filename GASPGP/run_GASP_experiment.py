"""
In-progress implementation of the GASP experiment/algorithm from:
https://www.nature.com/articles/s41598-023-37767-w.
This file handles the generational experiments. 
"""

import time
import random
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from GASP_steps import run_circuit, select_gene, create_individual, create_population, individual_to_circuit, calculate_fitness, crossover, mutate, circuit_to_individual, roulette_wheel_select_single, roulette_wheel_selection, breed_to_minimum
from direct_angle_optimizer import optimize_angles
from population_evals import selected_subset

def run_experiment(circuit_depth=3):
    init_pop_size = 5000
    n = 3
    mutation_rate = 0.5
    survival_rate = 0.75
    desired_fitness = 0.75
    maxiter = 500
    minimum_pop_size = 500

    population = create_population(init_pop_size, depth=circuit_depth)
    iterations_since_improvement = 0
    max_fitness_overall = 0
    average_fitness_overall = 0
    generation = 0

    #for visualizations
    gen_indices = []
    avg_fitness_vals = []
    avg_angle_opt_times = []

    while iterations_since_improvement < maxiter and average_fitness_overall < desired_fitness:
    # while iterations_since_improvement < maxiter and max_fitness_overall < desired_fitness:
        generation += 1
        print(f"Generation: {generation}")

        fitnesses = []
        for individual in population:
            circuit = individual_to_circuit(individual)
            fitness = calculate_fitness(circuit)
            fitnesses.append(fitness)
        
        max_fitness_gen = max(fitnesses)
        avg_fitness_gen = sum(fitnesses)/len(fitnesses)

        #recording stats for visualizations
        gen_indices += [generation]
        avg_fitness_vals += [avg_fitness_gen]

        average_fitness_overall = avg_fitness_gen
        print(f"Max fitness: {max_fitness_gen:.6f}, Avg fitness: {avg_fitness_gen:.6f}")
        if max_fitness_gen > max_fitness_overall:
            max_fitness_overall = max_fitness_gen
            iterations_since_improvement = 0
            print(f"New best fitness achieved! {max_fitness_overall}")
        else:
            iterations_since_improvement += 1
        
        # if max_fitness_overall >= desired_fitness:
        if average_fitness_overall >= desired_fitness:
            print(f"Target fitness of {desired_fitness} achieved!")
            break
        
        if iterations_since_improvement >= maxiter:
            print(f"Max iterations of {maxiter} since improvement reached.")
            break
            
        offspring = []
        print("BREEDING")
        for _ in range(len(population)//2):
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            child = crossover(parent1, parent2)
            offspring.append(child)
        
        print("MUTATING")
        mutated_population = []
        for individual in offspring:
            if random.random() < mutation_rate:
                mutated_population.append(mutate(individual))
            else:
                mutated_population.append(individual)
        
        print("OPTIMIZING ANGLES")
        optimized_population = []
        angle_opt_times = []
        for individual in mutated_population:
            start_time = time.time()
            optimized_individual = optimize_angles(individual)
            end_time = time.time()
            angle_opt_times += [end_time-start_time]
            optimized_population.append(optimized_individual)
        avg_angle_opt_times += [sum(angle_opt_times)/len(angle_opt_times)]
        
        print("ROULETTING")
        population = roulette_wheel_selection(optimized_population, survival_rate)
        population = breed_to_minimum(population, minimum_pop_size)
        print(f"Selected {len(population)} individuals for next generation.")
        print()

        #generating visualizations
        if generation % 10 == 0:
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
            axs[0].plot(gen_indices, avg_fitness_vals)
            axs[0].set_xlabel('Generation #')
            axs[0].set_ylabel('Average Fitness')
            axs[0].set_title('Average Fitness per Generation')
            axs[0].grid(True)
            if len(gen_indices) == len(avg_angle_opt_times)+1:
                gen_indices = gen_indices[:-1]
            axs[1].plot(gen_indices, avg_angle_opt_times)
            axs[1].set_xlabel('Generation #')
            axs[1].set_ylabel('Average Optimization Time')
            axs[1].set_title('Average Optimization Time per Generation')
            axs[1].grid(True)
            plt.tight_layout()
            plt.savefig('visualizations/Experiment_Vizs_1.png', dpi=300, bbox_inches='tight')

    selected_individuals = selected_subset(population, minimum_pop_size)
    for individual in selected_individuals:
        print(individual.draw(output='text'))
    print("SELECTED INDIVIDUALS: ", selected_individuals)

    print(f"Experiment complete!")

    return {
        'final_generation': generation,
        'max_fitness': max_fitness_overall,
        'average_fitness': average_fitness_overall,
        'iterations_since_improvement': iterations_since_improvement,
        'final_population_size': len(population)
    }


if __name__ == '__main__':
    run_experiment()
