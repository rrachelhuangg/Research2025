"""
In-progress implementation of the GASP experiment/algorithm from:
https://www.nature.com/articles/s41598-023-37767-w.
This file handles the generational experiments. 
"""

import random
import numpy as np
from qiskit import QuantumCircuit
from GASP_steps import run_circuit, select_gene, create_individual, create_population, individual_to_circuit, calculate_fitness, crossover, mutate, circuit_to_individual, roulette_wheel_select_single, roulette_wheel_selection, breed_to_minimum
from direct_angle_optimizer import optimize_angles

def run_experiment():
    init_pop_size = 10000
    n = 6
    mutation_rate = 0.8
    survival_rate = 0.9
    desired_fitness = 0.1
    maxiter = 50

    population = create_population(init_pop_size)
    iterations_since_improvement = 0
    max_fitness_overall = 0
    generation = 0

    while iterations_since_improvement < maxiter and max_fitness_overall < desired_fitness:
        generation += 1
        print(f"Generation: {generation}")

        fitnesses = []
        for individual in population:
            circuit = individual_to_circuit(individual)
            fitness = calculate_fitness(circuit)
            fitnesses.append(fitness)
        
        max_fitness_gen = max(fitnesses)
        avg_fitness_gen = sum(fitnesses)/len(fitnesses)
        print(f"Max fitness: {max_fitness_gen:.6f}, Avg fitness: {avg_fitness_gen:.6f}")
        if max_fitness_gen > max_fitness_overall:
            max_fitness_overall = max_fitness_gen
            iterations_since_improvement = 0
            print(f"New best fitness achieved! {max_fitness_overall}")
        else:
            iterations_since_improvement += 1
        
        if max_fitness_overall >= desired_fitness:
            print(f"Target fitness of {desired_fitness} achieved!")
            break
        
        if iterations_since_improvement >= maxiter:
            print(f"Max iterations of {maxiter} since improvement reached.")
            break
            
        offspring = []
        for _ in range(len(population)//2):
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            child = crossover(parent1, parent2)
            offspring.append(child)
        
        mutated_population = []
        for individual in offspring:
            if random.random() < mutation_rate:
                mutated_population.append(mutate(individual))
            else:
                mutated_population.append(individual)
        
        optimized_population = []
        for individual in mutated_population:
            optimized_individual = optimize_angles(individual)
            optimized_population.append(optimized_individual)
        
        population = roulette_wheel_selection(optimized_population, survival_rate)
        population = breed_to_minimum(population, 100)
        print(f"Selected {len(population)} individuals for next generation.")
        print()
    
    best_individual = None
    best_fitness = 0
    for individual in population:
        circuit = individual_to_circuit(individual)
        fitness = calculate_fitness(circuit)
        if fitness > best_fitness:
            best_fitness = fitness
            best_individual = individual
    
    print(f"Experiment complete!")
    return best_individual, best_fitness


if __name__ == '__main__':
    best_individual, best_fitness = run_experiment()
    print(f"Best fitness achieved: {best_fitness:.6f}")
    if best_individual is not None:
        print("Best individual:\n")
        best_circuit = individual_to_circuit(best_individual)
        print(best_circuit.draw(output='text'))