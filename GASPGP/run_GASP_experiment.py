"""
In-progress implementation of the GASP experiment/algorithm from:
https://www.nature.com/articles/s41598-023-37767-w.
This file handles the generational experiments.
"""

import argparse
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from GASP_steps import run_circuit, select_gene, create_individual, create_population, individual_to_circuit, calculate_fitness, crossover, mutate, circuit_to_individual, roulette_wheel_select_single, roulette_wheel_selection, breed_to_minimum
from direct_angle_optimizer import optimize_angles
from population_evals import selected_subset
from checkpoint_manager import load_checkpoint, save_checkpoint, get_checkpoint_path, save_circuits_to_text

def run_experiment(circuit_depth=3, checkpoint_path=None, save_every=10, experiment_name="gasp_experiment", num_circuits_to_save=100):
    init_pop_size = 1000
    n = 3
    mutation_rate = 0.5
    survival_rate = 0.75
    desired_fitness = 0.75
    maxiter = 500
    minimum_pop_size = 500

    # Load from checkpoint if provided
    if checkpoint_path:
        state = load_checkpoint(checkpoint_path)
        if state:
            population = state['population']
            iterations_since_improvement = state['iterations_since_improvement']
            max_fitness_overall = state['max_fitness_overall']
            average_fitness_overall = state['average_fitness_overall']
            generation = state['generation']
            # Restore visualization data if available
            gen_indices = state.get('gen_indices', [])
            avg_fitness_vals = state.get('avg_fitness_vals', [])
            avg_angle_opt_times = state.get('avg_angle_opt_times', [])
            avg_zx = state.get('avg_zx', [])
            avg_len = state.get('avg_len', [])
            # Restore circuit_depth from checkpoint
            checkpoint_circuit_depth = state.get('circuit_depth', circuit_depth)
            if checkpoint_circuit_depth != circuit_depth:
                print(f"⚠ Warning: Checkpoint was created with circuit_depth={checkpoint_circuit_depth}, but {circuit_depth} was specified.")
                print(f"  Using circuit_depth={checkpoint_circuit_depth} from checkpoint.")
                circuit_depth = checkpoint_circuit_depth
            print(f"✓ Resuming experiment from generation {generation}")
        else:
            print("Failed to load checkpoint, starting fresh experiment")
            checkpoint_path = None

    # Initialize new experiment if no checkpoint loaded
    if not checkpoint_path or not state:
        population = create_population(init_pop_size, depth=circuit_depth)
        iterations_since_improvement = 0
        max_fitness_overall = 0
        average_fitness_overall = 0
        generation = 0
        # For visualizations
        gen_indices = []
        avg_fitness_vals = []
        avg_angle_opt_times = []
        avg_zx = []
        avg_len = []

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
        pop_zx = []
        pop_len = []
        for individual in mutated_population:
            start_time = time.time()
            optimized_individual, fit, zx, length = optimize_angles(individual)
            pop_zx += [zx]
            pop_len += [length]
            end_time = time.time()
            angle_opt_times += [end_time-start_time]
            optimized_population.append(optimized_individual)
        #recording stats for visualizations
        avg_angle_opt_times += [sum(angle_opt_times)/len(angle_opt_times)]
        avg_zx += [sum(pop_zx)/len(pop_zx)]
        avg_len += [sum(pop_len)/len(pop_len)]
        
        print("ROULETTING")
        population = roulette_wheel_selection(optimized_population, survival_rate)
        population = breed_to_minimum(population, minimum_pop_size)
        print(f"Selected {len(population)} individuals for next generation.")
        print()

        #generating visualizations
        if generation % 10 == 0:
            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8,8))
            axs[0, 0].plot(gen_indices, avg_fitness_vals)
            axs[0, 0].set_xlabel('Generation #')
            axs[0, 0].set_ylabel('Average Fitness')
            axs[0, 0].set_title('Average Fitness per Generation')
            axs[0, 0].grid(True)
            if len(gen_indices) == len(avg_angle_opt_times)+1:
                gen_indices = gen_indices[:-1]
            axs[1, 0].plot(gen_indices, avg_angle_opt_times)
            axs[1, 0].set_xlabel('Generation #')
            axs[1, 0].set_ylabel('Average Optimization Time')
            axs[1, 0].set_title('Average Optimization Time per Generation')
            axs[1, 0].grid(True)
            axs[0, 1].plot(gen_indices, avg_len)
            axs[0, 1].set_xlabel('Generation #')
            axs[0, 1].set_ylabel('Average Length')
            axs[0, 1].set_title('Average Length per Generation')
            axs[0, 1].grid(True)
            axs[1, 1].plot(gen_indices, avg_zx)
            axs[1, 1].set_xlabel('Generation #')
            axs[1, 1].set_ylabel('Average ZX-Calcness')
            axs[1, 1].set_title('Average ZX-Calcness per Generation')
            axs[1, 1].grid(True)
            plt.tight_layout()
            plt.savefig(f'visualizations/{experiment_name}.png', dpi=300, bbox_inches='tight')

        # Save checkpoint periodically
        if generation % save_every == 0:
            checkpoint_file = get_checkpoint_path(experiment_name, circuit_depth)
            state_dict = {
                'generation': generation,
                'population': population,
                'max_fitness_overall': max_fitness_overall,
                'average_fitness_overall': average_fitness_overall,
                'iterations_since_improvement': iterations_since_improvement,
                'gen_indices': gen_indices,
                'avg_fitness_vals': avg_fitness_vals,
                'avg_angle_opt_times': avg_angle_opt_times,
                'avg_zx': avg_zx,
                'avg_len': avg_len,
                'circuit_depth': circuit_depth
            }
            save_checkpoint(checkpoint_file, state_dict)
            print(f"✓ Checkpoint saved to {checkpoint_file}")

            # Save sample circuits to text file
            txt_path = save_circuits_to_text(checkpoint_file, population, num_circuits_to_save, individual_to_circuit)
            print(f"✓ Sample circuits saved to {txt_path}")

    selected_individuals = selected_subset(population, minimum_pop_size)
    for individual in selected_individuals:
        print(individual.draw(output='text'))
    print("SELECTED INDIVIDUALS: ", selected_individuals)

    # Save final checkpoint
    checkpoint_file = get_checkpoint_path(experiment_name, circuit_depth)
    state_dict = {
        'generation': generation,
        'population': population,
        'max_fitness_overall': max_fitness_overall,
        'average_fitness_overall': average_fitness_overall,
        'iterations_since_improvement': iterations_since_improvement,
        'gen_indices': gen_indices,
        'avg_fitness_vals': avg_fitness_vals,
        'avg_angle_opt_times': avg_angle_opt_times,
        'avg_zx': avg_zx,
        'avg_len': avg_len,
        'circuit_depth': circuit_depth
    }
    save_checkpoint(checkpoint_file, state_dict)
    print(f"✓ Final checkpoint saved to {checkpoint_file}")

    # Save sample circuits to text file
    txt_path = save_circuits_to_text(checkpoint_file, population, num_circuits_to_save, individual_to_circuit)
    print(f"✓ Final sample circuits saved to {txt_path}")

    print(f"Experiment complete!")

    return {
        'final_generation': generation,
        'max_fitness': max_fitness_overall,
        'average_fitness': average_fitness_overall,
        'iterations_since_improvement': iterations_since_improvement,
        'final_population_size': len(population)
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--circuit-depth',
        type=int,
        default=3,
        help='Depth of random circuits in initial population'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint file to resume from'
    )
    parser.add_argument(
        '--save-every',
        type=int,
        default=10,
        help='Save checkpoint every N generations'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='gasp_experiment',
        help='Name for the experiment (used in checkpoint filenames)'
    )
    parser.add_argument(
        '--num-circuits-to-save',
        type=int,
        default=100,
        help='Number of random circuits to save to text file at each checkpoint'
    )
    args = parser.parse_args()

    run_experiment(
        circuit_depth=args.circuit_depth,
        checkpoint_path=args.checkpoint,
        save_every=args.save_every,
        experiment_name=args.experiment_name,
        num_circuits_to_save=args.num_circuits_to_save
    )
