"""
Hyperparameter search for good parameters to the initial population randomization. (Clauded)
"""

import json
import os
import sys
import subprocess
import itertools
from datetime import datetime
from pathlib import Path
import numpy as np
from run_GASP_experiment import run_experiment

HYPERPARAMETER_GRID = {
    'circuit_depth': [3, 5, 7],
}

NUM_TRIALS = 3  # Number of trials per hyperparameter config for robustness

def run_hyperparameter_search():
    """
    Run hyperparameter search over the circuit depth parameter.
    Each configuration is tested multiple times for statistical robustness.
    """
    results = []

    print(f"Starting hyperparameter search at {datetime.now()}")
    print(f"Testing circuit depths: {HYPERPARAMETER_GRID['circuit_depth']}")
    print(f"Number of trials per config: {NUM_TRIALS}")
    print("=" * 80)

    for circuit_depth in HYPERPARAMETER_GRID['circuit_depth']:
        print(f"\n{'='*80}")
        print(f"Testing circuit_depth = {circuit_depth}")
        print(f"{'='*80}\n")

        trial_results = []

        for trial in range(NUM_TRIALS):
            print(f"\n--- Trial {trial + 1}/{NUM_TRIALS} for depth={circuit_depth} ---")

            try:
                result = run_experiment(circuit_depth=circuit_depth)
                trial_results.append(result)

                print(f"Trial {trial + 1} completed:")
                print(f"  - Final generation: {result['final_generation']}")
                print(f"  - Max fitness: {result['max_fitness']:.6f}")
                print(f"  - Average fitness: {result['average_fitness']:.6f}")

            except Exception as e:
                print(f"Error in trial {trial + 1}: {e}")
                trial_results.append({
                    'error': str(e),
                    'final_generation': 0,
                    'max_fitness': 0.0,
                    'average_fitness': 0.0,
                    'iterations_since_improvement': 0,
                    'final_population_size': 0
                })

        # Aggregate results across trials
        successful_trials = [r for r in trial_results if 'error' not in r]

        if successful_trials:
            aggregated_result = {
                'circuit_depth': circuit_depth,
                'num_successful_trials': len(successful_trials),
                'avg_final_generation': np.mean([r['final_generation'] for r in successful_trials]),
                'std_final_generation': np.std([r['final_generation'] for r in successful_trials]),
                'avg_max_fitness': np.mean([r['max_fitness'] for r in successful_trials]),
                'std_max_fitness': np.std([r['max_fitness'] for r in successful_trials]),
                'avg_average_fitness': np.mean([r['average_fitness'] for r in successful_trials]),
                'std_average_fitness': np.std([r['average_fitness'] for r in successful_trials]),
                'avg_iterations_since_improvement': np.mean([r['iterations_since_improvement'] for r in successful_trials]),
                'raw_trials': trial_results
            }
        else:
            aggregated_result = {
                'circuit_depth': circuit_depth,
                'num_successful_trials': 0,
                'error': 'All trials failed',
                'raw_trials': trial_results
            }

        results.append(aggregated_result)

        print(f"\nAggregated results for depth={circuit_depth}:")
        if successful_trials:
            print(f"  - Successful trials: {len(successful_trials)}/{NUM_TRIALS}")
            print(f"  - Avg max fitness: {aggregated_result['avg_max_fitness']:.6f} ± {aggregated_result['std_max_fitness']:.6f}")
            print(f"  - Avg average fitness: {aggregated_result['avg_average_fitness']:.6f} ± {aggregated_result['std_average_fitness']:.6f}")
            print(f"  - Avg final generation: {aggregated_result['avg_final_generation']:.2f} ± {aggregated_result['std_final_generation']:.2f}")
        else:
            print(f"  - All trials failed")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"hyperparameter_searches/hyperparameter_results_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print("HYPERPARAMETER SEARCH COMPLETE")
    print(f"{'='*80}\n")
    print(f"Results saved to: {results_file}")

    # Find and report best configuration
    successful_results = [r for r in results if r['num_successful_trials'] > 0]

    if successful_results:
        # Sort by average max fitness (primary metric)
        best_by_max_fitness = max(successful_results, key=lambda x: x['avg_max_fitness'])
        # Sort by average average fitness (alternative metric)
        best_by_avg_fitness = max(successful_results, key=lambda x: x['avg_average_fitness'])
        # Sort by convergence speed (fewer generations is better)
        best_by_speed = min(successful_results, key=lambda x: x['avg_final_generation'])

        print("\nBEST CONFIGURATIONS:")
        print(f"\nBest by max fitness achieved:")
        print(f"  circuit_depth = {best_by_max_fitness['circuit_depth']}")
        print(f"  Avg max fitness: {best_by_max_fitness['avg_max_fitness']:.6f} ± {best_by_max_fitness['std_max_fitness']:.6f}")

        print(f"\nBest by average population fitness:")
        print(f"  circuit_depth = {best_by_avg_fitness['circuit_depth']}")
        print(f"  Avg average fitness: {best_by_avg_fitness['avg_average_fitness']:.6f} ± {best_by_avg_fitness['std_average_fitness']:.6f}")

        print(f"\nFastest convergence:")
        print(f"  circuit_depth = {best_by_speed['circuit_depth']}")
        print(f"  Avg generations: {best_by_speed['avg_final_generation']:.2f} ± {best_by_speed['std_final_generation']:.2f}")

        print("\n" + "="*80)
        print("\nRECOMMENDATION:")
        print(f"Use circuit_depth = {best_by_max_fitness['circuit_depth']} for best overall fitness performance")
        print("="*80)
    else:
        print("\nNo successful trials. Check experiment configuration.")

    return results


if __name__ == '__main__':
    run_hyperparameter_search()
