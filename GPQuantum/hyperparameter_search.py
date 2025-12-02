"""
Hyperparameter search for GASP experiments.
Runs multiple experiments with different parameter combinations to find optimal parameters.
"""

#number of crossover points also as a hyperparameter?

import json
import os
import sys
import subprocess
import itertools
from datetime import datetime
from pathlib import Path
import numpy as np

HYPERPARAMETER_GRID = {
    'init_pop_size': [1000, 5000, 10000],
    'n': [5, 6, 7],
    'mutation_rate': [0.05, 0.10, 0.15],
    'survival_rate': [0.20, 0.30, 0.50],
    'num_generations': [20, 30, 40]
}

RANDOM_SEARCH_TRIALS = 50

def run_gasp_experiment(params, experiment_id, results_dir):
    """
    Run a single GASP experiment with given parameters.
    """
    print(f"Running Experiment {experiment_id}")
    print(f"Parameters: {params}")

    exp_dir = Path(results_dir) / f"exp_{experiment_id}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    with open(exp_dir / "parameters.json", 'w') as f:
        json.dump(params, f, indent=2)

    cmd = [
        'python', 'parameterized_GASP.py',
        '--init_pop_size', str(params['init_pop_size']),
        '--n', str(params['n']),
        '--mutation_rate', str(params['mutation_rate']),
        '--survival_rate', str(params['survival_rate']),
        '--num_generations', str(params['num_generations']),
        '--output_dir', str(exp_dir)
    ]

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        stdout_lines = []
        stderr_lines = []

        for line in process.stdout:
            stdout_lines.append(line)

        process.wait(timeout=120)
        stderr_output = process.stderr.read()

        stdout_output = ''.join(stdout_lines)

        with open(exp_dir / "stdout.txt", 'w') as f:
            f.write(stdout_output)
        with open(exp_dir / "stderr.txt", 'w') as f:
            f.write(stderr_output)

        metrics = parse_metrics(stdout_output, exp_dir)
        metrics['success'] = process.returncode == 0

    except subprocess.TimeoutExpired:
        print(f"Experiment {experiment_id} timed out!")
        metrics = {
            'success': False,
            'error': 'timeout',
            'average_fitness': [],
            'final_fitness': None,
            'best_fitness': 0,
            'num_high_accuracy': 0,
            'final_counts': {}
        }
    except Exception as e:
        print(f"Experiment {experiment_id} failed: {e}")
        metrics = {
            'success': False,
            'error': str(e),
            'average_fitness': [],
            'final_fitness': None,
            'best_fitness': 0,
            'num_high_accuracy': 0,
            'final_counts': {}
        }

    result_data = {
        'experiment_id': experiment_id,
        'params': params,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }

    with open(exp_dir / "results.json", 'w') as f:
        json.dump(result_data, f, indent=2)

    return result_data


def parse_metrics(stdout, exp_dir):
    """
    Parse metrics from experiment output.
    """
    metrics = {
        'average_fitness': [],
        'final_fitness': None,
        'best_fitness': 0,
        'num_high_accuracy': 0,
        'final_counts': {}
    }

    for line in stdout.split('\n'):
        if 'AVERAGE FITNESS:' in line:
            try:
                fitness = float(line.split('AVERAGE FITNESS:')[1].strip())
                metrics['average_fitness'].append(fitness)
            except:
                pass
        elif 'COUNTS:' in line:
            try:
                counts_str = line.split('COUNTS:')[1].strip()
                metrics['final_counts'] = eval(counts_str)
                if 4 in metrics['final_counts']:
                    metrics['num_high_accuracy'] = metrics['final_counts'][4]
            except:
                pass

    if metrics['average_fitness']:
        metrics['final_fitness'] = metrics['average_fitness'][-1]
        metrics['best_fitness'] = max(metrics['average_fitness'])
        print("AVERAGE FITNESS: ", np.mean(metrics['average_fitness']))

        if len(metrics['average_fitness']) > 1:
            metrics['improvement_rate'] = (
                metrics['average_fitness'][-1] - metrics['average_fitness'][0]
            ) / len(metrics['average_fitness'])
        else:
            metrics['improvement_rate'] = 0

    return metrics


def grid_search(results_dir='hyperparameter_results'):
    """
    Perform grid search over all hyperparameter combinations.
    """
    print("Starting Grid Search...")
    print(f"Total combinations: {np.prod([len(v) for v in HYPERPARAMETER_GRID.values()])}")

    results_dir = Path(results_dir) / f"grid_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    experiment_id = 1

    keys = HYPERPARAMETER_GRID.keys()
    values = HYPERPARAMETER_GRID.values()

    for combination in itertools.product(*values):
        params = dict(zip(keys, combination))
        result = run_gasp_experiment(params, experiment_id, results_dir)
        all_results.append(result)
        experiment_id += 1

        save_summary(all_results, results_dir)

    print("Grid Search Complete!")
    print(f"Results saved to: {results_dir}")
    print_best_results(all_results)

    return all_results


def random_search(n_trials=RANDOM_SEARCH_TRIALS, results_dir='hyperparameter_results'):
    """
    Perform random search by sampling hyperparameters randomly.
    More efficient than grid search for large parameter spaces.
    """
    print(f"Starting Random Search with {n_trials} trials...")

    results_dir = Path(results_dir) / f"random_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for experiment_id in range(1, n_trials + 1):
        params = {
            'init_pop_size': np.random.choice(HYPERPARAMETER_GRID['init_pop_size']),
            'n': np.random.choice(HYPERPARAMETER_GRID['n']),
            'mutation_rate': np.random.choice(HYPERPARAMETER_GRID['mutation_rate']),
            'survival_rate': np.random.choice(HYPERPARAMETER_GRID['survival_rate']),
            'num_generations': np.random.choice(HYPERPARAMETER_GRID['num_generations'])
        }

        result = run_gasp_experiment(params, experiment_id, results_dir)
        all_results.append(result)

        save_summary(all_results, results_dir)

    print(f"\n{'='*60}")
    print("Random Search Complete!")
    print(f"Results saved to: {results_dir}")
    print_best_results(all_results)

    return all_results


def save_summary(results, results_dir):
    """
    Save summary of all results to JSON file.
    """
    summary_file = Path(results_dir) / "summary.json"
    max_fitness = 0
    best_exp = None
    for exp in results:
        best_fitness = exp["metrics"].get("best_fitness", 0)
        if best_fitness > max_fitness:
            max_fitness = best_fitness
            best_exp = exp
    print("BEST EXPERIMENT: ", best_exp)
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)


def print_best_results(results, top_n=5):
    """
    Print the best performing parameter combinations.
    """
    successful = [r for r in results if r['metrics'].get('success', False)]

    if not successful:
        print("No successful experiments!")
        return

    print(f"\nTop {top_n} Results by Final Fitness:")
    print("-" * 60)

    sorted_results = sorted(
        successful,
        key=lambda x: x['metrics'].get('final_fitness', 0),
        reverse=True
    )

    for i, result in enumerate(sorted_results[:top_n], 1):
        print(f"\n{i}. Experiment {result['experiment_id']}")
        print(f"   Parameters: {result['params']}")
        print(f"   Final Fitness: {result['metrics'].get('final_fitness', 'N/A'):.4f}")
        print(f"   Best Fitness: {result['metrics'].get('best_fitness', 'N/A'):.4f}")
        print(f"   High Accuracy Circuits: {result['metrics'].get('num_high_accuracy', 0)}")

    print("\n" + "-" * 60)
    print(f"\nTop {top_n} Results by Number of High Accuracy Circuits:")
    print("-" * 60)

    sorted_results = sorted(
        successful,
        key=lambda x: x['metrics'].get('num_high_accuracy', 0),
        reverse=True
    )

    for i, result in enumerate(sorted_results[:top_n], 1):
        print(f"\n{i}. Experiment {result['experiment_id']}")
        print(f"   Parameters: {result['params']}")
        print(f"   High Accuracy Circuits: {result['metrics'].get('num_high_accuracy', 0)}")
        print(f"   Final Fitness: {result['metrics'].get('final_fitness', 'N/A'):.4f}")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'grid':
        results = grid_search()
    elif len(sys.argv) > 1 and sys.argv[1] == 'random':
        n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else RANDOM_SEARCH_TRIALS
        results = random_search(n_trials=n_trials)
