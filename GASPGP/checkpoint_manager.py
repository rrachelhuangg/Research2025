"""
Checkpoint management for GASP experiments.
Allows saving and loading experiment state for resuming runs.
"""

import os
import json
import random
import pickle
from datetime import datetime
from GASP_steps import individual_to_circuit


def save_checkpoint(checkpoint_path, state_dict):
    """
    Save experiment state to a checkpoint file.

    Args:
        checkpoint_path: Path to save the checkpoint file
        state_dict: Dictionary containing all state to save
    """
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    with open(checkpoint_path, 'wb') as f:
        pickle.dump(state_dict, f)

    summary_path = checkpoint_path.replace('.pkl', '_summary.json')
    summary = {
        'generation': state_dict['generation'],
        'max_fitness': state_dict['max_fitness_overall'],
        'average_fitness': state_dict['average_fitness_overall'],
        'iterations_since_improvement': state_dict['iterations_since_improvement'],
        'population_size': len(state_dict['population']),
        'timestamp': datetime.now().isoformat()
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)


def load_checkpoint(checkpoint_path):
    """
    Load experiment state from a checkpoint file.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Dictionary containing saved state, or None if file doesn't exist
    """
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return None

    with open(checkpoint_path, 'rb') as f:
        state_dict = pickle.load(f)

    print(f"✓ Checkpoint loaded from {checkpoint_path}")
    print(f"  Resuming from generation {state_dict['generation']}")
    print(f"  Max fitness: {state_dict['max_fitness_overall']:.6f}")
    print(f"  Average fitness: {state_dict['average_fitness_overall']:.6f}")
    print(f"  Population size: {len(state_dict['population'])}")

    return state_dict


def get_checkpoint_path(experiment_name, circuit_depth):
    """
    Generate a standard checkpoint path for an experiment.

    Args:
        experiment_name: Name of the experiment
        circuit_depth: Circuit depth parameter

    Returns:
        Path to checkpoint file
    """
    checkpoint_dir = "checkpoints"
    filename = f"{experiment_name}_depth{circuit_depth}.pkl"
    return os.path.join(checkpoint_dir, filename)


def save_circuits_to_text(checkpoint_path, population, num_circuits, individual_to_circuit_func):
    """
    Save a random sample of circuits from the population to a text file.

    Args:
        checkpoint_path: Path to the checkpoint file (used to generate txt filename)
        population: List of individuals in gene format
        num_circuits: Number of circuits to randomly sample and save
        individual_to_circuit_func: Function to convert individual to circuit
    """
    txt_path = checkpoint_path.replace('.pkl', '_circuits.txt')

    num_to_sample = min(num_circuits, len(population))
    sampled_individuals = random.sample(population, num_to_sample)

    with open(txt_path, 'w') as f:
        f.write(f"Random sample of {num_to_sample} circuits from population\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write("=" * 80 + "\n\n")

        for idx, individual in enumerate(sampled_individuals, 1):
            circuit = individual_to_circuit(individual)
            f.write(f"Circuit {idx}/{num_to_sample}:\n")
            f.write("-" * 80 + "\n")
            f.write(str(circuit.draw(output='text')))
            f.write("\n\n")

    return txt_path


def list_checkpoints(checkpoint_dir="checkpoints"):
    """
    List all available checkpoints with their summaries.

    Args:
        checkpoint_dir: Directory containing checkpoints
    """
    if not os.path.exists(checkpoint_dir):
        print(f"No checkpoint directory found at {checkpoint_dir}")
        return

    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pkl')]

    if not checkpoint_files:
        print("No checkpoints found")
        return

    print(f"\nAvailable checkpoints in {checkpoint_dir}:")
    print("-" * 80)

    for checkpoint_file in sorted(checkpoint_files):
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        summary_path = checkpoint_path.replace('.pkl', '_summary.json')

        print(f"\n{checkpoint_file}")

        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            print(f"  Generation: {summary['generation']}")
            print(f"  Max fitness: {summary['max_fitness']:.6f}")
            print(f"  Average fitness: {summary['average_fitness']:.6f}")
            print(f"  Population size: {summary['population_size']}")
            print(f"  Timestamp: {summary['timestamp']}")
        else:
            print(f"  (No summary available)")

    print("-" * 80)
