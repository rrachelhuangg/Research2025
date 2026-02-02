"""
Evaluation criteria of a population.
Includes calculate_fitness from GASP_steps (primary eval criteria right now).
"""

import heapq
from GASP_steps import calculate_fitness, individual_to_circuit


def selected_subset(population, select_num):
    """
    I: population in gene format
    O: list of individuals in circuit format
    Only considers a circuit's fitness (calculated by the calculate_fitness function) right now.
    """
    circuit_tags = {}
    evaluations = {}
    selected_circuits = []

    i = 0
    for individual in population:
        circuit = individual_to_circuit(individual)
        circuit_tags[i] = circuit
        evaluations[i] = calculate_fitness(circuit)
        i += 1

    selected_circuit_idxs = heapq.nlargest(select_num, evaluations, key=evaluations.get)
    for idx in selected_circuit_idxs:
        selected_circuits.append(circuit_tags[idx])
    return selected_circuits
