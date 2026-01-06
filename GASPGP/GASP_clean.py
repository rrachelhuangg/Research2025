"""
In-progress implementation of the GASP experiment/algorithm from:
https://www.nature.com/articles/s41598-023-37767-w.
This file handles the generational experiments. 
"""

import random
import numpy as np
from qiskit import QuantumCircuit
from GASP_steps import run_circuit, select_gene, create_individual, create_population, individual_to_circuit, calculate_fitness, crossover, mutate
from direct_angle_optimizer import optimize_angles

if __name__ == '__main__':
    population = create_population()
    for individual in population: 
        print("Individual: \n", individual)
        print("Optimized Individual: \n", optimize_angles(individual))
