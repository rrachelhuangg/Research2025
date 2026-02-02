"""
Implementation of the GASP step that optimizes circuit angles.
Direct angle optimization performed on a parameterized Qiskit circuit.
"""

import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from GASP_steps import calculate_fitness
from zx_helper import assign_zx_value


def individual_to_parameterized_circuit(individual, n):
    created_circuit = QuantumCircuit(n, n)
    parameters = [] #angles
    for i, gene in enumerate(individual):
        if gene[1] in ["R_X", "R_Y", "R_Z"]:
            param = Parameter(f'θ_{i}')
            parameters.append(param)
            if gene[1] == "R_X":
                created_circuit.rx(param, gene[0])
            elif gene[1] == "R_Y":
                created_circuit.ry(param, gene[0])
            elif gene[1] == "R_Z":
                created_circuit.rz(param, gene[0])
        elif gene[1] == "CNOT":
            created_circuit.cx(gene[2], gene[0])
            parameters.append(None)
    
    return created_circuit, parameters


def optimize_angles(individual):
    circuit, param_list = individual_to_parameterized_circuit(individual, 6)
    def fitness_cost_function(angles):
        """
        Cost function v1: maximizes fitness of an individual
        """
        param_dict = {param: angle for param, angle in zip([p for p in param_list if p], angles) if param is not None}
        bound_circuit = circuit.assign_parameters(param_dict)
        fitness = calculate_fitness(bound_circuit)
        return -fitness #minimize negative fitness = maximize fitness
    def zx_cost_function(angles):
        """
        Cost function v2: maximizes "zx-calcness" of an individual
        """
        param_dict = {param: angle for param, angle in zip([p for p in param_list if p], angles) if param is not None}
        bound_circuit = circuit.assign_parameters(param_dict)
        zx_value = assign_zx_value(bound_circuit)
        return -zx_value #minimize negative zx-calcness = maximize zx-calcness
    def cost_function(angles):
        fitness_cost = fitness_cost_function(angles)
        zx_cost = zx_cost_function(angles)
        length_cost = len(individual)
        overall_cost = (0.4*fitness_cost) + (0.5*zx_cost) + (0.1*length_cost)
        return overall_cost
    initial_angles = [gene[3] for gene in individual if gene[1] != "CNOT"]
    if len(initial_angles) == 0:
        return individual
    result = minimize(
        cost_function,
        x0=initial_angles,
        bounds=[(0, 2*np.pi)] * len(initial_angles),
        method="SLSQP"
    )
    optimized_individual = []
    angle_idx = 0
    for gene in individual:
        new_gene = gene.copy()
        if gene[1] != "CNOT":
            new_gene[3] = result.x[angle_idx]
            angle_idx += 1
        optimized_individual += [new_gene]
    return optimized_individual