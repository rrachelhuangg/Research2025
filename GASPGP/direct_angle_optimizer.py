"""
Implementation of the GASP step that optimizes circuit angles.
Direct angle optimization performed on a parameterized Qiskit circuit.
"""

import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from GASP_steps import calculate_fitness


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
    def cost_function(angles):
        param_dict = {param: angle for param, angle in zip([p for p in param_list if p], angles) if param is not None}
        bound_circuit = circuit.assign_parameters(param_dict)
        fitness = calculate_fitness(bound_circuit)
        return -fitness #minimize negative fitness = maximize fitness
    initial_angles = [gene[3] if gene[1] != "CNOT" else 0 for gene in individual]
    result = minimize(
        cost_function,
        x0=initial_angles,
        bounds=[(0, 2*np.pi)] * len(initial_angles),
        method="COBYLA"
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
