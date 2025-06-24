import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from math import gcd
import pandas as pd
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import QFT
from qiskit_aer import Aer, AerSimulator
import pyzx
from qiskit import qasm2

def initialize_qubits(circuit, n, m):
    #superposition with Hadamard gates => 2^n computational states
    circuit.h(range(n))
    circuit.x(n+m-1)

def modular_exponentiation(a, x):
    #a is a coprime of the target prime, and x is the number of times we want to exponentiate a
    #creates the circuit of the the function that we want to find the period of
    U = QuantumCircuit(4)
    for iter in range(x):
        if a in [2,13]:
            U.swap(0, 1)
            U.swap(1, 2)
            U.swap(2, 3)
        if a in [7,8]:
            U.swap(2, 3)
            U.swap(1, 2)
            U.swap(0, 1)
        if a == 11:
            U.swap(1,3)
            U.swap(0,2)
        if a in [7, 11, 13]:
            for q in range(4):
                U.x(q)
    U_viz = U.decompose().draw(output='text')
    with open(f"shors_15/modular_function.txt", "w") as file:
        file.write(str(U_viz))
    U = U.to_gate()
    c_U = U.control()
    return c_U

def modulate_circuit(circuit, n, m, a):
    for x in range(n):
        exponent = 2**x
        circuit.append(modular_exponentiation(a,exponent), [x]+list(range(n, n+m)))

def inverse_qft(circuit, measurement_qubits):
    #shor's uses qft to speed up the period finding process as compared to classic algorithms
    #applying inverse of qft because qft is applied to the superposition to encode the period info,
    #and then inverse qft transforms this information back into a measurable form
    #applies qft to input circuit, with results being stored in measurement_qubits
    circuit.append(QFT(len(measurement_qubits), do_swaps=False).inverse(), measurement_qubits)

def shors_alg(n, m, a):
    qc = QuantumCircuit(n+m, n)
    initialize_qubits(qc, n, m)
    qc.barrier() #barrier tells qiskit transpiler to not merge/optimize operations on either side of the barrier during circuit compilation
    modulate_circuit(qc, n, m, a)
    qc.barrier()
    inverse_qft(qc, range(n))
    qc.measure(range(n), range(n))
    qc_viz = qc.decompose().draw(output='text')
    with open(f"shors_15/algorithm.txt", "w") as file:
        file.write(str(qc_viz))
    return qc

if __name__ == '__main__':
    n = 4 #measurement qubits
    m = 4 #target qubits
    a = 7 #7 is the coprime of 15 that is used to test for this implementation of finding the factors of 15
    final_circuit = shors_alg(n, m, a)
    simulator = AerSimulator()
    compiled_circuit = transpile(final_circuit, simulator)
    counts = simulator.run(compiled_circuit).result().get_counts(final_circuit)
    plot_histogram(counts, filename='shors_15/histogram.png')
    for i in counts:
        measured_value = int(i[::-1], 2)
        if measured_value % 2 != 0:
            continue
        x = int((a**(measured_value/2))%15)
        if(x+1)%15 == 0:
            continue
        factors = gcd(x+1, 15), gcd(x-1, 15)
        print("FACTORS: ", factors)

