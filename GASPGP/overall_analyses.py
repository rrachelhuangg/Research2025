import ast
from experiment_result_top_gene_circuits import qv_no_zx, qv_yes_zx, adder_no_zx, adder_yes_zx
from GASP_steps import individual_to_circuit, target_state_circuit
from zx_helper import assign_zx_value
from qiskit.circuit.library import QuantumVolume
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler import generate_preset_pass_manager
from GASP_steps import calculate_fitness

no_zx_circuits = []
yes_zx_circuits = []

with open("checkpoints/qv-no-zx_depth_7_circuits.txt", "r") as f:
    circuits = f.read()
    no_zx_circuits = circuits.split("\n\n")[:-1]

with open("checkpoints/qv-yes-zx_depth_7_circuits.txt", "r") as f:
    circuits = f.read()
    yes_zx_circuits = circuits.split("\n\n")[:-1]

avg_no_fitness, avg_no_zxness, avg_no_len, avg_no_speed = 0, 0, 0, 0
avg_yes_fitness, avg_yes_zxness, avg_yes_len, avg_yes_speed = 0, 0, 0, 0
for i in range(100):
    print(f"Processing circuit{i}")
    no_zx_circuit = individual_to_circuit(ast.literal_eval(no_zx_circuits[i]))
    yes_zx_circuit = individual_to_circuit(ast.literal_eval(yes_zx_circuits[i]))
    avg_no_fitness += calculate_fitness(no_zx_circuit)
    avg_yes_fitness += calculate_fitness(yes_zx_circuit)
    avg_no_zxness += assign_zx_value(no_zx_circuit)
    avg_yes_zxness += assign_zx_value(yes_zx_circuit)
    avg_no_len += len(no_zx_circuit)
    avg_yes_len += len(yes_zx_circuit)

    service = QiskitRuntimeService()
    backend = service.backend("ibm_torino")
    simulator = AerSimulator.from_backend(backend)

    transpiled_no_zx = transpile(no_zx_circuit, simulator)
    no_zx_results = simulator.run(transpiled_no_zx, shots=1000).result()
    avg_no_speed += no_zx_results.time_taken
    transpiled_yes_zx = transpile(yes_zx_circuit, simulator)
    yes_zx_results = simulator.run(transpiled_yes_zx, shots=1000).result()
    avg_yes_speed += yes_zx_results.time_taken

avg_no_fitness /= 100
avg_no_zxness /= 100
avg_no_len /= 100
avg_no_speed /= 100

avg_yes_fitness /= 100
avg_yes_zxness /= 100
avg_yes_len /= 100
avg_yes_speed /= 100

print("NO ZX, AVERAGE FITNESS: ", avg_no_fitness)
print("NO ZX, AVERAGE ZXNESS: ", avg_no_zxness)
print("NO ZX, AVERAGE LENGTH: ", avg_no_len)
print("NO ZX, AVERAGE SPEED: ", avg_no_speed)

print("YES ZX, AVERAGE FITNESS: ", avg_yes_fitness)
print("YES ZX, AVERAGE ZXNESS: ", avg_yes_zxness)
print("YES ZX, AVERAGE LENGTH: ", avg_yes_len)
print("YES ZX, AVERAGE SPEED: ", avg_yes_speed)
