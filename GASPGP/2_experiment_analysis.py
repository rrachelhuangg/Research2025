from experiment_result_top_gene_circuits import qv_no_zx, qv_yes_zx, adder_no_zx, adder_yes_zx
from GASP_steps import individual_to_circuit, target_state_circuit
from zx_helper import assign_zx_value
from qiskit.circuit.library import QuantumVolume
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler import generate_preset_pass_manager

target_circuit = target_state_circuit
print("TARGET CIRCUIT: ", target_circuit.draw(output='text'))

no_zx_circuit = individual_to_circuit(adder_no_zx)
yes_zx_circuit = individual_to_circuit(adder_yes_zx)

no_zx_circuit = individual_to_circuit(qv_no_zx)
print("QV NO ZX CIRCUIT: ", no_zx_circuit.draw(output='text'))

yes_zx_circuit = individual_to_circuit(qv_yes_zx)
print("QV YES ZX CIRCUIT: ", yes_zx_circuit.draw(output='text'))

#stuff to compare
#zx-calc-ness
print("NO ZX CIRCUIT ZX VALUE: ", assign_zx_value(no_zx_circuit))
print("YES ZX CIRCUIT ZX VALUE: ", assign_zx_value(yes_zx_circuit))
#length
print("NO ZX CIRCUIT LENGTH: ", len(no_zx_circuit))
print("NO ZX CIRCUIT GATE STATS:", no_zx_circuit.count_ops())
print("YES ZX CIRCUIT LENGTH: ", len(yes_zx_circuit))
print("YES ZX CIRCUIT GATE STATS: ", yes_zx_circuit.count_ops())
#speed
service = QiskitRuntimeService()
backend = service.backend("ibm_torino")
simulator = AerSimulator.from_backend(backend) #simulator modeled after the chosen hardware backend
transpiled_no_zx = transpile(no_zx_circuit, simulator)
print("NO ZX CIRCUIT RUN RESULTS: ", simulator.run(transpiled_no_zx, shots=1000).result())
transpiled_yes_zx = transpile(yes_zx_circuit, simulator)
print("YES ZX CIRCUIT RUN RESULTS: ", simulator.run(transpiled_yes_zx, shots=1000).result())
#functionality (how to use qv circuit)
#fitness (already calculated)


