from qiskit import transpile, QuantumCircuit, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import Sampler
from generate_random_circuit import generate_random, apply_zx_calc
import pyzx as zx
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.qasm2 import dumps

def io_caps_teleport():
    """
    input: '0'*n
    output: '1' in third to last qubit
    """
    for i in range(100):
        simulator = AerSimulator(method="matrix_product_state")
        circuit = generate_random(3)

        circuit.measure([0,1],[0,1])
        circuit.barrier()
        circuit.cx(1,2)
        circuit.cz(0,2)
        circuit.measure(2,2)

        #local simulator
        untranspiled_circuit = circuit.copy()
        circuit = transpile(circuit, simulator)
        #hardware
        # circuit = apply_zx_calc(circuit, 3)
        # circuit.measure_all()
        
        #local
        result = simulator.run(circuit, shots=1000).result().get_counts()
        correct, total = 0, 0

        #hardware
        # opt_level = 0
        # layout_method = ''
        # routing_method = ''
        # translation_method = ''
        # service = QiskitRuntimeService()
        # backend = service.backend("ibm_brisbane")
        # sampler = Sampler(mode=backend)
        # pass_manager = generate_preset_pass_manager(
        #     optimization_level=opt_level, backend=backend, layout_method=layout_method, routing_method=routing_method, translation_method=translation_method
        # )
        # transpiled = pass_manager.run(circuit)
        # job = sampler.run([(transpiled,)])
        # result = job.result()[0].join_data().get_counts()
        # print("RESULT: ", result)

        for key in result.keys():
            if key[len(key)-3]=='1':
                correct += 1
            total += 1
        
        accuracy = correct/total
        accurate_circuits_file = "local_accurate_circuits.txt"
        # if accuracy >= 0.8: #for hardware
        if accuracy == 1.0:
            with open(accurate_circuits_file, "a") as file:
                file.write(str(untranspiled_circuit))
                file.write("\n")
                file.write(f"Measurement Results: {result}\n")
                file.write(f"Accuracy: {correct/total}\n")
                file.write("\n" + "---------------------------")
                print("RANDOM CIRCUIT: ", untranspiled_circuit)
                print(f"Measurement Results: {result}\n")
                print(f"Accuracy: {correct/total}\n")

        i += 1

io_caps_teleport()