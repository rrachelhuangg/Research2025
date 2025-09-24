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
        # circuit = transpile(circuit, simulator)
        #hardware
        # circuit = apply_zx_calc(circuit, 3)
        # circuit.measure_all()
        
        #local
        # result = simulator.run(circuit, shots=1000).result().get_counts()
        untranspiled_circuit_transpiled = transpile(circuit, simulator)
        result = simulator.run(untranspiled_circuit_transpiled, shots=1000).result().get_counts()
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
            #for testing/specific circuit preservation on hardware
            #zx-calculus doesn't seem to help? (went from a potential 5 correct states to 4 correct states)
            # with open(f"qasm_accurate.txt", "w") as f:
            #     qasm_circuit = str(dumps(circuit))
            #     f.write(qasm_circuit)
            # with open(f"qasm_accurate.txt", "r") as f:
            #     lines = f.read().split('\n')
            #     format_lines = ""
            #     for l in lines:
            #         if 'creg' not in l and 'measure' not in l and 'barrier' not in l:
            #             format_lines += (l+'\n')
            # qasm_file_name = f"qasm_accurate.qasm"
            # with open(qasm_file_name, "w") as f:
            #     qasm_circuit = QuantumCircuit.from_qasm_str(format_lines)
            #     formatted = dumps(qasm_circuit)
            #     f.write(formatted)
            # loaded_circuit = zx.Circuit.load(qasm_file_name)

            # graph = loaded_circuit.to_graph(compress_rows=True)
            # print("ZX-Calculus Reduction Steps:")
            # print("----------------------------")
            # zx.full_reduce(graph, quiet=False)
            # print("\n")
            # graph.normalize()
            # optimized_circuit = zx.extract_circuit(graph.copy())
            # opt_qasm = optimized_circuit.to_qasm()
            # circuit = QuantumCircuit.from_qasm_str(opt_qasm)
            # circuit.add_register(ClassicalRegister(3))
            # circuit.measure(range(3), range(3))

            opt_level = 0
            layout_method = ''
            routing_method = ''
            translation_method = ''
            print('Circuit stats: ', circuit.count_ops())
            service = QiskitRuntimeService()
            backend = service.backend("ibm_brisbane")
            sampler = Sampler(mode=backend)
            pass_manager = generate_preset_pass_manager(
                optimization_level=opt_level, backend=backend, layout_method=layout_method, routing_method=routing_method, translation_method=translation_method
            )
            transpiled = pass_manager.run(circuit)
            job = sampler.run([(transpiled,)])
            result = job.result()[0].join_data().get_counts()
            print("RESULT: ", result)
            print("CIRCUIT: ", circuit)
            break

        i += 1

io_caps_teleport()
