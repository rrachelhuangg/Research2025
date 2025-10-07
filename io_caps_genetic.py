from qiskit import transpile, QuantumCircuit, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import Sampler
from generate_random_circuit import generate_random, apply_zx_calc
import pyzx as zx
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.qasm2 import dumps
import json
import random
from mutate_circuit import mutate_circuit

gen_n = 1
circuit_data = {}
generation_data_file = f"genetic_init_population/generation_{gen_n}.json"
survival_rate = 0.5 #want half of each generation's population to make it to the next generation

def create_init_population():
    """
    input: '0'*n
    output: '1' in third to last qubit
    """
    accurate_circuit_count = 0
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
        init_pop_dir = "genetic_init_population/"
        circuit_file = f"circuit_{i}.txt"
        qasm_txt_file = f"circuit_txt_{i}.txt"
        qasm_file = f"circuit_{i}.qasm"
        if accuracy == 1.0: #just for debugging purposes
            accurate_circuit_count += 1
        circuit_data[circuit_file] = accuracy
        with open(f"{init_pop_dir}{circuit_file}", "a") as file:
            file.write(str(untranspiled_circuit))
            file.write("\n")
            file.write(f"Measurement Results: {result}\n")
            file.write(f"Accuracy: {correct/total}\n")
            file.write("\n" + "---------------------------")
            # print("RANDOM CIRCUIT: ", untranspiled_circuit)
            # print(f"Measurement Results: {result}\n")
            # print(f"Accuracy: {correct/total}\n")

            with open(f"{init_pop_dir}{qasm_txt_file}", "w") as f:
                qasm_circuit = str(dumps(circuit))
                f.write(qasm_circuit)
            with open(f"{init_pop_dir}{qasm_txt_file}", "r") as f:
                lines = f.read().split('\n')
                format_lines = ""
                for l in lines:
                    if 'creg' not in l and 'measure' not in l and 'barrier' not in l:
                        format_lines += (l+'\n')
            with open(f"{init_pop_dir}{qasm_file}", "w") as f:
                qasm_circuit = QuantumCircuit.from_qasm_str(format_lines)
                formatted = dumps(qasm_circuit)
                f.write(formatted)

            #for testing/specific circuit preservation on hardware

            # opt_level = 0
            # layout_method = ''
            # routing_method = ''
            # translation_method = ''
            # print('Circuit stats: ', circuit.count_ops())
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
            # print("CIRCUIT: ", circuit)
    print("NUMBER OF CIRCUITS WITH ACCURACY 1.0: ", accurate_circuit_count)
    with open(generation_data_file, "w") as file:
        json.dump(circuit_data, file, indent=4)

def select_next_generation(curr_generation_data_file):
    data = None
    total = 0
    count = 0
    with open(curr_generation_data_file, "r") as file:
        data = json.load(file)
    for key in data:
        count += 1
        total += data[key]
    print("COUNT: ", count)
    if count == 0:
        return "Termination"
    average = total/count
    half_n = int(count*survival_rate)
    curr = 0
    selected_individuals = []
    keys = list(data.keys())
    while curr < half_n:
        check = random.choice(keys)
        if data[check] >= average:
            selected_individuals += [check]
            curr += 1
    return selected_individuals

def run_generation(selected_individuals):
    global gen_n
    circuit_data = {}
    gen_n += 1
    generation_data_file = f"genetic_init_population/generation_{gen_n}.json"
    for individual in selected_individuals:
        name = individual[:individual.find(".txt")]
        qasm_file = "genetic_init_population/" + name + ".qasm"
        qasm_circuit = zx.Circuit.load(qasm_file)
        graph = qasm_circuit.to_graph(compress_rows=True)
        circuit = zx.extract_circuit(graph.copy())
        qasm_intermediary = circuit.to_qasm()
        desired_circuit = QuantumCircuit.from_qasm_str(qasm_intermediary)
        mutated_circuit = mutate_circuit(desired_circuit)
        mutated_circuit.add_register(ClassicalRegister(3))
        mutated_circuit.measure(2,2)
        simulator = AerSimulator(method="matrix_product_state")
        transpiled = transpile(mutated_circuit, simulator)
        result = simulator.run(transpiled, shots=1000).result().get_counts()
        correct, total = 0, 0
        for key in result.keys():
            if key[len(key)-3]=='1':
                correct += 1
            total += 1
        accuracy = correct/total
        circuit_data[individual] = accuracy
    with open(generation_data_file, "w") as file:
        json.dump(circuit_data, file, indent=4)
    return generation_data_file
        


if __name__=='__main__':
    create_init_population()
    selected_individuals = select_next_generation(generation_data_file)
    for i in range(10):
        curr_gen_data_file = run_generation(selected_individuals)
        selected_individuals = select_next_generation(curr_gen_data_file)
        print("SELECTED INDIVIDUALS: ", selected_individuals)
        if selected_individuals == "Termination":
            break

    



