import pyzx as zx
import sys
import time
from qiskit import QuantumCircuit, transpile, ClassicalRegister
from qiskit_aer import Aer
from qiskit.circuit.library import QFT
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.qasm2 import dumps
from datetime import datetime


def apply_zx_calc(circuit, n: int=3):
    #convert to qasm intermediary format
    with open(f"hardware_qft/qft_{n}.txt", "w") as f:
        qasm_circuit = str(dumps(circuit))
        f.write(qasm_circuit)
    with open(f"hardware_qft/qft_{n}.txt", "r") as f:
        lines = f.read().split('\n')
        format_lines = ""
        for l in lines:
            if 'creg' not in l and 'measure' not in l and 'barrier' not in l:
                format_lines += (l+'\n')
    qasm_file_name = f"hardware_qft/qft_{n}.qasm"
    with open(qasm_file_name, "w") as f:
        qasm_circuit = QuantumCircuit.from_qasm_str(format_lines)
        formatted = dumps(qasm_circuit)
        f.write(formatted)
    #apply zx calculus
    zx_qasm_circuit = zx.Circuit.load(qasm_file_name)
    graph = zx_qasm_circuit.to_graph(compress_rows=True)
    print("ZX-Calculus Reduction Steps:")
    print("----------------------------")
    zx.full_reduce(graph, quiet=False)
    print("\n")
    graph.normalize()
    optimized_circuit = zx.extract_circuit(graph.copy())
    g = zx_qasm_circuit
    p = optimized_circuit
    opt_qasm = optimized_circuit.to_qasm()
    opt_circuit = QuantumCircuit.from_qasm_str(opt_qasm)
    stats_circuit('Pre ZX-calculus optimized stats: ', g)
    stats_circuit('Post ZX-calculus optimized stats: ', p)
    # print("optimization validation check: ", zx.compare_tensors(g, p))
    # print("optimization validation check: ", zx_qasm_circuit.verify_equality(optimized_circuit, up_to_swaps=False, up_to_global_phase=True))
    return opt_circuit


def output_circuit(message, circuit):
    print(message)
    print(circuit.draw(output='text'))
    print('\n')


def stats_circuit(message, circuit):
    print(message)
    print(circuit.to_basic_gates().stats())
    print('\n')


def controller(n_qubits:int=3, hardware: str='s', zx_opt:str='False'):
    circuit = QuantumCircuit(n_qubits, n_qubits)
    circuit.x(n_qubits-1) #n qubit state where the n-1th qubit is a 1
    qft_gate = QFT(num_qubits=n_qubits, inverse=False)
    circuit.append(qft_gate, range(n_qubits))
    inverse_qft_gate = QFT(num_qubits=n_qubits, inverse=True)
    circuit.append(inverse_qft_gate, range(n_qubits))
    circuit.measure(range(n_qubits), range(n_qubits))

    result, start_time, end_time=None, None, None
    if hardware=='r':
        message='Pre ZX-calculus optimized circuit: '
        if zx_opt=='True':
            circuit = apply_zx_calc(circuit, n_qubits)
            circuit.add_register(ClassicalRegister(n))
            circuit.measure(range(n), range(n))
            message='Post ZX-calculus optimized circuit: '
        print('Circuit stats: ', circuit.count_ops())
        output_circuit(message, circuit)
        service = QiskitRuntimeService()
        backend = service.backend("ibm_brisbane")
        sampler = Sampler(mode=backend)
        pass_manager = generate_preset_pass_manager(
            optimization_level=1, backend=backend
        )
        transpiled = pass_manager.run(circuit)
        print('Transpiled circuit stats: ', transpiled.count_ops())
        job = sampler.run([(transpiled,)])
        result = job.result()[0].join_data().get_counts()
        start_time = datetime.strptime(str(job.result().metadata['execution']['execution_spans'].start), '%Y-%m-%d %H:%M:%S.%f')
        end_time = datetime.strptime(str(job.result().metadata['execution']['execution_spans'].stop), '%Y-%m-%d %H:%M:%S.%f')
    elif hardware=='s':
        message='Pre ZX-calculus optimized circuit: '
        if zx_opt=='True':
            circuit = apply_zx_calc(circuit, n_qubits)
            circuit.add_register(ClassicalRegister(n))
            circuit.measure(range(n), range(n))
            message='Post ZX-calculus optimized circuit: '
        print("Circuit stats: ", circuit.count_ops())
        output_circuit(message, circuit)
        backend = Aer.get_backend('qasm_simulator')
        prepped = transpile(circuit, backend)
        print("Tranpiled circuit stats: ", prepped.count_ops())
        start_time = time.time()
        job = backend.run(prepped)
        end_time = time.time()
        result = dict(job.result().get_counts())
    print(f"Measurement Result: {max(result, key=result.get)}\n")
    print(f"Time taken: {end_time-start_time}\n")


if __name__=='__main__':
    n = int(sys.argv[1])
    hardware = sys.argv[2]
    zx_opt = sys.argv[3]
    if zx_opt == 'True':
        controller(n, hardware, 'True')
    else:
        controller(n, hardware, 'False')
