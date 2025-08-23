from qiskit import QuantumCircuit, ClassicalRegister
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector
import numpy as np
import pyzx as zx
import time
from datetime import datetime
import sys
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import Sampler
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import QFT
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler import generate_preset_pass_manager, PassManager
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.qasm2 import dumps
from qiskit.circuit import library as lib
from qiskit.transpiler.passes import (
    ALAPScheduleAnalysis,
    InverseCancellation,
    PadDynamicalDecoupling,
)

def apply_zx_calc(circuit, n: int=3):
    #convert to qasm intermediary format
    with open(f"teleportation/{n}.txt", "w") as f:
        qasm_circuit = str(dumps(circuit))
        f.write(qasm_circuit)
    with open(f"teleportation/{n}.txt", "r") as f:
        lines = f.read().split('\n')
        format_lines = ""
        for l in lines:
            if 'creg' not in l and 'measure' not in l and 'barrier' not in l:
                format_lines += (l+'\n')
    qasm_file_name = f"teleportation/{n}.qasm"
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

def custom_pass_stages(backend, opt:int=1):
    if opt==1:
        dd_sequence = [lib.XGate(), lib.XGate()]
        scheduling_pm = PassManager(
            [
                ALAPScheduleAnalysis(target=backend.target),
                PadDynamicalDecoupling(target=backend.target, dd_sequence=dd_sequence),
            ]
        )
        inverse_gate_list = [
            lib.CXGate(),
            lib.HGate(),
            (lib.RXGate(np.pi / 4), lib.RXGate(-np.pi / 4)),
            (lib.PhaseGate(np.pi / 4), lib.PhaseGate(-np.pi / 4)),
            (lib.TGate(), lib.TdgGate()),
        ]
        logical_opt = PassManager([InverseCancellation(inverse_gate_list)])
        return logical_opt, scheduling_pm

def controller(n_qubits:int=3, hardware: str='s', opt_method:str=''):
    circuit = QuantumCircuit(n_qubits,n_qubits)
    circuit.x(0)
    circuit.barrier()
    #essentially entangling all of the qubits with each other
    circuit.h(1)
    circuit.cx(1,2)
    circuit.cx(0,1)
    circuit.h(0)
    circuit.barrier()
    #measure qubits 0 and 1 into classical bits 0 and 1
    circuit.measure([0,1],[0,1])
    #based on the classical bits measured/received, unitary x/z gates are applied to entangled qubit to get state idential to initial state
    circuit.barrier()
    circuit.cx(1,2)
    circuit.cz(0,2)

    result, start_time, end_time, correct, total=None, None, None, 0, 0
    if hardware=='r':
        message=f'Pre {opt_method} optimized circuit: '
        opt_level = 0
        layout_method = ''
        routing_method = ''
        translation_method = ''
        if opt_method=='ZX':
            circuit = apply_zx_calc(circuit, n_qubits)
            circuit.add_register(ClassicalRegister(n))
            circuit.measure(range(n), range(n))
            message='Post ZX-calculus optimized circuit: '
        elif opt_method=='plugin-combo':
            message = f'Post {opt_method} optimized circuit: '
            opt_level=3
            layout_method='sabre'
            routing_method='stochastic'
            translation_method='synthesis'
        print('Circuit stats: ', circuit.count_ops())
        output_circuit(message, circuit)
        service = QiskitRuntimeService()
        backend = service.backend("ibm_brisbane")
        sampler = Sampler(mode=backend)
        pass_manager = generate_preset_pass_manager(
            optimization_level=opt_level, backend=backend, layout_method=layout_method, routing_method=routing_method, translation_method=translation_method
        )
        if opt_method=='custom-pass':
            custom_stage_1, custom_stage_2 = custom_pass_stages(backend, 1)
            pass_manager.pre_layout = custom_stage_1
            pass_manager.scheduling = custom_stage_2
        transpiled = pass_manager.run(circuit)
        print('Transpiled circuit stats: ', transpiled.count_ops())
        # output_circuit(f'Transpiled {message}', transpiled)
        job = sampler.run([(transpiled,)])
        result = job.result()[0].join_data().get_counts()
        counts = result
        # start_time = datetime.strptime(str(job.result().metadata['execution']['execution_spans'].start), '%Y-%m-%d %H:%M:%S.%f')
        # end_time = datetime.strptime(str(job.result().metadata['execution']['execution_spans'].stop), '%Y-%m-%d %H:%M:%S.%f')
        #for qpu torino instead of brisbane
        # print(job.result().metadata['execution']['execution_spans']['__value__']['spans'][0].start)
        start_time = datetime.strptime(str(job.result().metadata['execution']['execution_spans']['__value__']['spans'][0].start), '%Y-%m-%d %H:%M:%S.%f')
        end_time = datetime.strptime(str(job.result().metadata['execution']['execution_spans']['__value__']['spans'][0].stop), '%Y-%m-%d %H:%M:%S.%f')
    elif hardware=='s':
        message='Pre ZX-calculus optimized circuit: '
        if opt_method=='ZX':
            circuit = apply_zx_calc(circuit, n_qubits)
            circuit.add_register(ClassicalRegister(n))
            circuit.measure(range(n), range(n))
            message='Post ZX-calculus optimized circuit: '
        print("Circuit stats: ", circuit.count_ops())
        output_circuit(message, circuit)
        simulator = AerSimulator(method="matrix_product_state")
        circuit.measure(2,2)
        start_time = time.time()
        result = simulator.run(circuit, shots=1000).result().get_counts()
        end_time = time.time()
    for key in result.keys():
        if key[len(key)-3]=='1':
            correct += 1
        total += 1
    print(f"Measurement Results: {result}\n")
    print(f"Accuracy: {correct/total}\n")
    print(f"Time taken: {end_time-start_time}\n")

if __name__=='__main__':
    n = int(sys.argv[1])
    hardware = sys.argv[2]
    if len(sys.argv)>3:
        opt_method = sys.argv[3]
        controller(n, hardware, opt_method)
    else:
        controller(n, hardware, '')
