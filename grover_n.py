from qiskit import QuantumCircuit, transpile, ClassicalRegister
from qiskit_algorithms import AmplificationProblem, Grover
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import ZGate
from qiskit_aer import AerSimulator
from qiskit.primitives import BackendSampler
import time, sys
from qiskit.qasm2 import dumps
import pyzx as zx
import numpy as np
from datetime import datetime
from qiskit_aer import Aer
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

def apply_zx_calc(oracle, n: int=3,):
    with open(f"grover/{n}.qasm", "w") as f:
        basis_circuit = transpile(oracle)
        qasm_circuit = dumps(basis_circuit)
        f.write(qasm_circuit)
    zx_qasm_circuit = zx.Circuit.load(f"grover/{n}.qasm")
    graph = zx_qasm_circuit.to_graph(compress_rows=True)
    print("ZX-Calculus Reduction Steps:")
    print("----------------------------")
    zx.full_reduce(graph, quiet=False)
    print("\n")
    graph.normalize()
    optimized_circuit = zx.extract_circuit(graph.copy())
    #for post optimization validation
    g = zx_qasm_circuit
    p = optimized_circuit
    #converting back to qiskit circuit for testing and drawing
    opt_qasm = optimized_circuit.to_qasm()
    opt_circuit = QuantumCircuit.from_qasm_str(opt_qasm)
    stats_circuit('Pre ZX-calculus optimized stats: ', g)
    stats_circuit('Post ZX-calculus optimized stats: ', p)
    return opt_circuit

def output_circuit(message, circuit):
    print(message)
    print(circuit)
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

def controller(n: int=15, hardware: str='s', opt_method:str=''):
    result, start_time, end_time = None, None, None
    oracle = QuantumCircuit(n)
    mcz_gate = ZGate().control(num_ctrl_qubits=(n-1),ctrl_state='1'*(n-1))
    oracle.append(mcz_gate, list(range(n)))
    if hardware=='r':
        if opt_method=='':
            message=f'Unoptimized circuit: '
        else:
            message=f'Pre {opt_method} optimized circuit: '
        opt_level = 0
        layout_method = ''
        routing_method = ''
        translation_method = ''
        if opt_method=='ZX':
            oracle = apply_zx_calc(oracle, n)
            message='Post ZX-calculus optimized circuit: '
        elif opt_method=='plugin-combo':
            message = f'Post {opt_method} optimized circuit: '
            opt_level=3
            layout_method='sabre'
            routing_method='stochastic'
            translation_method='synthesis'
        problem = AmplificationProblem(oracle, is_good_state=['1'*n])
        oracle_viz = problem.grover_operator.oracle.decompose().draw(output='text')
        output_circuit(message, oracle_viz)
        service = QiskitRuntimeService()
        backend = service.backend("ibm_brisbane")
        sampler = Sampler(mode=backend)
        grover = Grover(sampler=sampler, iterations=1)
        circuit = grover.construct_circuit(problem)
        pass_manager = generate_preset_pass_manager(
            optimization_level=opt_level, backend=backend, layout_method=layout_method, routing_method=routing_method, translation_method=translation_method
        )
        if opt_method=='custom-pass':
            custom_stage_1, custom_stage_2 = custom_pass_stages(backend, 1)
            pass_manager.pre_layout = custom_stage_1
            pass_manager.scheduling = custom_stage_2
        transpiled = pass_manager.run(circuit)
        transpiled.add_register(ClassicalRegister(n))
        transpiled.measure(range(n), range(n))
        print('Transpiled circuit stats: ', transpiled.count_ops())
        # output_circuit(f'Transpiled {message}', transpiled)
        job = sampler.run([(transpiled,)])
        result = job.result()[0].join_data().get_counts()
        start_time = datetime.strptime(str(job.result().metadata['execution']['execution_spans'].start), '%Y-%m-%d %H:%M:%S.%f')
        end_time = datetime.strptime(str(job.result().metadata['execution']['execution_spans'].stop), '%Y-%m-%d %H:%M:%S.%f')
    if hardware=='s':
        message='Pre ZX-calculus optimized circuit: '
        if opt_method == 'ZX':
            oracle = apply_zx_calc(oracle, n)
            message='Post ZX-calculus optimized circuit: '
        problem = AmplificationProblem(oracle, is_good_state=['1'*n])
        oracle_viz = problem.grover_operator.oracle.decompose().draw(output='text')
        mps_backend = AerSimulator(method="matrix_product_state")
        sampler = BackendSampler(backend=mps_backend)
        grover = Grover(sampler=sampler)
        start_time = time.time()
        result = grover.amplify(problem).top_measurement
        end_time = time.time()
        output_circuit(message, oracle_viz)
    print(f"Measurement result: {result}")
    print(f"Time taken: {end_time-start_time}\n")

if __name__ == '__main__':
    n = int(sys.argv[1])
    hardware = sys.argv[2]
    if len(sys.argv)>3:
        opt_method = sys.argv[3]
        controller(n, hardware, opt_method)
    else:
        controller(n, hardware, '')


