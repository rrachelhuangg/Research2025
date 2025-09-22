from qiskit import QuantumCircuit, ClassicalRegister
import numpy as np
import pyzx as zx
import time
from datetime import datetime
import sys
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import Sampler
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler import generate_preset_pass_manager, PassManager
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.qasm2 import dumps
from qiskit.circuit import library as lib
from qiskit.circuit.library import XXMinusYYGate, XXPlusYYGate, C3XGate, C4XGate, CCXGate, CCZGate, CSwapGate
import random

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

one_qubit_ops = ["HGATE", "IGATE", "PHASEGATE", "RGATE", "RXGATE", "RYGATE", "RZGATE", "SGATE", "SDGGATE", "SXGATE", "SXDGGATE", "TGATE", "TDGGATE", "UGATE", "U1GATE", "U2GATE", "XGATE", "YGATE", "ZGATE"]
two_qubit_ops = ["CHGATE", "CPHASEGATE", "CRXGATE", "CRYGATE", "CRZGATE", "CSGATE", "CSDGGATE", "CSXGATE", "CUGATE", "CXGATE", "CYGATE", "CZGATE", "DCXGATE", "ISWAPGATE", "RXXGATE", "RZZGATE", "SWAPGATE"]
three_qubit_ops = ["CCXGATE", "CCZGATE", "CSWAPGATE"]
four_qubit_ops = ["C3XGATE", "RCCCXGATE"]
def generate_random(n_qubits):
    circuit = QuantumCircuit(n_qubits, n_qubits)
    #use a seed for replication?
    #probably randomly generate phi, theta, etc. angle values that are fixed per circuit
    i = 0
    width = 5
    j = 0
    while j < width:
        i = 0
        while i < n_qubits:
            random_op_idx = random.randint(1, 4)
            if random_op_idx == 1:
                random_idx = random.randint(0,18)
                selected_op = one_qubit_ops[random_idx]
                if selected_op == "HGATE":
                    circuit.h(i)
                    i += 1
                elif selected_op == "IGATE":
                    circuit.id(i)
                    i += 1
                elif selected_op == "PHASEGATE":
                    circuit.p(1.5, i)
                    i += 1
                elif selected_op == "RXGATE":
                    circuit.rx(1.5, i)
                    i += 1
                elif selected_op == "RYGATE":
                    circuit.ry(1.5, i)
                    i += 1
                elif selected_op == "RZGATE":
                    circuit.rz(1.5, i)
                    i += 1
                elif selected_op == "SGATE":
                    circuit.s(i)
                    i += 1
                elif selected_op == "SDGGATE":
                    circuit.sdg(i)
                    i += 1
                elif selected_op == "SXGATE":
                    circuit.sx(i)
                    i += 1
                elif selected_op == "SXDGGATE":
                    circuit.sxdg(i)
                    i += 1
                elif selected_op == "TGATE":
                    circuit.t(i)
                    i += 1
                elif selected_op == "TDGGATE":
                    circuit.tdg(i)
                    i += 1
                elif selected_op == "UGATE":
                    circuit.u(1.5, 2.5, 3.5, i)
                    i += 1
                elif selected_op == "U1GATE":
                    circuit.u(0, 0, 1.5, i)
                    i += 1
                elif selected_op == "U2GATE":
                    circuit.u(np.pi/2, 1.5, 1.5, i)
                    i += 1
                elif selected_op == "XGATE":
                    circuit.x(i)
                    i += 1
                elif selected_op == "YGATE":
                    circuit.y(i)
                    i += 1
                elif selected_op == "ZGATE":
                    circuit.z(i)
                    i += 1
            elif random_op_idx == 2 and n_qubits-i>1:
                random_idx = random.randint(0,16)
                selected_op = two_qubit_ops[random_idx]
                if selected_op == "CHGATE":
                    circuit.ch(i, i+1)
                    i += 2
                elif selected_op == "CPHASEGATE":
                    circuit.cp(1.5, i, i+1)
                    i += 2
                elif selected_op == "CRXGATE":
                    circuit.crx(1.5, i, i+1)
                    i += 2
                elif selected_op == "CRYGATE":
                    circuit.cry(1.5, i, i+1)
                    i += 2
                elif selected_op == "CRZGATE":
                    circuit.crz(1.5, i, i+1)
                    i += 2
                elif selected_op == "CSGATE":
                    circuit.cs(i, i+1)
                    i += 2
                elif selected_op == "CSDGGATE":
                    circuit.csdg(i, i+1)
                    i += 2
                elif selected_op == "CSXGATE":
                    circuit.csx(i, i+1)
                    i += 2
                elif selected_op == "CUGATE":
                    circuit.cu(1.5, 1.5, 1.5, 1.5, i, i+1)
                    i += 2
                elif selected_op == "CXGATE":
                    circuit.cx(i, i+1)
                    i += 2
                elif selected_op == "CYGATE":
                    circuit.cy(i, i+1)
                    i += 2
                elif selected_op == "CZGATE":
                    circuit.cz(i, i+1)
                    i += 2
                elif selected_op == "DCXGATE":
                    circuit.dcx(i, i+1)
                    i += 2
                elif selected_op == "ISWAPGATE":
                    circuit.iswap(i, i+1)
                    i += 2
                elif selected_op == "RXXGATE":
                    circuit.rxx(1.5, i, i+1)
                    i += 2
                elif selected_op == "RZZGATE":
                    circuit.rzz(1.5, i, i+1)
                    i += 2
                elif selected_op == "SWAPGATE":
                    circuit.swap(i, i+1)
                    i += 2
                elif selected_op == "XXMINUSYYGATE":
                    xx = XXMinusYYGate(1.5, 1.5)
                    circuit.append(xx, [i, i+1])
                    i += 2
                elif selected_op == "XXPLUSYYGATE":
                    yy = XXPlusYYGate(1.5, 1.5)
                    circuit.append(yy, [i, i+1])
                    i += 2
            elif random_op_idx == 3 and n_qubits-i>2:
                random_idx = random.randint(0,2)
                selected_op = three_qubit_ops[random_idx]
                if selected_op == "CCXGATE":
                    circuit.append(CCXGate(), [i, i+1, i+2])
                    i += 3
                elif selected_op == "CCZGATE":
                    circuit.append(CCZGate(), [i, i+1, i+2])
                    i += 3
                elif selected_op == "CSWAPGATE":
                    circuit.append(CSwapGate(), [i, i+1, i+2])
                    i += 3
            elif random_op_idx == 4 and n_qubits-i>3:
                random_idx = random.randint(0,1)
                selected_op = four_qubit_ops[random_idx]
                if selected_op == "C3XGATE":
                    circuit.append(C3XGate(), [i, i+1, i+2, i+3])
                    i += 4
                elif selected_op == "RCCCXGATE":
                    circuit.rcccx(i, i+1, i+2, i+3)
                    i += 4
        j += 1
    return circuit

def controller(n_qubits:int=3, hardware: str='s', opt_method:str=''):
    circuit = generate_random(n_qubits)
    result, start_time, end_time, correct, total=None, None, None, 0, 0
    if hardware=='r':
        message=f'Pre {opt_method} optimized circuit: '
        opt_level = 0
        layout_method = ''
        routing_method = ''
        translation_method = ''
        temp_circuit = circuit
        if opt_method=='ZX':
            circuit = apply_zx_calc(circuit, n_qubits)
            message='Post ZX-calculus optimized circuit: '
        print('Circuit stats: ', circuit.count_ops())
        output_circuit(message, circuit)
        service = QiskitRuntimeService()
        backend = service.backend("ibm_brisbane")
        sampler = Sampler(mode=backend)
        pass_manager = generate_preset_pass_manager(
            optimization_level=opt_level, backend=backend, layout_method=layout_method, routing_method=routing_method, translation_method=translation_method
        )
        pre_opt_transpiled = pass_manager.run(temp_circuit)
        print("PRE OPT TRANSPILED: ", pre_opt_transpiled.count_ops())
        transpiled = pass_manager.run(circuit)
        with open("pre_opt_transpiled.txt", "w") as f:
            f.write(str(pre_opt_transpiled))
        with open("post_opt_transpiled.txt", "w") as f:
            f.write(str(transpiled))
        print('Transpiled circuit stats: ', transpiled.count_ops())
        job = sampler.run([(transpiled,)])
    elif hardware=='s':
        message='Pre ZX-calculus optimized circuit: '
        if opt_method=='ZX':
            circuit = apply_zx_calc(circuit, n_qubits)
            circuit.add_register(ClassicalRegister(n))
            circuit.measure(range(n), range(n))
            message='Post ZX-calculus optimized circuit: '
        print("Circuit stats: ", circuit.count_ops())
        output_circuit(message, circuit)
        # simulator = AerSimulator(method="matrix_product_state")
        # circuit.measure(2,2)
        # start_time = time.time()
        # result = simulator.run(circuit, shots=1000).result().get_counts()
        # end_time = time.time()

if __name__=='__main__':
    n = int(sys.argv[1])
    hardware = sys.argv[2]
    if len(sys.argv)>3:
        opt_method = sys.argv[3]
        controller(n, hardware, opt_method)
    else:
        controller(n, hardware, '')
