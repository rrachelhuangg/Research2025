import os
import random
from qiskit import qpy
import numpy as np
from qiskit.circuit.library import HGate, IGate, PhaseGate, RXGate, RYGate, RZGate, SGate, SdgGate, SXGate, XXMinusYYGate, XXPlusYYGate, C3XGate, C4XGate, CCXGate, CCZGate, CSwapGate, SXdgGate, TGate, TdgGate, UGate, XGate, YGate, ZGate, CHGate, CPhaseGate, CRXGate, CRYGate, CRZGate, CSGate, CSdgGate, CSXGate, CUGate, CXGate, CYGate, CZGate, DCXGate, iSwapGate, RXXGate, RZZGate, SwapGate, XXMinusYYGate, XXPlusYYGate

one_qubit_ops = ["HGATE", "IGATE", "PHASEGATE", "RGATE", "RXGATE", "RYGATE", "RZGATE", "SGATE", "SDGGATE", "SXGATE", "SXDGGATE", "TGATE", "TDGGATE", "UGATE", "U1GATE", "U2GATE", "XGATE", "YGATE", "ZGATE"]
two_qubit_ops = ["CHGATE", "CPHASEGATE", "CRXGATE", "CRYGATE", "CRZGATE", "CSGATE", "CSDGGATE", "CSXGATE", "CUGATE", "CXGATE", "CYGATE", "CZGATE", "DCXGATE", "ISWAPGATE", "RXXGATE", "RZZGATE", "SWAPGATE"]
three_qubit_ops = ["CCXGATE", "CCZGATE", "CSWAPGATE"]
four_qubit_ops = ["C3XGATE", "RCCCXGATE"]

def mutate_circuit(circuit):
    print("PRE MUTATION CIRCUIT: ", circuit)
    print("\n")
    n_idx = 1
    n_to_mutate = random.randint(1, int(len(circuit.data)/4))+1
    mutate_idxs = []
    while len(mutate_idxs) < n_to_mutate:
        rand_idx = random.randint(0, int(len(circuit.data)/4))
        if rand_idx not in mutate_idxs:
            mutate_idxs += [rand_idx]
    for i in mutate_idxs:
        if(n_idx < n_to_mutate):
            instruction = circuit.data[i]
            if instruction.operation.num_qubits == 1:
                op = one_qubit_ops[random.randint(0, len(one_qubit_ops)-1)]
                if op == "HGATE":
                    circuit.data[i] = (HGate(), instruction[1], instruction[2])
                elif op == "IGATE":
                    circuit.data[i] = (IGate(), instruction[1], instruction[2])
                elif op == "PHASEGATE":
                    circuit.data[i] = (PhaseGate(1.5), instruction[1], instruction[2])
                elif op == "RXGATE":
                    circuit.data[i] = (RXGate(1.5), instruction[1], instruction[2])
                elif op == "RYGATE":
                    circuit.data[i] = (RYGate(1.5), instruction[1], instruction[2])
                elif op == "RZGATE":
                    circuit.data[i] = (RZGate(1.5), instruction[1], instruction[2])
                elif op == "SGATE":
                    circuit.data[i] = (SGate(), instruction[1], instruction[2])
                elif op == "SDGGATE":
                    circuit.data[i] = (SdgGate(), instruction[1], instruction[2])
                elif op == "SXGATE":
                    circuit.data[i] = (SXGate(), instruction[1], instruction[2])
                elif op == "SXDGGATE":
                    circuit.data[i] = (SXdgGate(), instruction[1], instruction[2])
                elif op == "TGATE":
                    circuit.data[i] = (TGate(), instruction[1], instruction[2])
                elif op == "TDGGATE":
                    circuit.data[i] = (TdgGate(), instruction[1], instruction[2])
                elif op == "UGATE":
                    circuit.data[i] = (UGate(1.5, 1.5, 1.5), instruction[1], instruction[2])
                elif op == "U1GATE":
                    circuit.data[i] = (UGate(0, 0, 1.5), instruction[1], instruction[2])
                elif op == "U2GATE":
                    circuit.data[i] = (UGate(np.pi/2, 1.5, 1.5), instruction[1], instruction[2])
                elif op == "XGATE":
                    circuit.data[i] = (XGate(), instruction[1], instruction[2])
                elif op == "YGATE":
                    circuit.data[i] = (YGate(), instruction[1], instruction[2])
                elif op == "ZGATE":
                    circuit.data[i] = (ZGate(), instruction[1], instruction[2])
                n_idx += 1
            elif instruction.operation.num_qubits == 2:
                op = two_qubit_ops[random.randint(0, len(two_qubit_ops)-1)]
                if op == "CHGATE":
                    circuit.data[i] = (CHGate(), instruction[1], instruction[2])
                elif op == "CPHASEGATE":
                    circuit.data[i] = (CPhaseGate(1.5), instruction[1], instruction[2])
                elif op == "CRXGATE":
                    circuit.data[i] = (CRXGate(1.5), instruction[1], instruction[2])
                elif op == "CRYGATE":
                    circuit.data[i] = (CRYGate(1.5), instruction[1], instruction[2])
                elif op == "CRZGATE":
                    circuit.data[i] = (CRZGate(1.5), instruction[1], instruction[2])
                elif op == "CSGATE":
                    circuit.data[i] = (CSGate(), instruction[1], instruction[2])
                elif op == "CSDGGATE":
                    circuit.data[i] = (CSdgGate(), instruction[1], instruction[2])
                elif op == "CSXGATE":
                    circuit.data[i] = (CSXGate(), instruction[1], instruction[2])
                elif op == "CUGATE":
                    circuit.data[i] = (CUGate(1.5, 1.5, 1.5, 1.5), instruction[1], instruction[2])
                elif op == "CXGATE":
                    circuit.data[i] = (CXGate(), instruction[1], instruction[2])
                elif op == "CYGATE":
                    circuit.data[i] = (CYGate(), instruction[1], instruction[2])
                elif op == "CZGATE":
                    circuit.data[i] = (CZGate(), instruction[1], instruction[2])
                elif op == "DCXGATE":
                    circuit.data[i] = (DCXGate(), instruction[1], instruction[2])
                elif op == "ISWAPGATE":
                    circuit.data[i] = (iSwapGate(), instruction[1], instruction[2])
                elif op == "RXXGATE":
                    circuit.data[i] = (RXXGate(1.5), instruction[1], instruction[2])
                elif op == "RZZGATE":
                    circuit.data[i] = (RZZGate(1.5), instruction[1], instruction[2])
                elif op == "SWAPGATE":
                    circuit.data[i] = (SwapGate(), instruction[1], instruction[2])
                elif op == "XXMINUSYYGATE":
                    circuit.data[i] = (XXMinusYYGate(1.5, 1.5), instruction[1], instruction[2])
                elif op == "XXPLUSYYGATE":
                    circuit.data[i] = (XXPlusYYGate(1.5, 1.5), instruction[1], instruction[2])
                n_idx += 2
            elif instruction.operation.num_qubits == 3:
                if op == "CCXGATE":
                    circuit.data[i] = (CCXGate(), instruction[1], instruction[2])
                elif op == "CCZGATE":
                    circuit.data[i] = (CCZGate(), instruction[1], instruction[2])
                elif op == "CSWAPGATE":
                    circuit.data[i] = (CSwapGate(), instruction[1], instruction[2])
                n_idx += 3
            elif instruction.operation.num_qubits == 4:
                if op == "C3XGATE":
                    circuit.data[i] = (C3XGate(), instruction[1], instruction[2])
                n_idx += 4
        else:
            break
    print("POST MUTATION CIRCUIT:", circuit)
    print("\n")
    return circuit

if __name__=='__main__':
    for filename in os.listdir('circuit_population/'):
        if "qpy" in filename:
            name = f"circuit_population/{filename}"
            print("NAME: ", name)
            with open(name, "rb") as file:
                circuit = qpy.load(file)[0]
                mutate_circuit(circuit)