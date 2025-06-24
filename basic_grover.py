from qiskit import QuantumCircuit
from qiskit_algorithms import AmplificationProblem, Grover
from qiskit.primitives import Sampler

"""
making the oracle for the circuit:
- the oracle is essentially able to differentiate between the marked state and 
unmarked states when amplifying phases in the input superposition of states.
- applying the controlled-z gate to a default circuit makes it the oracle because
cz gates flip the phase of the target gate when the control qubit is 1
- cz(0,1) targets qubit 1, with qubit 0 as the control qubit
- AmplificationProblem is the qiskit algorithm class for amplitude amplification algorithms
- algo circuits are wrapped in a gate so they can function as a block. have to decompose the operator 
to see its component gates.
- successful search returns oracle_evaluation=True
- Sampler executes the circuits and returns the probability distributions of the measurement outcomes

- places for modification
 - size of oracle circuit
 - type of oracle circuit
 - input state (qubit preparation)
 - can explicitly specify each part of the Grover operator
 - number of iterations (of applying the Grover operator)
 - the marked state
 - optimization and where to make it more complex/simple
"""

marked_state = ['11']

oracle = QuantumCircuit(2) 
oracle.cz(0, 1)

problem = AmplificationProblem(oracle, is_good_state=marked_state)

circuit_text = problem.grover_operator.decompose().draw(output='text')
with open("circuit_drawings/basic_grover.txt", "w") as file: 
    file.write(str(circuit_text))

grover = Grover(sampler=Sampler())
result = grover.amplify(problem)
print("Result of oracle evaluation: ", result.oracle_evaluation)
print('Top measurement:', result.top_measurement)