* Qiskit install locally: https://docs.quantum.ibm.com/guides/install-qiskit#local 

* circuit info (normal circuits and angle-based...): https://docs.quantum.ibm.com/guides/circuit-library
  * also talks about time-evolution circuit for qaoa circuit
  * benchmarking: accuracy score increases with size of circuit that can be reliably run. also takes
  into account qubit count, quality of instructions, etc. and post-processing results

* qiskit operators (matrices that act on quantum states): https://docs.quantum.ibm.com/guides/operators-overview 
  * Pauli operators

* Grover's search algorithm
  * the actual Grover paper: https://arxiv.org/pdf/quant-ph/9605043 
  * qiskit grover operator from api: https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.grover_operator
  * geeks for geeks explanation: https://www.geeksforgeeks.org/introduction-to-grovers-algorithm/
  * Step 1: phase inversion (on superposition of all input values):
    * if X is the special element, then the phase is inverted during each iteration (sqrt n)
    * all amplitudes start out equal at 1/(sqrt n)
  * Step 2: inversion about the mean 
    * flip the amplitudes over the mean
  * these two steps are done over and over again for ~sqrt n steps
  * after sqrt n steps, the marked element will have been found with amplitude 1/(sqrt 2) 
  * once the specified state is reached, the index of of the object corresponding to that state is returned

* extremely helpful youtube lecture series for in general stuff: https://www.youtube.com/watch?v=PAVKuYv1HC8&list=PLXEJgM3ycgQW5ysL69uaEdPoof4it6seB&index=44

* qiskit general instructions page: https://docs.quantum.ibm.com/guides/map-problem-to-circuits
* circuit library: https://docs.quantum.ibm.com/guides/circuit-library
* pre-built operators for specific circuits: https://docs.quantum.ibm.com/api/qiskit/circuit_library#particular-quantum-circuits
* qiskit api home page: https://docs.quantum.ibm.com/api/qiskit
* kind of related project: https://grishmaprs.medium.com/zx-calculus-qiskit-transpiler-pass-pyzx-and-my-qamp-fall-2022-experience-cf17a6731cda
* pyZX Python library implements ZX-calculus functionality: https://github.com/zxcalc/pyzx

* really helpful application of qiskit api stuff (this link is specifically for grover's algorithm, but good repo): https://github.com/Qiskit/qiskit-tutorials/blob/master/tutorials/algorithms/06_grover.ipynb

--------------------------------------------------------------------------------------------------
* 6/4 Meeting notes:
  * Work on graphics research project
  * start with an easier circuit that I know the answer to and implement it with API docs. compare it to a found given implementation to see where the gaps are.
  * have grover's base done by the very least for next meeting. to show and test. maybe have a couple variations.
  * maybe reevaluate timeline 

