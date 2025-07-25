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

* shor's algorithm helpful github page: https://github.com/qiskit-community/qiskit-community-tutorials/blob/master/algorithms/shor_algorithm.ipynb
* paper about a scalable shor's algorithm: https://www.science.org/doi/10.1126/science.aad9480
* another github about shor's: https://github.com/tiagomsleao/ShorAlgQiskit/blob/master/Shor_Normal_QFT.py
* helpful page with step by step classical and quantum shor's implementations: https://softwaredominos.com/home/science-technology-and-other-fascinating-topics/quantum-computing-beyond-qubits-part-4-shors-algorithm-for-factoring-large-numbers/
* another helpful shor's page with step by step explanation: https://qmunity.thequantuminsider.com/2024/06/10/shors-algorithm/

* pyzx docs: https://pyzx.readthedocs.io/en/latest/simplify.html
* valid pyzx circuit format: https://pyzx.readthedocs.io/en/latest/representations.html
* another interesting quantum project writeup: https://grishmaprs.medium.com/zx-calculus-qiskit-transpiler-pass-pyzx-and-my-qamp-fall-2022-experience-cf17a6731cda
* stackoverflow post to possibly generalize quantum teleportation to n states: https://quantumcomputing.stackexchange.com/questions/21600/generalizing-the-circuit-for-quantum-teleportation-for-n-qubit-states
* another stackoverflow post on a similar topic to the above: https://quantumcomputing.stackexchange.com/questions/21600/generalizing-the-circuit-for-quantum-teleportation-for-n-qubit-states
* n-qubit quantum state teleportation paper: https://arxiv.org/pdf/1704.05294
* qiskit teleportation docs: https://learning.quantum.ibm.com/course/utility-scale-quantum-computing/lesson-03-teleportation
* geeks for geeks quantum teleportation basic step-by-step implementation: https://www.geeksforgeeks.org/python/quantum-teleportation-in-python/

* qiskit qft docs: https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.QFT
* a github page about qft implementation: https://github.com/qiskit-community/qiskit-community-tutorials/blob/master/terra/qis_adv/fourier_transform.ipynb
* openqasm docs: https://github.com/openqasm/openqasm
* another qiskit qft implementation webpage: https://quantumcomputinguk.org/tutorials/quantum-fourier-transform-in-qiskit
* gate-level optimization (non zx, though): https://pyzx.readthedocs.io/en/latest/simplify.html
* polynomial circuits paper: https://arxiv.org/abs/1303.2042
* pyzx implementation paper: https://arxiv.org/pdf/1903.10477
* pyzx verify_equality and compare_tensors: https://pyzx.readthedocs.io/en/latest/api.html
* can decompose a qiskit circuit into fundamental gates using qiskit transpile and decompose

* ibm qiskit docs: https://quantum.cloud.ibm.com/docs/en/api/qiskit/0.24/qiskit.circuit.library.QFT
* super helpful qft github with instructions for running on actual hardware too!!! https://github.com/qiskit-community/qiskit-community-tutorials/blob/master/terra/qis_adv/fourier_transform.ipynb
* ^ above used c1 and u1 links were deprecated though - had to be replaced
* qiskit also has documentation for specific gates: https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.CU1Gate
* what the qiskit qft circuit returns: https://www.google.com/search?q=what+does+the+qiskit+qft+circuit+return&oq=what+does+the+qiskit+qft+circuit+return&gs_lcrp=EgZjaHJvbWUyCQgAEEUYORigATIHCAEQIRiPAjIHCAIQIRiPAtIBCDUyMDJqMGo3qAIAsAIA&sourceid=chrome&ie=UTF-8
* super helpful github for studying for the qiskit certificate: https://github.com/pratjz/IBM-Qiskit-Certification-Exam-Prep

* qiskit runtime api docs: https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime
* stack overflow about checking the fidelity value of quantum states: https://quantumcomputing.stackexchange.com/questions/41169/studying-qft-circuit-noise-in-qiskit-comparing-statevectors-with-state-fidelity
* ibm quantum cloud hardware service updates link: https://quantum.cloud.ibm.com/announcements/en/service-alerts
* QFT circuits compilation paper: https://arxiv.org/html/2312.16114v1
* application of qft circuit: https://www.google.com/search?q=application+of+qft+circuit&oq=application+of+qft+circuit&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIHCAEQIRigAdIBCDQ2MDBqMGo5qAIAsAIB&sourceid=chrome&ie=UTF-8
* IBM quantum cloud hardware link: https://quantum.cloud.ibm.com/
* tranpiler that has to be used when running on IBM hardware: https://quantum.cloud.ibm.com/docs/en/guides/transpile-with-pass-managers
* setting up ibm quantum cloud account: https://quantum.cloud.ibm.com/docs/en/guides/cloud-setup#cloud-save

* writing plugins: https://quantum.cloud.ibm.com/docs/en/api/qiskit/transpiler_plugins#module-qiskit.transpiler.preset_passmanagers.plugin
* super helpful qiskit pass manager/transpiler docs: https://quantum.cloud.ibm.com/docs/en/api/qiskit/transpiler
* qiskit visualization docs: https://quantum.cloud.ibm.com/docs/en/api/qiskit/visualization
* qiskit compiler docs: https://quantum.cloud.ibm.com/docs/en/api/qiskit/compiler
* installing plugins: https://quantum.cloud.ibm.com/docs/en/guides/transpiler-plugins
* reasons for circuit optimization and key optimization metrics: https://www.google.com/search?q=optimizing+qiskit+circuit+techniques&oq=optimizing+qiskit+circuit+techniques&gs_lcrp=EgZjaHJvbWUyCQgAEEUYORigATIHCAEQIRigATIHCAIQIRigAdIBCTEwOTY5ajBqN6gCALACAA&sourceid=chrome&ie=UTF-8
  * reduces required resources
  * increases accuracy
* qiskit pass manager class api docs: https://quantum.cloud.ibm.com/docs/en/api/qiskit/0.38/qiskit.transpiler.preset_passmanagers.generate_preset_pass_manager
* lower-level tranpiler passes: https://quantum.cloud.ibm.com/docs/en/api/qiskit/transpiler_passes
* updated qiskit package transpilation docs: https://quantum.cloud.ibm.com/docs/en/guides/transpile
* dynamic decoupling: https://www.google.com/search?q=what+are+dynamic+decoupling+sequences+in+qiskit+transpiler&oq=what+are+dynamic+decoupling+sequences+in+qiskit+transpiler&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIHCAEQIRigATIHCAIQIRigATIHCAMQIRigATIHCAQQIRigATIHCAUQIRigATIHCAYQIRiPAtIBCTE1OTgyajBqN6gCALACAA&sourceid=chrome&ie=UTF-8
* other commonly used parameters for transpilation: https://quantum.cloud.ibm.com/docs/en/guides/common-parameters
* 

