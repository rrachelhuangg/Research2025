- [x] Get Qiskit set up locally
- [x] Implement a generic version of Grover's algorithm for testing/understanding it
  - [x] Modify implementation based on specific project specs
  - [x] Make it configurable and try different options
- [x] Figure out what type of problem to apply these circuits to (just searching for a marked state or something more complex)
- [x] Implement Shor's algorithm

- [x] Implemented one-state quantum teleportation with 3 qubits
- [x] Applied ZX-calculus to teleportation circuit
- [x] Verified that teleportation circuit functionality was the same before and after applying zx-calculus

- [x] Build out data collection spreadsheet for experiments (running locally section)
- [x] Implement QFT algorithm circuit 
- [x] Implement pyzx full_reduce optimization for QFT circuit
- [x] Collect data from QFT experiments
- [x] Setup and run experiments for the smallest of each circuit's experimental/control group circuit

- [x] Run a circuit on IBM quantum hardware!
- [x] Realized that original QFT implementation was kinda scuffed and only work on local simulators w/o noise. Scrapped and completely rewrote for hardware. 
- [x] Figure out direction of project moving forward/where to focus on (compare ZX-calculus with other optimization methods)
- [x] Buff up code for QFT to be more modular as a model for the other circuits
- [x] Start collecting data for QFT n circuit using just ZX-calculus for pre/post comparison

- [x] Pick 2 other optimization methods to implement
- [x] Collect data for QFT 100 circuit (ZX-calculus)
- [x] Implement the 2 optimization methods for QFT
- [x] Collect data for 3 QFT circuits (using the other 2 optimization methods)
- [x] Record at least 3 trials for each experiment

- [x] Debugged original QFT Grover implementation
- [x] Refactor Grover circuit(s) to be modular with both circuit size and optimization method like revamped QFT circuit setup

-----------------------------------------------

- [ ] Automate experiment running in a sort of Dockerized pipeline CLI
- [ ] Automate creg/measure/barrier removal from qasm circuits (and/or circuit decomposition/transpilation) and then insertion back conversion process during application of zx
- [ ] QR code videos!
- [ ] Record a demo video of running pipeline on a couple examples
- [ ] Buff up all four circuits for ZX-calculus testing (the different test groups for each circuit and a control)
- [ ] Apply ZX-calculus to all circuits
- [ ] ...
- [ ] Implement tests for checking if optimized circuits and original circuits still have the same functionality

- [ ] Pass the test for this certificate [IBM Qiskit Certificate](https://www.ibm.com/training/certification/ibm-certified-associate-developer-quantum-computation-using-qiskit-v02x-C0010300) probably around when this project
is close to being completed (August)

