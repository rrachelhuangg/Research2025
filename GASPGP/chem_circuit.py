from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_algorithms.optimizers import L_BFGS_B
from qiskit_aer.primitives import Estimator

driver = PySCFDriver(
    atom='H .0 .0 .0; H .0 .0 0.735',
    unit=DistanceUnit.ANGSTROM,
    basis='sto3g',
)
problem = driver.run()

mapper = ParityMapper(num_particles=problem.num_particles)

optimizer = L_BFGS_B()

estimator = Estimator()

ansatz = UCCSD(
    problem.num_spatial_orbitals,
    problem.num_particles,
    mapper,
    initial_state=HartreeFock(
        problem.num_spatial_orbitals,
        problem.num_particles,
        mapper,
    ),
)

print("ANSATZ: ", ansatz.draw(output='text'))

# set up our actual VQE instance
from qiskit_algorithms import VQE

vqe = VQE(estimator, ansatz, optimizer)
# ensure that the optimizer starts in the all-zero state which corresponds to
# the Hartree-Fock starting point
vqe.initial_point = [0] * ansatz.num_parameters

# prepare the ground-state solver and run it
from qiskit_nature.second_q.algorithms import GroundStateEigensolver

algorithm = GroundStateEigensolver(mapper, vqe)

electronic_structure_result = algorithm.solve(problem)
electronic_structure_result.formatting_precision = 6
print(electronic_structure_result)

print("Optimal parameters: ", electronic_structure_result.raw_result.optimal_parameters)
print("Parameters: ", ansatz.parameters)
print("Num parameters: ", ansatz.num_parameters)