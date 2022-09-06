import numpy as np
import os
import pygmo as pg
import multiprocessing as mp

# Tudatpy imports
import sys
sys.path.insert(0, "/Users/sean/Desktop/tudelft/tudat/tudat-bundle/build/tudatpy")

import tudatpy
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.math import interpolators
from tudatpy.kernel.trajectory_design import shape_based_thrust
from tudatpy.kernel.trajectory_design import transfer_trajectory


from pygmo_problem import MGALowThrustTrajectoryOptimizationProblem

current_dir = os.getcwd()
write_results_to_file = True
subdirectory = '/test_optimization_results/'


# Create problem
mga_low_thrust_problem = MGALowThrustTrajectoryOptimizationProblem(no_of_free_parameters=1)
prob = pg.problem(mga_low_thrust_problem)


# Run optimization
if __name__ == '__main__':
    mp.freeze_support()
    cpu_count = os.cpu_count()

    num_gen = 2
    pop_size = 100

    my_problem = prob
    my_population = pg.population(my_problem, size=pop_size, seed=32)
    my_algorithm = pg.algorithm(pg.gaco(gen=num_gen))
    my_island = pg.mp_island()
    archi = pg.archipelago(n=cpu_count, algo = my_algorithm, prob=my_problem, pop_size = pop_size)#, udi = my_island)
    print("CPU count : %d \nIsland number : %d" % (cpu_count, cpu_count))

    for _ in range(2):
        archi.evolve()
        archi.wait_check()

    print(archi.get_champions_x())

