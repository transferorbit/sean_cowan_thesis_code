'''
Author: Sean Cowan
Purpose: MSc Thesis
Date Created: 26-07-2022

This module performs the optimization calculations using the help modules from mga-low-thrust-utilities.py and
pygmo-utilities.py
'''
###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
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


# import mga_low_thrust_utilities as mga_util
import pygmo_island as isl
from pygmo_problem import MGALowThrustTrajectoryOptimizationProblem

current_dir = os.getcwd()
write_results_to_file = True
subdirectory = '/test_optimization_results/'

###########################################################################
# OPTIMIZE PROBLEM ########################################################
###########################################################################

mga_low_thrust_problem = MGALowThrustTrajectoryOptimizationProblem(no_of_free_parameters=2)
prob = pg.problem(mga_low_thrust_problem)


parallel = True
if parallel:
    #my_population = pg.population(mga_low_thrust_problem, size = 100, seed=33333)
    if __name__ == '__main__':
        mp.freeze_support()
        cpu_count = os.cpu_count()
        # print(os.sched_getaffinity(0))

        num_gen = 2
        pop_size = 100

        # my_problem = pg.minlp_rastrigin(300, 60)
        my_problem = prob
        my_population = pg.population(my_problem, size=pop_size, seed=32)
        # my_algorithm = pg.algorithm(pg.gaco(gen=num_gen))
        my_algorithm = pg.algorithm(pg.gaco(gen=num_gen))
        my_island = pg.mp_island()
        # my_island = isl.userdefinedisland_v3()
        # my_island = isl.my_isl()
        # print(prob)
        archi = pg.archipelago(n=cpu_count, algo = my_algorithm, prob=my_problem, pop_size = pop_size)#, udi = my_island)
        print("CPU count : %d \nIsland number : %d" % (cpu_count, cpu_count))

        # print(archi)
        for _ in range(2):
            archi.evolve()
            # archi.status
            archi.wait_check()

        print(archi.get_champions_x())
        # print(archi)
        #
        # champions_x = archi.get_champions_x()
        # print(champions_x[0])
        # best_trajectory = mga_low_thrust_problem.fitness(champions_x[0])
        # print(mga_low_thrust_problem.delta_v)
        # my_island.run_evolve(algo=my_algorithm, pop=my_population)
        #
        # Only try with island to reproduce problem
        # my_island = isl.my_isl()
        # my_island = pg.mp_island()
        # isl = pg.island(algo = my_algorithm, pop=my_population, udi = my_island)
        # isl.status
        # isl.evolve()
        # isl.status
        # my_island.shutdown_pool()

else:
    num_evol = 50
    pop_size = 100
    pop = pg.population(prob, size=pop_size)
    algo = pg.algorithm(pg.sga()) #multiple algorithms possible, sga, gaco, de
    # x_best = pop.get_x()
    # x_best[-1] = 0; x_best[-2] = 0
    # pop.push_back(x_best)

    fitness_list=  []
    population_list=  []
    for i in range(num_evol):
        pop = algo.evolve(pop)
        fitness_list.append(pop.get_f())
        population_list.append(pop.get_x())
        print(pop.champion_f)
        print('Evolving population; at generation ' + str(i))


    print(pop.champion_x)
    best_trajectory = mga_low_thrust_problem.fitness(pop.champion_x)

    state_history = mga_low_thrust_problem.get_states_along_trajectory(500)
    thrust_acceleration = mga_low_thrust_problem.get_inertial_thrust_accelerations_along_trajectory(500)
    node_times_list = mga_low_thrust_problem.get_node_times()

    node_times = {}
    for it, time in enumerate(node_times_list):
        node_times[it] = time

    print(node_times)

    if write_results_to_file:
        output_path = current_dir + subdirectory
    else:
        output_path = None

    if write_results_to_file:
        save2txt(state_history, 'state_history.dat', output_path)
        save2txt(thrust_acceleration, 'thrust_acceleration.dat', output_path)
        save2txt(node_times, 'node_times.dat', output_path)
