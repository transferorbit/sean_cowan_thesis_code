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
# Still necessary to implement most recent version of the code

# import sys
# sys.path.insert(0, "/Users/sean/Desktop/tudelft/tudat/tudat-bundle/build/tudatpy")

import tudatpy
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.math import interpolators
from tudatpy.kernel.trajectory_design import shape_based_thrust
from tudatpy.kernel.trajectory_design import transfer_trajectory


# import mga_low_thrust_utilities as mga_util
# import pygmo_island as isl
from pygmo_problem import MGALowThrustTrajectoryOptimizationProblem

current_dir = os.getcwd()
write_results_to_file = True

###########################################################################
# OPTIMIZE PROBLEM ########################################################
###########################################################################

"""
Number of generations increases the amount of time spent in parallel computing the different islands
Number of evolutions requires redefining the islands which requires more time on single thread.
Population size; unknown
"""

julian_day = constants.JULIAN_DAY

# test minlp optimization
# my_problem = pg.minlp_rastrigin(300, 60) 

# testing problem functionality
# transfer_body_order = ["Earth", "Mars", "Mars", "Jupiter"]
# free_param_count = 0
# num_gen = 10
# pop_size = 500
# no_of_points = 500
# bounds = [[-1000*julian_day, 1, 50*julian_day, -10**6, 0],
#         [1000*julian_day, 2000, 2000*julian_day, 10**6, 6]]
# subdirectory = '/test_optimization_results/'

# verification
transfer_body_order = ["Earth", "Mars"]
free_param_count = 0
num_gen = 1
pop_size = 70
no_of_points = 500
# bounds = [[8000*julian_day, 1, 50*julian_day, -10**6, 0],
#         [10000*julian_day, 2000, 2000*julian_day, 10**6, 6]]
# subdirectory = '/verification/verification_results/'

# bounds = [[9265*julian_day, 1, 1070*julian_day, -10**6, 0],
#         [9265*julian_day, 1, 1070*julian_day, 10**6, 6]]
bounds = [[9300*julian_day, 150, 1185*julian_day, -10**6, 2],
        [9300*julian_day, 150, 1185*julian_day, 10**6, 4]]
# subdirectory = '/verification/verification_results/'
subdirectory = '/verification/tdep9300_vdep150_tof1185_rev3/'

# validation

# optimization

# making problem
mga_low_thrust_problem = \
MGALowThrustTrajectoryOptimizationProblem(transfer_body_order=transfer_body_order,
        no_of_free_parameters=free_param_count, bounds=bounds)
# mga_low_thrust_problem.get_system_of_bodies()
prob = pg.problem(mga_low_thrust_problem)

if __name__ == '__main__': #to prevent this code from running if this file is not the source file.
# https://stackoverflow.com/questions/419163/what-does-if-name-main-do
    mp.freeze_support()
    cpu_count = os.cpu_count()
    # print(os.sched_getaffinity(0))

    my_population = pg.population(prob, size=pop_size, seed=32)
    my_algorithm = pg.algorithm(pg.gaco(gen=num_gen))
    # my_island = pg.mp_island()
    archi = pg.archipelago(n=cpu_count, algo = my_algorithm, prob=prob, pop_size = pop_size)#, udi = my_island)
    print("CPU count : %d \nIsland number : %d" % (cpu_count, cpu_count))

    for _ in range(1): # step between which topology steps are executed
        archi.evolve()
        # archi.status
        # archi.wait_check()
        archi.wait_check()

# End of simulations

    champions = archi.get_champions_x()
    champion_fitness = archi.get_champions_f()
    print(champions[0])

# Saving the trajectories for post-processing
    for i in range(len(champions)):
        mga_low_thrust_problem = \
        MGALowThrustTrajectoryOptimizationProblem(transfer_body_order=transfer_body_order,
                no_of_free_parameters=free_param_count, bounds=bounds)
        # mga_low_thrust_problem.get_system_of_bodies()
        # print("Champion: ", champions[i])
        mga_low_thrust_problem.post_processing_states(champions[i])

        # State history
        state_history = \
        mga_low_thrust_problem.transfer_trajectory_object.states_along_trajectory(no_of_points)

        # Thrust acceleration
        thrust_acceleration = \
        mga_low_thrust_problem.transfer_trajectory_object.inertial_thrust_accelerations_along_trajectory(no_of_points)

        # Node times
        node_times_list = mga_low_thrust_problem.node_times
        node_times_days_list = [i / constants.JULIAN_DAY for i in node_times_list]

        # print(state_history)
        print(node_times_list)
        print(node_times_days_list)

        node_times = {}
        for it, time in enumerate(node_times_list):
            node_times[it] = time

        # Auxiliary information
        delta_v = mga_low_thrust_problem.transfer_trajectory_object.delta_v
        delta_v_per_leg = mga_low_thrust_problem.transfer_trajectory_object.delta_v_per_leg
        number_of_legs = mga_low_thrust_problem.transfer_trajectory_object.number_of_legs
        number_of_nodes = mga_low_thrust_problem.transfer_trajectory_object.number_of_nodes
        time_of_flight = mga_low_thrust_problem.transfer_trajectory_object.time_of_flight

        auxiliary_info = {}
        auxiliary_info['Number of legs'] = number_of_legs 
        auxiliary_info['Number of nodes'] = number_of_nodes 
        auxiliary_info['Total ToF (Days)'] = time_of_flight / 86400.0
        departure_velocity = delta_v
        for j in range(number_of_legs):
            auxiliary_info['Delta V for leg %s'%(j)] = delta_v_per_leg[j]
            departure_velocity -= delta_v_per_leg[j]
        auxiliary_info['Delta V'] = delta_v 
        auxiliary_info['Departure velocity'] = departure_velocity
        # auxiliary_info['Number of revolutions %s' % (i)] = champions[i][-1]
        # auxiliary_info['Number of revolutions %s' % (i)] = champions[i][-2]
        # auxiliary_info['Number of revolutions %s' % (i)] = champions[i][-3]
        # auxiliary_info['Number of revolutions %s' % (i)] = champions[i][-4]



        # Saving files
        unique_identifier = "island_" + str(i) + "/"

        if write_results_to_file:
            output_path = current_dir + subdirectory + unique_identifier
        else:
            output_path = None

        if write_results_to_file:
            save2txt(state_history, 'state_history.dat', output_path)
            save2txt(thrust_acceleration, 'thrust_acceleration.dat', output_path)
            save2txt(node_times, 'node_times.dat', output_path)
            save2txt(auxiliary_info, 'auxiliary_info.dat', output_path)
