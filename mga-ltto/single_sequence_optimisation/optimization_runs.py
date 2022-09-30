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
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel.trajectory_design import shape_based_thrust
from tudatpy.kernel.trajectory_design import transfer_trajectory
from tudatpy.kernel.interface import spice
spice.load_standard_kernels()


import mga_low_thrust_utilities as util
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
my_problem = pg.rastrigin(5) 

# testing problem functionality
transfer_body_order = ["Earth", "Mars", "Jupiter"]
mga_sequence_characters = util.transfer_body_order_conversion.get_mga_characters_from_list(
        transfer_body_order)
free_param_count = 0
num_gen = 2
pop_size = 100
no_of_points = 500
bound_names= ['Departure date', 'Departure velocity', 'Time of Flight',
        'Free coefficient', "Number of revolutions"]
bounds = [[1000*julian_day, 100, 50*julian_day, -10**6, 0],
        [1000*julian_day, 100, 800*julian_day, 10**6, 0]]
subdirectory = '/test_optimization_results/'

# verification
# transfer_body_order = ["Earth", "Mars"]
# free_param_count = 0
# num_gen = 5
# pop_size = 200
# no_of_points = 500
# bounds = [[1000*julian_day, 1, 50*julian_day, -10**6, 0],
#         [1200*julian_day, 200, 2000*julian_day, 10**6, 6]]
# subdirectory = '/verification/verification_results/'

# TGRRoegiers p.116
# mjd_depart_lb = 58849
# mjd_depart_ub = 61770
# mjd_2000=  51544.5
# bounds = [[(mjd_depart_lb-mjd_2000)*julian_day, 1, 500*julian_day, -10**6, 1],
#         [(mjd_depart_ub-mjd_2000)*julian_day, 1, 2000*julian_day, 10**6, 4]]
# subdirectory = '/verification/roegiers_test5/'


# bounds = [[9265*julian_day, 1, 1070*julian_day, -10**6, 0],
#         [9265*julian_day, 1, 1070*julian_day, 10**6, 6]]
# bounds = [[9300*julian_day, 150, 1185*julian_day, -10**6, 2],
#         [9300*julian_day, 150, 1185*julian_day, 10**6, 4]]
# subdirectory = '/verification/verification_results/'

# validation

# optimization

# making problem
# bodies_to_create = ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn"]
# central_body_mu = 1.3271244e20 # m^3 / s^2
#
# planet_kep_states = []
# for i in bodies_to_create:
#     current_cartesian_elements = spice.get_body_cartesian_state_at_epoch(i,
#             "Sun",
#             "ECLIPJ2000",
#             'None',
#             bounds[0][0])
#     planet_kep_states.append(element_conversion.cartesian_to_keplerian(current_cartesian_elements,
#         central_body_mu))
#
# print(planet_kep_states)

# print('Creating problem class')
mga_low_thrust_problem = \
MGALowThrustTrajectoryOptimizationProblem(transfer_body_order=transfer_body_order,
        no_of_free_parameters=free_param_count, bounds=bounds)#, planet_kep_states=planet_kep_states)
# mga_low_thrust_problem.get_system_of_bodies()
# prob = my_problem
prob = pg.problem(mga_low_thrust_problem)

if __name__ == '__main__': #to prevent this code from running if this file is not the source file.
# https://stackoverflow.com/questions/419163/what-does-if-name-main-do
    mp.freeze_support()
    cpu_count = os.cpu_count()
    number_of_islands = cpu_count
    # print(os.sched_getaffinity(0))

    my_population = pg.population(prob, size=pop_size, seed=32)
    my_algorithm = pg.algorithm(pg.sga(gen=num_gen))
    # my_island = pg.mp_island()
    print('Creating archipelago')
    archi = pg.archipelago(n=number_of_islands, algo = my_algorithm, prob=prob, pop_size = pop_size)#, udi = my_island)
    print("CPU count : %d \nIsland number : %d" % (cpu_count, cpu_count))

    for _ in range(1): # step between which topology steps are executed
        print('Evolving ..')
        archi.evolve()
        # archi.status
        # archi.wait_check()
        archi.wait_check()
    print('Evolution finished')

# End of simulations

    champions = archi.get_champions_x()
    champion_fitness = archi.get_champions_f()
    print(champions[0])

# Saving the trajectories for post-processing
    champions_dict = {}
    champion_fitness_dict = {}
    for i in range(len(champions)):
        mga_low_thrust_problem.fitness(champions[i], post_processing=True)

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
        # print(node_times_list)
        # print(node_times_days_list)

        node_times = {}
        for it, time in enumerate(node_times_list):
            node_times[it] = time

        # Auxiliary information
        delta_v = mga_low_thrust_problem.transfer_trajectory_object.delta_v
        delta_v_per_leg = mga_low_thrust_problem.transfer_trajectory_object.delta_v_per_leg
        number_of_legs = mga_low_thrust_problem.transfer_trajectory_object.number_of_legs
        number_of_nodes = mga_low_thrust_problem.transfer_trajectory_object.number_of_nodes
        time_of_flight = mga_low_thrust_problem.transfer_trajectory_object.time_of_flight

        if write_results_to_file:
            auxiliary_info = {}
            auxiliary_info['Number of legs,'] = number_of_legs 
            auxiliary_info['Number of nodes,'] = number_of_nodes 
            auxiliary_info['Total ToF (Days),'] = time_of_flight / 86400.0
            departure_velocity = delta_v
            for j in range(number_of_legs):
                auxiliary_info['Delta V for leg %s,'%(j)] = delta_v_per_leg[j]
                departure_velocity -= delta_v_per_leg[j]
            auxiliary_info['Delta V,'] = delta_v 
            auxiliary_info['Departure velocity,'] = departure_velocity
            auxiliary_info['MGA Sequence,'] = mga_sequence_characters

            # Saving files
            unique_identifier = "island_" + str(i) + "/"
            save2txt(state_history, 'state_history.dat', current_dir + subdirectory +
                    unique_identifier)
            save2txt(thrust_acceleration, 'thrust_acceleration.dat', current_dir + subdirectory +
                unique_identifier)
            save2txt(node_times, 'node_times.dat', current_dir + subdirectory + unique_identifier)
            save2txt(auxiliary_info, 'auxiliary_info.dat', current_dir + subdirectory +
                unique_identifier)


    if write_results_to_file:
        champions_dict[i] = champions[i]
        champion_fitness_dict[i] = champion_fitness[i]

        optimisation_characteristics = {}
        optimisation_characteristics['Transfer body order,'] = mga_sequence_characters
        optimisation_characteristics['Free parameter count,'] = free_param_count
        optimisation_characteristics['Number of generations,'] = num_gen
        optimisation_characteristics['Population size,'] = pop_size
        optimisation_characteristics['CPU count,'] = cpu_count
        optimisation_characteristics['Number of islands,'] = number_of_islands
        for j in range(len(bounds)):
            for k in range(len(bounds[0])):
                optimisation_characteristics[bound_names[k] + ','] = bounds[j][k]
        # optimisation_characteristics['Bounds'] = bounds
        
        # This should be for the statistics, we already save all information per
        unique_identifier = "champions/"
        save2txt(champion_fitness_dict, 'champion_fitness.dat', current_dir + subdirectory +
                unique_identifier)
        save2txt(champions_dict, 'champions.dat', current_dir + subdirectory +
                unique_identifier)
        unique_identifier = ""
        save2txt(optimisation_characteristics, 'optimisation_characteristics.dat', current_dir +
                subdirectory + unique_identifier)
