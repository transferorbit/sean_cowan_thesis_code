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

# If conda environment does not work
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
import mga_low_thrust_utilities as util
import manual_topology as topo
# from manual_topology import *

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

## constants
julian_day = constants.JULIAN_DAY

## General parameters
max_number_of_exchange_generations = 1
seed = 1032022


# test minlp optimization
# my_problem = pg.minlp_rastrigin(300, 60) 

# testing problem functionality
# max_no_of_gas = 6
# transfer_body_order = ["Earth", "Mars"]
# departure_planet = "Earth"
# arrival_planet = "Mars"
# free_param_count = 0
# num_gen = 15
# pop_size = 1000
# no_of_points = 4000
# bounds = [[-1000*julian_day, 1, 50*julian_day, -10**6, 0],
#         [1000*julian_day, 2000, 4000*julian_day, 10**6, 6]]
# subdirectory = '/test_optimization_results/'

max_no_of_gas = 6
# transfer_body_order = ["Earth", "Mars"]
departure_planet = "Earth"
arrival_planet = "Jupiter"
free_param_count = 0
num_gen = 1
pop_size = 100
no_of_points = 1000
bounds = [[10000*julian_day, 100, 50*julian_day, -10**6, 0],
        [10200*julian_day, 100, 2000*julian_day, 10**6, 6]]
subdirectory = '/island_testing/'


# validation

# optimization


planet_characters = ['Y', 'V', 'E', 'M', 'J', 'S', 'U', 'N']
planet_list = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]

# Remove excess planets above target planet (except for depature)
for it, i in enumerate(planet_list):
    if i == departure_planet:
        dep_index = it
    if i == arrival_planet:
        arr_index = it
if arr_index < dep_index:
    number_of_truncations = len(planet_list) - dep_index - 1
    # print(number_of_truncations)
else:
    number_of_truncations = len(planet_list) - arr_index - 1
    # print(number_of_truncations)
for i in range(number_of_truncations):
    planet_list.pop()

evaluated_sequences_database = current_dir + '/topology_database/evaluated_sequences_database.txt'
predefined_sequences_database = current_dir + '/topology_database/predefined_sequences_database.txt'

if os.path.exists(evaluated_sequences_database):
    os.remove(evaluated_sequences_database)
if os.path.exists(predefined_sequences_database):
    os.remove(predefined_sequences_database)

if __name__ == '__main__': #to prevent this code from running if this file is not the source file.
# https://stackoverflow.com/questions/419163/what-does-if-name-main-do
    mp.freeze_support()
    cpu_count = os.cpu_count() # not very relevant because differnent machines + asynchronous

    number_of_islands = 3*len(planet_list) + 1
    # print(len(planet_list)

    # Loop for number of sequence recursions
    for p in range(3): # gonna be max_no_of_gas
        print('Iteration: ', p, '\n')
        current_max_no_of_gas = max_no_of_gas - p
        temp_evaluated_sequences = []
        temp_predefined_sequences = []
        champions_x = {}
        champions_f = {}
        archi = pg.archipelago(n=0, seed=seed)

        # Add islands with pre-defined sequences
        island_problems = []
        if p == 0:
            it = -2 # -1 first iteration, second iteration it is 0
        else:
            it = -1 # goes immediately to 0

        for i in range(number_of_islands):
            it += 1
            if it == len(planet_list):
                it -= len(planet_list)

            predefined_sequence = [departure_planet]
            random_sequence = []
            random_sequence = \
                    topo.manualTopology.create_random_transfer_body_order(
                            arrival_planet=arrival_planet, max_no_of_gas=current_max_no_of_gas)

            if p == 0 and i == 0:
                random_sequence = [arrival_planet]
            elif p == 0:
                predefined_sequence += [planet_list[it]]
            else:
                predefined_sequence += locally_best_sequence + [planet_list[it]]

            temp_predefined_sequences.append(predefined_sequence)
            transfer_body_order = predefined_sequence + random_sequence
            print(transfer_body_order)
            temp_evaluated_sequences.append(transfer_body_order) 
            # print(temp_evaluated_sequences)

            island_to_be_added, current_island_problem = \
            topo.manualTopology.create_island(transfer_body_order=transfer_body_order,
                    free_param_count=free_param_count, bounds=bounds, num_gen=num_gen,
                    pop_size=pop_size)
            island_problems.append(current_island_problem)
            archi.push_back(island_to_be_added)

        # Evolve all islands
        for _ in range(max_number_of_exchange_generations): # step between which topology steps are executed
            archi.evolve()
            # print(archi)
            # archi.wait_check()
        archi.wait_check()

        champions_x[p] = archi.get_champions_x()
        # print(champions_x)
        champions_f[p] = archi.get_champions_f()
        # print(champion_f)

        # Write mga sequences to evaluated sequence database file
        delta_v = {}
        delta_v_per_leg = {}
        for j in range(number_of_islands):

            # define delta v to be used for evaluation of best sequence
            island_problems[j].post_processing_states(champions_x[p][j])
            delta_v[j] = island_problems[j].transfer_trajectory_object.delta_v
            delta_v_per_leg[j] = island_problems[j].transfer_trajectory_object.delta_v_per_leg

            # Save evaluated sequences to database with extra information
            file_object = open(evaluated_sequences_database, 'a+')
            mga_sequence_characters = \
                    util.transfer_body_order_conversion.get_mga_characters_from_list(
                                    temp_evaluated_sequences[j])
            file_object.write(mga_sequence_characters + ',' + str(delta_v[j]) + ',' +
                    str(delta_v_per_leg[j]) + '\n')
            file_object.close()
            
            island_fitness = champions_f[p][i]

            # Save predefined sequences to database
            file_object = open(predefined_sequences_database, 'a+')
            mga_sequence_characters = \
                    util.transfer_body_order_conversion.get_mga_characters_from_list(
                                    temp_predefined_sequences[j])
            file_object.write(mga_sequence_characters + '\n')
            file_object.close()

            # check auxiliary help for good legs -> SOLVE PICKLE ERROR
            # we have island_problms in a list
            # deltav per leg weighted average based on how far away the transfer is (EM is stricter
            # than YN

        print(delta_v)

        #assess sequences and define locally_best_sequence
        best_key = 0
        best_value = delta_v[0]
        for key, value in delta_v.items():
            if value < best_value:
                best_value = value
                best_key = key
        locally_best_sequence = temp_predefined_sequences[best_key]
        locally_best_sequence.pop(0)
        print(locally_best_sequence)
        # print(temp_predefined_sequences)





# Saving the trajectories for post-processing
    pp = False
    if pp:
        for i in range(number_of_islands):
            # print("Champion: ", champions[i])
            mga_low_thrust_problem = island_problems[i]
            mga_low_thrust_problem.post_processing_states(champions_x[2][j]) # 2 is one loop

            # State history
            state_history = \
            mga_low_thrust_problem.transfer_trajectory_object.states_along_trajectory(no_of_points)

            # Thrust acceleration
            thrust_acceleration = \
            mga_low_thrust_problem.transfer_trajectory_object.inertial_thrust_accelerations_along_trajectory(no_of_points)

            # Node times
            node_times_list = mga_low_thrust_problem.node_times

            # print(state_history)
            # print(node_times_list)

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
            auxiliary_info['Number of legs,'] = number_of_legs 
            auxiliary_info['Number of nodes,'] = number_of_nodes 
            auxiliary_info['Total ToF (Days),'] = time_of_flight / 86400.0
            departure_velocity = delta_v
            for j in range(number_of_legs):
                auxiliary_info['Delta V for leg %s,'%(j)] = delta_v_per_leg[j]
                departure_velocity -= delta_v_per_leg[j]
            auxiliary_info['Delta V,'] = delta_v 
            auxiliary_info['Departure velocity,'] = departure_velocity
            auxiliary_info['MGA Sequence,'] = \
            util.transfer_body_order_conversion.get_mga_characters_from_list(temp_evaluated_sequences[i])

            # auxiliary_info['Design parameter vector Island %s' % (i)] = champions[i]




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
