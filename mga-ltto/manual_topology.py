'''
Author: Sean Cowan
Purpose: MSc Thesis
Date Created: 14-09-2022

This module aims at implementing the algorithm that exchanges information and determines what
islands to pass to the archipelago
'''
###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import numpy as np
import os
import pygmo as pg
import multiprocessing as mp
import random

# Tudatpy imports
# Still necessary to implement most recent version of the code

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
from pygmo_problem import MGALowThrustTrajectoryOptimizationProblem
import mga_low_thrust_utilities as util

current_dir = os.getcwd()
write_results_to_file = True

class manualTopology:

    def __init__(self) -> None:
        pass

    @staticmethod
    def create_island(transfer_body_order, free_param_count, bounds, num_gen, pop_size):
        mga_low_thrust_object = \
        MGALowThrustTrajectoryOptimizationProblem(transfer_body_order=transfer_body_order,
                no_of_free_parameters=free_param_count, bounds=bounds)
        problem = pg.problem(mga_low_thrust_object)
        algorithm = pg.algorithm(pg.gaco(gen=num_gen))
        return pg.island(algo=algorithm, prob=problem, size=pop_size, udi=pg.mp_island()), mga_low_thrust_object

    @staticmethod
    def create_island_2(transfer_body_order, free_param_count, bounds, num_gen, pop_size):
    # Create islands with population adding rather than problem creating and then random population
    # initialization
        no_of_predefined_individuals = len(predefined_legs)
        no_of_random_individuals = pop_size - no_of_predefined_individuals
        population = \
        pg.population(prob=MGALowThrustTrajectoryOptimizationProblem(transfer_body_order=transfer_trajectory,
            no_of_free_parameters=free_param_count, bounds=bounds), size=no_of_random_individuals)
        for i in range(no_of_predefined_individuals):
            # tof = 
            population.push_back()

    @staticmethod
    def create_random_transfer_body_order(arrival_planet, possible_planets, seed=None, max_no_of_gas=6) -> list:

        # transfer_body_order.append(predefined_sequence) if predefined_sequence != [] else None #?
        # create random sequence of numbers with max_no_of_gas length
        random.seed(seed)
        sequence_length = random.randrange(0, max_no_of_gas)
        sequence_digits = [random.randrange(1, (1 + possible_planets)) for _ in range(sequence_length)]

        # transform that into transfer_body_order
        transfer_body_strings = util.transfer_body_order_conversion.get_transfer_body_list(sequence_digits)

        transfer_body_strings.append(arrival_planet)

        return transfer_body_strings

    def add_sequence_to_database(self):
        pass

    @staticmethod
    def remove_excess_planets(planet_list : list, departure_planet : str, arrival_planet : str) -> list:
        """
        Removes excess planets from planet_list or planet_characters

        Returns
        -------
        Truncated list of planets/planet characters
        """
        for it, i in enumerate(planet_list):
            if i == departure_planet or i == departure_planet[0]:
                dep_index = it
            if i == arrival_planet or i == arrival_planet[0]:
                arr_index = it
        if arr_index < dep_index:
            number_of_truncations = len(planet_list) - dep_index - 1
            # print(number_of_truncations)
        else:
            number_of_truncations = len(planet_list) - arr_index - 1
            # print(number_of_truncations)
        for i in range(number_of_truncations):
            planet_list.pop()
        return planet_list

    @staticmethod
    def get_leg_specifics(mga_sequence : str, champion_x : list, delta_v_per_leg : list,
            number_of_free_coefficients : int = 0) -> str:
        """
        'EMMJ' EM MM MJ
        EM, dV, ToF, #rev
        MM, dV, ToF, #rev
        MJ, dV, ToF, #rev
        """
        leg_information = ""
        chars = [i for i in mga_sequence]
        number_of_legs = len(chars) - 1
        for i in range(number_of_legs):
            current_leg = chars[i] + chars[i+1]
            current_dV = delta_v_per_leg[i]
            current_tof = champion_x[2+i] / 86400
            current_rev = champion_x[2 + number_of_legs + number_of_free_coefficients * 3 * number_of_legs + i]
            leg_information += "%s, %d, %d, %i\n" % (current_leg, current_dV, current_tof, current_rev)
        return leg_information

    @staticmethod
    def get_itbs(dv : dict=None, ptbs : dict=None, type_of_selection : str="max", pc : list = None,
            pl : list = None, dt_tuple : tuple=(None, None)):
        if type_of_selection == "max":
            best_key = 0
            best_value = dv[0]
            for key, value in dv.items():
                if value < best_value:
                    best_value = value
                    best_key = key
            locally_best_sequence = ptbs[best_key]
            locally_best_sequence.pop(0)
            return locally_best_sequence
        if type_of_selection == "proportional":

            ### Get results into lists
            sequence_delta_v_value_dict = {}
            no_of_possible_planets = len(pl)
            for i in range(no_of_possible_planets): # for each possible planet
                current_planet_char = pc[i % len(pc)]
                list_of_delta_v_for_current_pc = []
                for it, j in enumerate(ptbs): # go check all indices 
                    ptbs_characters = util.transfer_body_order_conversion.get_mga_characters_from_list(j)
                    last_character = ptbs_characters[-1]
                    if last_character == current_planet_char: # if last value is j
                        list_of_delta_v_for_current_pc.append(dv[it]) # +1 because direct transfer removed

                        ### Make search function recursive, to not double check
                        # make use of list(delta_v.keys()) to change the keys of dict
                        # ptbs.pop(it) # remove this item fromdv and ptbs
                        # dv.pop(it)
                        # for it2 in dv.keys():
                        #     if it2 == 0:
                        #         continue
                        #     dv[it2-1] = dv.pop(it2)

                sequence_delta_v_value_dict[i] = list_of_delta_v_for_current_pc

            ### Get statistics
            quantities = ['max', 'min', 'mean', 'median']
            ptbs_stats = np.zeros((no_of_possible_planets, len(quantities))) 
            for i, j in sequence_delta_v_value_dict.items():
                print
                current_max = np.max(j)
                current_min = np.min(j)
                current_mean = np.mean(j)
                current_median = np.median(j)
                # print(current_max, current_min, current_mean, current_median)
                ptbs_stats[i, :] = np.array([current_max, current_min, current_mean,
                    current_median])
            # print(ptbs_stats)

            ### Determine scalarization (what is an optimal combination of statistical quantities
            arg_min = np.argmin(ptbs_stats[:,1])
            local_char = pc[arg_min]
            itbs = [util.transfer_body_order_conversion.get_mga_list_from_characters(local_char) for _ in
                range (no_of_possible_planets)]
            # locally_best_sequence = [pc[ for _ in range(no_of_possible_pl

            return itbs

def run_mgso_optimisation(departure_planet : str,
                            arrival_planet : str,
                            free_param_count : int,
                            num_gen : int,
                            pop_size : int,
                            no_of_points : int,
                            bounds : list,
                            output_directory : str,
                            subdirectory : str,
                            max_no_of_gas = 1,
                            no_of_sequence_recursions = 1,
                            max_number_of_exchange_generations = 1,
                            number_of_sequences_per_planet : list =  [],
                            seed : int = 421):

    planet_characters = ['Y', 'V', 'E', 'M', 'J', 'S', 'U', 'N']
    planet_list = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
    
    # Remove excess planets above target planet (except for depature)
    planet_list = manualTopology.remove_excess_planets(planet_list, departure_planet, arrival_planet)
    planet_characters = manualTopology.remove_excess_planets(planet_characters, departure_planet, arrival_planet)
    unique_identifier = '/topology_database'

    evaluated_sequences_database = output_directory + subdirectory + unique_identifier +  '/evaluated_sequences_database.txt'
    separate_leg_database = output_directory + subdirectory +  unique_identifier +  '/separate_leg_database.txt'
    
    if not os.path.exists(output_directory + subdirectory + unique_identifier):
        os.makedirs(output_directory + subdirectory + unique_identifier)
    if not os.path.exists(evaluated_sequences_database):
        open(evaluated_sequences_database, 'a+').close()
    if not os.path.exists(separate_leg_database):
        open(separate_leg_database, 'a+').close()

    mp.freeze_support()
    cpu_count = os.cpu_count() # not very relevant because differnent machines + asynchronous

    # print(len(planet_list)
    evaluated_sequences_results = 'Sequence, Delta V, ToF\n'
    leg_results = 'Leg, Delta V, ToF, #rev\n'
    champions_x = {}
    champions_f = {}
    island_problems = {}
    evaluated_sequences_dict = {}
    mga_sequence_characters_list = []
    itbs = []
    number_of_islands_list = [] # list of number of islands per recursion
    list_of_lists_of_x_dicts = []# list containing list of dicts of champs per gen
    list_of_lists_of_f_dicts = []
    
###########################################################################
# MGSO Optimisation ###############################################
###########################################################################
    
    # Loop for number of sequence recursions
    p_bump = False
    for p in range(no_of_sequence_recursions): # gonna be max_no_of_gas

        possible_planets = len(planet_list)
        combinations_left = possible_planets**(max_no_of_gas-p)# or no_of_sequence_recursions

        if p_bump == True:
            p -= 1
            p_bump = False
        print('Iteration: ', p, '\n')
        # Size of random sequence decreases
        current_max_no_of_gas = max_no_of_gas - p
        current_island_problems = []
        temp_evaluated_sequences = []
        temp_ptbs = []
        print('Creating archipelago')
        archi = pg.archipelago(n=0, seed=seed)

        # Add islands with pre-defined sequences
        directtransferbump = 0
        if p == 0:
            number_of_islands = number_of_sequences_per_planet[p]*len(planet_list) + 1
            transfer_body_order = [departure_planet, arrival_planet]
            temp_evaluated_sequences.append(transfer_body_order) 
            temp_ptbs.append([departure_planet])
            island_to_be_added, current_island_problem = \
            manualTopology.create_island(transfer_body_order=transfer_body_order,
                    free_param_count=free_param_count, bounds=bounds, num_gen=1, #check also p! vv
                    pop_size=pop_size)
            current_island_problems.append(current_island_problem)
            archi.push_back(island_to_be_added)
            directtransferbump = 1
        else:
            number_of_islands = number_of_sequences_per_planet[p]*len(planet_list)

            if number_of_islands >= combinations_left:
                print('Number of islands from %i to %i' % (number_of_islands, combinations_left))
                number_of_islands = combinations_left

        print('Number of islands: ', number_of_islands, '\n')
        number_of_islands_list.append(number_of_islands)
        if p == 0:
            print(transfer_body_order)
        for i in range(number_of_islands - directtransferbump):
            ptbs = [departure_planet]
            random_sequence = []
            # can add seed argument but that does not work yet as intended
            random_sequence = \
                    manualTopology.create_random_transfer_body_order(
                            arrival_planet=arrival_planet, possible_planets=possible_planets, max_no_of_gas=current_max_no_of_gas)

            if p == 0:
                ptbs += [planet_list[i % len(planet_list)]]
            else:
                # [i] so that each island can start with its own predefined
                ptbs += itbs[i % len(planet_list)] + [planet_list[i % len(planet_list)]]

            temp_ptbs.append(ptbs)
            transfer_body_order = ptbs + random_sequence
            temp_evaluated_sequences.append(transfer_body_order) 

            print(transfer_body_order)
            island_to_be_added, current_island_problem = \
            manualTopology.create_island(transfer_body_order=transfer_body_order,
                    free_param_count=free_param_count, bounds=bounds, num_gen=1, #check also p=0 ^^
                    pop_size=pop_size)
            current_island_problems.append(current_island_problem)
            archi.push_back(island_to_be_added)

        island_problems[p] = current_island_problems

        list_of_f_dicts = []
        list_of_x_dicts = []
        for i in range(num_gen): # step between which topology steps are executed
            print('Evolving Gen : %i / %i' % (i, num_gen))
            archi.evolve()
            # archi.status
            # archi.wait_check()
            champs_dict_per_gen = {}
            champ_f_dict_per_gen = {}
            for j in range(number_of_islands_list[p]):
                champs_dict_per_gen[j] = archi.get_champions_x()[j]
                champ_f_dict_per_gen[j] = archi.get_champions_f()[j]
            list_of_x_dicts.append(champs_dict_per_gen)
            list_of_f_dicts.append(champ_f_dict_per_gen)
            archi.wait_check()

        list_of_lists_of_x_dicts.append(list_of_x_dicts)
        list_of_lists_of_f_dicts.append(list_of_f_dicts)
        print('Evolution finished')
        # print('Number of islands this iteration', number_of_islands_list[p])
        # print('champ_f_dict_per_gen : ', champ_f_dict_per_gen)
        # print('list_of_f_dicts : ', list_of_f_dicts)

    ### Algorithm for next island generation ###
        champions_x[p] = archi.get_champions_x() # 2, no_of_islands 
        champions_f[p] = archi.get_champions_f()

        # Write mga sequences to evaluated sequence database file
        delta_v = {}
        delta_v_per_leg = {}
        tof = {}
        evaluated_sequences_dict[p] = [[] for _ in range(number_of_islands)]

        for j in range(number_of_islands):
            # define delta v to be used for evaluation of best sequence
            island_problems[p][j].fitness(champions_x[p][j], post_processing=True)
            delta_v[j] = island_problems[p][j].transfer_trajectory_object.delta_v
            delta_v_per_leg[j] = island_problems[p][j].transfer_trajectory_object.delta_v_per_leg
            tof[j]=  island_problems[p][j].transfer_trajectory_object.time_of_flight

            # Save evaluated sequences to database with extra information
            mga_sequence_characters = \
                    util.transfer_body_order_conversion.get_mga_characters_from_list(
                                    temp_evaluated_sequences[j])
            mga_sequence_characters_list.append(mga_sequence_characters)
            current_sequence_result = [mga_sequence_characters, delta_v[j], tof[j] / 86400]
            current_sequence_result_string = " %s, %d, %d\n" % (current_sequence_result[0],
                    current_sequence_result[1], current_sequence_result[2])
            evaluated_sequences_results += current_sequence_result_string
            evaluated_sequences_dict[p][j] = current_sequence_result

            # Save separate leg information
            current_sequence_leg_results = manualTopology.get_leg_specifics(mga_sequence_characters, champions_x[p][j],
                    delta_v_per_leg[j], number_of_free_coefficients=free_param_count)
            leg_results += current_sequence_leg_results

            # check auxiliary help for good legs -> SOLVE PICKLE ERROR
            # we have island_problms in a list
            # deltav per leg weighted average based on how far away the transfer is (EM is stricter
            # than YN

        ### Define ITBS ###
        if p == 0:
            # print(delta_v, temp_ptbs)
            dt_delta_v = delta_v[0]
            dt_sequence = temp_ptbs[0]
            temp_ptbs.pop(0)
            delta_v.pop(0)
            for it in list(delta_v):
                if it == 0:
                    continue
                delta_v[it-1] = delta_v.pop(it)

        # print(delta_v, temp_ptbs)
        current_itbs = manualTopology.get_itbs(dv=delta_v, ptbs=temp_ptbs,
            type_of_selection="proportional", dt_tuple=(dt_delta_v, dt_sequence), pc=planet_characters, pl=planet_list)
        
        # print(current_itbs)
        if p == 0:
            # itbs.append(current_itbs)
            itbs = current_itbs
        else:
            itbs = [itbs[i] + current_itbs[i] for i in range(len(itbs))]

        # print(itbs)
        # for i in range(len(current_itbs)):
        #     itbs[]

        # itbs = []

        # increase size of island population if the direct transfer is found to be optimal
        # if itbs == []:
        #     number_of_sequences_per_planet[p] *= 3
        #     p_bump =  True

        # print(temp_ptbs)

    ##########################################
    # End of predefined sequence length loop #
    ##########################################
    ##########################################

###########################################################################
# Post processing #########################################################
###########################################################################

# Saving the trajectories for post-processing
    if write_results_to_file:
        # with open(evaluated_sequences_database, 'r+') as file_object:
        #     file_object.write(evaluated_sequences_results)
        #     file_object.close()
        with open(evaluated_sequences_database, 'r+') as file_object:
            file_object.write(evaluated_sequences_results)
            file_object.close()

        with open(separate_leg_database, 'r+') as file_object:
            file_object.write(leg_results)
            file_object.close()

    for layer in range(no_of_sequence_recursions): # 2
        layer_folder = '/layer_%i' % (layer)
        champions_dict = {}
        champion_fitness_dict = {}
        # for i in range(len(champions_x[layer])):
        for i in range(number_of_islands_list[layer]):
            # print("Champion: ", champions[i])
            mga_low_thrust_problem = island_problems[layer][i]
            # print(champions_x)
            mga_low_thrust_problem.fitness(champions_x[layer][i], post_processing=True) # 2 is one loop
    
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
                auxiliary_info['MGA Sequence,'] = \
                mga_sequence_characters_list[layer*len(champions_x[layer]) + i]
                auxiliary_info['Maximum thrust'] = np.max([np.linalg.norm(j[1:]) for _, j in
                    enumerate(thrust_acceleration.items())])
                #evaluated_sequences_dict[p][j][0]
    
                unique_identifier = "/islands/island_" + str(i) + "/"
                save2txt(state_history, 'state_history.dat', output_directory + subdirectory +
                        layer_folder + unique_identifier)
                save2txt(thrust_acceleration, 'thrust_acceleration.dat', output_directory +
                        subdirectory + layer_folder + unique_identifier)
                save2txt(node_times, 'node_times.dat', output_directory + subdirectory +
                        layer_folder + unique_identifier)
                save2txt(auxiliary_info, 'auxiliary_info.dat', output_directory + subdirectory +
                        layer_folder + unique_identifier)
                current_island_f = {}
                current_island_x = {}
                for j in range(num_gen):
                    # print(j, i)
                    current_island_f[j] = list_of_lists_of_f_dicts[layer][j][i]
                    current_island_x[j] = list_of_lists_of_x_dicts[layer][j][i]
                save2txt(current_island_f, 'champ_f_per_gen.dat', output_directory
                        +  subdirectory + layer_folder +  unique_identifier)
                save2txt(current_island_x, 'champs_per_gen.dat', output_directory +
                        subdirectory + layer_folder + unique_identifier)

                champions_dict[i] = champions_x[layer][i]
                champion_fitness_dict[i] = champions_f[layer][i]
        
        if write_results_to_file:
            # print(champions_dict)
            # unique_identifier = "/champions/"
            # file_object = open("%schampions.dat" % (output_directory + unique_identifier), 'a+')
            # file_object.write(champions_x[layer][i])
            # file_object.close("%schampions.dat" % (output_directory + unique_identifier))
            #
            # file_object = open("%schampions_fitness.dat" % (output_directory + unique_identifier), 'a+')
            # file_object.write(champions_f[layer][i])
            # file_object.close("%schampions_fitness.dat" % (output_directory + unique_identifier))
            pass
    
