'''
Author: Sean Cowan
Purpose: MSc Thesis
Date Created: 14-09-2022

This module aims at implementing the algorithm that exchanges information and determines what
islands to pass to the archipelago

This file was originally manual_topology.py, but was renamed to mgaso_py for more precise library navigation
'''
###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General 
import numpy as np
import pygmo as pg
import random

# Tudatpy
from tudatpy.io import save2txt

# Local
import core.multipurpose.pygmo_problem as prob
import core.multipurpose.mga_low_thrust_utilities as util
import core.mgaso.leg_mechanics as legs
import core.mgaso.algo_elements as algo

# import sys
# sys.path.insert(0, "/Users/sean/Desktop/tudelft/tudat/tudat-bundle-test/build/tudatpy")

def create_island(iteration, 
                    transfer_body_order, 
                    free_param_count, 
                    bounds, 
                    pop_size,
                    leg_exchange, 
                    leg_database, 
                    manual_base_functions=False, 
                    elitist_fraction=0.1,
                    dynamic_bounds=None,
                    objectives=['dv'],
                    no_of_points=100,
                    manual_tof_bounds=None,
                    Isp=None,
                    m0=None,
                    zero_revs=False):
    """
    Function that can create an island with completely random individuals or with partially
    predefined individuals that were found to be high fitness
    """
    mga_low_thrust_problem = \
    prob.MGALowThrustTrajectoryOptimizationProblemOptimAngles(transfer_body_order=transfer_body_order,
            no_of_free_parameters=free_param_count, 
            bounds=bounds,
            manual_base_functions=manual_base_functions, 
            objectives=objectives, 
            Isp=Isp,
            m0=m0,
            dynamic_bounds=dynamic_bounds,
            no_of_points=no_of_points,
            zero_revs=zero_revs,
            manual_tof_bounds=manual_tof_bounds)

    if len(objectives) == 1:
        algorithm = pg.algorithm(pg.sga(gen=1))
    else:
        algorithm = pg.algorithm(pg.nsga2(gen=1))

    if leg_exchange:
        # This is an artifact from when I was doing Morante2019 multiobjective experiments
        no_of_predefined_individuals = int(pop_size * elitist_fraction) if iteration != 0 else 0
        no_of_random_individuals = pop_size - no_of_predefined_individuals
        pop_size_calc = lambda x, y: x + y
        if pop_size < 5:
            no_of_random_individuals += 5 - pop_size
            pop_size = no_of_random_individuals + no_of_predefined_individuals
            print(f'Number of individuals was too small, increased to {pop_size}')
        modulus_random = no_of_random_individuals % 4
        if modulus_random != 0:
            no_of_random_individuals += (4-modulus_random)
            pop_size = pop_size_calc(no_of_random_individuals, no_of_predefined_individuals)
            print(f"Number of random individuals not divisible by 4, increased by {4-modulus_random} to {no_of_random_individuals}")
        modulus_pre = no_of_predefined_individuals % 4
        modulus_pop = pop_size % 4
        if modulus_pop != 0:
            no_of_predefined_individuals += (4-modulus_pop)
            pop_size = pop_size_calc(no_of_random_individuals, no_of_predefined_individuals)
            print(f'Population size not divisible by 4, increased no_of_predefined_individuals by {4-modulus_pop} to {no_of_predefined_individuals}')
            print(f'Population size is {pop_size_calc(no_of_random_individuals, no_of_predefined_individuals)}')
        # assert pop_size % 4, 0
            # print(f'Number of random individuals increased by {modulus}. no_of_random_individuals : {no_of_random_individuals}') 

        population = pg.population(prob=mga_low_thrust_problem, size=no_of_random_individuals)
        island_mga_characters = \
        util.transfer_body_order_conversion.get_mga_characters_from_list(transfer_body_order)
        island_leg_dict = \
        util.transfer_body_order_conversion.get_dict_of_legs_from_characters(island_mga_characters)

        pre_design_parameter_vectors = []
        for _ in range(no_of_predefined_individuals):
            pre_design_parameter_vectors.append(population.random_decision_vector())
        # for leg_number, leg in enumerate(island_leg_dict): 
        for leg, leg_number in island_leg_dict.items(): 
            if leg_database == []: #exit if 1st iteration where leg_database is empty
                break
            leg_data = legs.legDatabaseMechanics.get_leg_data_from_database(leg, leg_database)
            filtered_legs = legs.legDatabaseMechanics.get_filtered_legs(leg_data,
                    no_of_predefined_individuals)
            pre_design_parameter_vectors = \
            legs.legDatabaseMechanics.get_dpv_from_leg_specifics(pre_design_parameter_vectors,
                    filtered_legs, transfer_body_order, free_param_count, leg_number)
        
            for i in range(no_of_predefined_individuals):
                population.push_back(pre_design_parameter_vectors[i])
    else:
        population = pg.population(prob=mga_low_thrust_problem, size=pop_size)

    return pg.island(algo=algorithm, pop=population, udi=pg.mp_island()), mga_low_thrust_problem

def create_archipelago(iteration,
                       departure_planet,
                       arrival_planet, 
                       free_param_count, 
                       pop_size,
                       bounds,
                       max_no_of_gas,
                       number_of_sequences_per_planet,
                       islands_per_sequence,
                       itbs,
                       planet_list, 
                       leg_exchange,
                       leg_database=None,
                       manual_base_functions=None, 
                       dynamic_bounds=None,
                       elitist_fraction=None,
                       seed=None,
                       objectives=None,
                       evaluated_sequences_chars_list=None,
                       Isp=None,
                       m0=None,
                       zero_revs=None):

    current_max_no_of_gas = max_no_of_gas - iteration
    current_island_problems = []
    temp_evaluated_sequences_chars = []
    # temp_unique_evaluated_sequences_chars = []
    temp_ptbs = []

    no_of_possible_planets = len(planet_list)
    combinations_left = no_of_possible_planets**(max_no_of_gas-iteration)# or no_of_sequence_recursions

    print('Leg exchange enabled') if leg_exchange == True else print('Leg exchange disabled') 

    print(f'Population size : {pop_size}')
    # no_of_predefined_individuals = int(pop_size * elitist_fraction) if iteration != 0 else 0
    # no_of_random_individuals = pop_size - no_of_predefined_individuals
    # print(f'Number of random individuals : {no_of_random_individuals}')
    # print(f'Number of predefined individuals : {no_of_predefined_individuals}')

    print("""
====================
Creating archipelago
====================
          """)
    archi = pg.archipelago(n=0, seed=seed)

    # Add islands with pre-defined sequences
    final_iteration = False
    if iteration == 0:
        number_of_sequences = number_of_sequences_per_planet[iteration]*(len(planet_list)) + 1
        directtransferbump = 1
    else:
        number_of_sequences = number_of_sequences_per_planet[iteration]*(len(planet_list))
        directtransferbump = 0

        # keep number of islands to a maximum if combinatorial space is small enough
        if number_of_sequences >= combinations_left:
            print('Number of sequences from %i to %i' % (number_of_sequences , combinations_left))
            number_of_sequences = combinations_left
            final_iteration = True

    number_of_islands = number_of_sequences * islands_per_sequence
    # number_of_islands = number_of_islands.copy()
    print('Number of islands: ',number_of_islands , '\n')
    for i in range(number_of_sequences):
        if i == 0 and iteration == 0:
            transfer_body_order = [departure_planet, arrival_planet]
            ptbs = [departure_planet]
        elif not final_iteration:
            # [i] so that each island can start with its own predefined
            ptbs = [departure_planet] + itbs + [planet_list[(i-directtransferbump) % len(planet_list)]]
            # can add seed argument but that does not work yet as intended
            random_sequence = create_random_transfer_body_order(
                    possible_planets=planet_list, max_no_of_gas=current_max_no_of_gas)
            transfer_body_order = ptbs + random_sequence + [arrival_planet]

            # Check if the transfer body order is unique
            it = 0
            while not check_uniqueness(transfer_body_order, evaluated_sequences_chars_list,
                                                  temp_evaluated_sequences_chars):
                random_sequence = create_random_transfer_body_order(
                        possible_planets=planet_list, max_no_of_gas=current_max_no_of_gas)
                transfer_body_order = ptbs + random_sequence + [arrival_planet]
                it+=1
                if it>20:
                    raise RuntimeError("There are only duplicate transfers left over. Something is wrong.")
        else:
            ptbs = [departure_planet] + itbs + [planet_list[(i-directtransferbump) % len(planet_list)]]
            transfer_body_order = ptbs + [arrival_planet]
            if not check_uniqueness(transfer_body_order, evaluated_sequences_chars_list,
                                                  temp_evaluated_sequences_chars):
                print(f"Island skipped because not unique and final iteration: {transfer_body_order}")
                number_of_islands -= islands_per_sequence
                continue #skip island


        temp_ptbs.append(ptbs)
        print(transfer_body_order)
        # temp_unique_evaluated_sequences_chars.append( \
        #         util.transfer_body_order_conversion.get_mga_characters_from_list(transfer_body_order))
        for j in range(islands_per_sequence):
            temp_evaluated_sequences_chars.append( \
                    util.transfer_body_order_conversion.get_mga_characters_from_list(transfer_body_order))
            island_to_be_added, current_island_problem = \
            create_island(iteration=iteration, 
                                        transfer_body_order=transfer_body_order,
                                        free_param_count=free_param_count, 
                                        bounds=bounds, 
                                        pop_size=pop_size, 
                                        leg_exchange=leg_exchange, 
                                        leg_database=leg_database,
                                        manual_base_functions=manual_base_functions, 
                                        elitist_fraction=elitist_fraction, 
                                        dynamic_bounds=dynamic_bounds,
                                        objectives=objectives,
                                        Isp=Isp, 
                                        m0=m0, 
                                        zero_revs=zero_revs)

            print(f'Island {j+1}/{islands_per_sequence} added', end='\r')
            current_island_problems.append(current_island_problem)
            archi.push_back(island_to_be_added)

    assert number_of_islands == len(current_island_problems)

    return temp_ptbs, temp_evaluated_sequences_chars, number_of_islands, current_island_problems, archi

def check_uniqueness(tbo, evaluated_sequences_chars_list, temp_evaluated_sequences_chars):
    tbo_chars = util.transfer_body_order_conversion.get_mga_characters_from_list(tbo) 
    for i in evaluated_sequences_chars_list + temp_evaluated_sequences_chars:
        if tbo_chars == i:
            print(f'\n{tbo_chars} is not unique')
            return False
    print(f'\n{tbo_chars} is unique')
    return True


def determine_itbs(p, evaluated_sequences_results=None, evaluated_sequences_results_dict=None,
                   temp_evaluated_sequences_chars=None, evaluated_sequences_database=None, leg_database=None,
                   leg_results=None, number_of_islands_array=None, number_of_sequences_array=None,
                   islands_per_sequence_array=None, island_problems=None, champions_x=None, output_directory=None,
                   subdirectory=None, itbs=None, fitness_proportion=1.0, compute_mass=False, max_no_of_gas=None,
                   Isp=None, m0=None, write_results_to_file=False, no_of_points=100):

    # Write mga sequences to evaluated sequence database file
    evaluated_sequences_results_dict[p] = [[] for _ in range(number_of_islands_array[p])]
    min_delta_v_per_sequence = {} # per layer
    mean_delta_v_per_sequence = {} # per layer

    fitness_per_sequence_calc = lambda min_delta_v, mean_delta_v, fitness_proportion : fitness_proportion * min_delta_v \
    + (1-fitness_proportion) * mean_delta_v

    # Get previously evaluated sequences
    fitness_per_sequence_prev = {}
    if p != 0:
        for i in evaluated_sequences_database:
            sequence_from_other_layer = util.transfer_body_order_conversion.get_mga_list_from_characters(i[0])
            if len(sequence_from_other_layer) > p+2:
                print(itbs, sequence_from_other_layer)
                if itbs[p-1] == sequence_from_other_layer[p]:
                    print(f'Added {i[0]} from another layer to fitness evaluation')
                    fitness_per_sequence_prev[i[0]] = fitness_per_sequence_calc(i[1], i[2], fitness_proportion)

    # Save unique sequences
    temp_unique_evaluated_sequences_chars = list(dict.fromkeys(temp_evaluated_sequences_chars))

    island_number = 0
    for i in range(number_of_sequences_array[p]):
        delta_v = []
        delta_v_per_leg = []
        tof = []
        delivery_masses = []
        current_sequence_chars = temp_unique_evaluated_sequences_chars[i]

        for j in range(islands_per_sequence_array[p]):
            # if len(objectives) == 2:
            #     champions_x[p][j] = champions_x[p][j][0]
            # define delta v to be used for evaluation of best sequence
            # for MO, the 0 indicates that the best dV is chosen for the database

            island_problems[p][island_number].fitness(champions_x[p][island_number], post_processing=True)
            delta_v.append(island_problems[p][island_number].transfer_trajectory_object.delta_v)
            delta_v_per_leg.append(island_problems[p][island_number].transfer_trajectory_object.delta_v_per_leg)
            tof.append(island_problems[p][island_number].transfer_trajectory_object.time_of_flight)

            if compute_mass:
                thrust_acceleration = \
                    island_problems[p][island_number].transfer_trajectory_object.inertial_thrust_accelerations_along_trajectory(no_of_points)
                _, delivery_mass, _= algo.get_mass_propagation(thrust_acceleration, Isp, m0)
                delivery_masses.append(delivery_mass)


            # Save separate leg information
            if leg_results != None:
                current_sequence_leg_mechanics_object = legs.separateLegMechanics(current_sequence_chars,
                        champions_x[p][island_number], delta_v_per_leg[j])
                current_sequence_leg_results = \
                current_sequence_leg_mechanics_object.get_sequence_leg_specifics()
                for leg in current_sequence_leg_results:
                    leg_results += "%s, %d, %d, %d\n" % tuple(leg) # only values
                    leg_database.append(leg)

            island_number += 1

        # Statistics on objective
        mean_delta_v_per_sequence[current_sequence_chars] = np.mean(delta_v)
        min_delta_v_per_sequence[current_sequence_chars] = np.min(delta_v)
        index_min_delta_v = np.argmin(delta_v)
        if islands_per_sequence_array[p] == 1:
            assert np.mean(delta_v) == np.min(delta_v)

        # Save chars, dv, and tof of best of each islands
        current_sequence_result = [current_sequence_chars, min_delta_v_per_sequence[current_sequence_chars] ,
                                   mean_delta_v_per_sequence[current_sequence_chars] , tof[index_min_delta_v] / 86400]
        current_sequence_result_string = "%s, %d, %d, %d\n" % tuple(current_sequence_result)
        if compute_mass:
            current_sequence_result_string.replace('\n', f', {delivery_masses[index_min_delta_v] / m0}\n')
            current_sequence_result.append(delivery_masses[index_min_delta_v] / m0)
        evaluated_sequences_results += current_sequence_result_string
        evaluated_sequences_database.append(current_sequence_result)

        evaluated_sequences_results_dict[p][i] = current_sequence_result

    if max_no_of_gas == 0:
        raise RuntimeError("The maximum number of gas allowed is 0. Something is wrong.")

    fitness_per_sequence = {}
    for i in range(number_of_sequences_array[p]):
        if p == 0 and i == 0:
            sequence_dt = temp_unique_evaluated_sequences_chars[0]
            min_delta_v_dt = {sequence_dt : min_delta_v_per_sequence[sequence_dt]}
            mean_delta_v_dt = {sequence_dt : mean_delta_v_per_sequence[sequence_dt]}

            if write_results_to_file:
                save2txt(min_delta_v_dt, 'min_dv_dt.dat', output_directory + subdirectory + '/')
                save2txt(mean_delta_v_dt, 'mean_dv_dt.dat', output_directory + subdirectory + '/')

            min_delta_v_per_sequence.pop(sequence_dt)
            mean_delta_v_per_sequence.pop(sequence_dt)
            temp_unique_evaluated_sequences_chars.pop(0)
            continue

        # fitness_per_sequence[temp_unique_evaluated_sequences_chars[i-1]] = fitness_proportion * \
        #         min_delta_v_per_sequence[temp_unique_evaluated_sequences_chars[i-1]] \ + (1-fitness_proportion) * \
        #         mean_delta_v_per_sequence[temp_unique_evaluated_sequences_chars[i-1]]
        fitness_per_sequence[temp_unique_evaluated_sequences_chars[i-1]] = \
        fitness_per_sequence_calc(min_delta_v_per_sequence[temp_unique_evaluated_sequences_chars[i-1]],
                                  mean_delta_v_per_sequence[temp_unique_evaluated_sequences_chars[i-1]],
                                  fitness_proportion) 
        if fitness_proportion == 1.0:
            assert int(fitness_per_sequence[temp_unique_evaluated_sequences_chars[i-1]]) == \
            int(min_delta_v_per_sequence[temp_unique_evaluated_sequences_chars[i-1]])

    if write_results_to_file:
        save2txt(min_delta_v_per_sequence, 'min_dv_per_sequence.dat', output_directory + subdirectory + f'/layer_{p}/')
        save2txt(mean_delta_v_per_sequence, 'mean_dv_per_sequence.dat', output_directory + subdirectory + f'/layer_{p}/')
        save2txt(fitness_per_sequence, 'fitness_per_sequence.dat', output_directory + subdirectory + f'/layer_{p}/')

    # Add relevant sequences from other layers
    # Not super efficient, one could create multiple databases based on each layer of the 
    # A class for the database of sequences and a class for each transfer, for example
    fitness_per_sequence.update(fitness_per_sequence_prev)

    #Determine best fitness and apply that to next itbs
    best_sequence = min(fitness_per_sequence, key=fitness_per_sequence.get)
    # best_sequence_value = min(fitness_per_sequence.values())

    current_itbs = util.transfer_body_order_conversion.get_mga_list_from_characters(best_sequence)[p+1]
    itbs.append(current_itbs)
    print(f'Initial Target Body Sequence : {itbs}')

    return itbs, evaluated_sequences_results, evaluated_sequences_database, leg_results, leg_database, delivery_masses

def create_random_transfer_body_order(possible_planets, seed=None, max_no_of_gas=6) -> list:

    body_dict = {0: "Null",
            1: "Mercury",
            2: "Venus",
            3: "Earth",
            4: "Mars",
            5: "Jupiter",
            6: "Saturn",
            7: "Uranus",
            8: "Neptune"}

    # transfer_body_order.append(predefined_sequence) if predefined_sequence != [] else None #?
    # create random sequence of numbers with max_no_of_gas length
    random.seed(seed)
    sequence_length = random.randrange(0, max_no_of_gas)
    sequence_digits = [random.randrange(0, len(possible_planets)) for _ in
            range(sequence_length)]
    transfer_body_list = [possible_planets[i] for i in sequence_digits]

    # transform that into transfer_body_order
    # transfer_body_list = \
    # util.transfer_body_order_conversion.get_transfer_body_list(sequence_digits)
    return transfer_body_list

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

    if arr_index < dep_index: #transfer to inner planet
        number_of_truncations = len(planet_list) - dep_index# - 1 #do not include final target
        # print(number_of_truncations)
    else: #transfer to outer planet
        number_of_truncations = len(planet_list) - arr_index# - 1
        # print(number_of_truncations)
    for i in range(number_of_truncations):
        planet_list.pop()
    return planet_list

# def get_itbs(dv : dict=None, ptbs : dict=None, type_of_selection : str="max", pc : list = None,
#         pl : list = None, dt_tuple : tuple=(None, None)):
#     if type_of_selection == "max":
#         best_key = 0
#         best_value = dv[0]
#         for key, value in dv.items():
#             if value < best_value:
#                 best_value = value
#                 best_key = key
#         locally_best_sequence = ptbs[best_key]
#         locally_best_sequence.pop(0)
#         return locally_best_sequence
#     if type_of_selection == "proportional":
#
#         ### Get results into lists
#         sequence_delta_v_value_dict = {}
#         no_of_possible_planets = len(pl)
#         for i in range(no_of_possible_planets): # for each possible planet
#             current_planet_char = pc[i % len(pc)]
#             list_of_delta_v_for_current_pc = []
#             for it, j in enumerate(ptbs): # go check all indices 
#                 ptbs_characters = \
#                 util.transfer_body_order_conversion.get_mga_characters_from_list(j)
#                 last_character = ptbs_characters[-1]
#                 if last_character == current_planet_char: # if last value is j
#                     list_of_delta_v_for_current_pc.append(dv[it]) # +1 because direct transfer removed
#
#                     ### Make search function recursive, to not double check
#                     # make use of list(delta_v.keys()) to change the keys of dict
#                     # ptbs.pop(it) # remove this item fromdv and ptbs
#                     # dv.pop(it)
#                     # for it2 in dv.keys():
#                     #     if it2 == 0:
#                     #         continue
#                     #     dv[it2-1] = dv.pop(it2)
#
#             sequence_delta_v_value_dict[i] = list_of_delta_v_for_current_pc
#
#         # print('all_values', sequence_delta_v_value_dict)
#         ### Get statistics
#         quantities = ['max', 'min', 'mean', 'median']
#         ptbs_stats = np.zeros((no_of_possible_planets, len(quantities))) 
#         for i, j in sequence_delta_v_value_dict.items():
#             current_max = np.max(j)
#             current_min = np.min(j)
#             current_mean = np.mean(j)
#             current_median = np.median(j)
#             # print(current_max, current_min, current_mean, current_median)
#             ptbs_stats[i, :] = np.array([current_max, current_min, current_mean,
#                 current_median])
#         # print('ptbs_stats', ptbs_stats)
#
#         ### Determine scalarization (what is an optimal combination of statistical quantities
#         arg_min = np.argmin(ptbs_stats[:,1]) # take minimum
#         local_char = pc[arg_min]
#         itbs = [util.transfer_body_order_conversion.get_mga_list_from_characters(local_char) for _ in
#             range (no_of_possible_planets)]
#         # locally_best_sequence = [pc[ for _ in range(no_of_possible_pl
#
#         return itbs

