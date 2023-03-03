'''
Author: Sean Cowan
Purpose: MSc Thesis
Date Created: 31-01-2023

This file includes two functions: One that performs a local optimisation of a 
'''
###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General 
import numpy as np
import os
import pygmo as pg
import multiprocessing as mp
import random

# Local
import core.multipurpose.mga_low_thrust_utilities as util
import core.mgaso.algo_elements as core
import core.multipurpose.create_files as post
import core.multipurpose.perform_evolution as evol

# import sys
# sys.path.insert(0, "/Users/sean/Desktop/tudelft/tudat/tudat-bundle-test/build/tudatpy")


def run_mgaso_optimisation(departure_planet : str,
                            arrival_planet : str,
                            free_param_count : int,
                            Isp : int = None,
                            m0 : int = None,
                            num_gen : int = 2,
                            pop_size : int = 100,
                            no_of_points : int = 100,
                            bounds : list = None,
                            output_directory : str = '',
                            subdirectory : str = '',
                            possible_ga_planets : list = None,
                            max_no_of_gas = 1,
                            no_of_sequence_recursions = 1,
                            islands_per_sequence=1,
                            elitist_fraction=0.1,
                            number_of_sequences_per_planet = None,
                            fraction_ss_evaluated = None,
                            seed : int = 421,
                            write_results_to_file=False,
                            manual_base_functions=False,
                            dynamic_bounds=None,
                            leg_exchange = False,
                            top_x_sequences = 10, # for what legs to replace
                            objectives=['dv'],
                            zero_revs=False,
                            fitness_proportion=1.0,
                            fitprop_itbs=1.0,
                            topology_weight=0.01):


    # if os.path.exists(output_directory + subdirectory):
    #     shutil.rmtree(output_directory + subdirectory)
    compute_mass = False
    if any(x == 'pmf' or x == 'dm' or x == 'dmf' for x in objectives):
        compute_mass = True

    ### Determine possible flyby planets
    if possible_ga_planets != None:
        planet_list = possible_ga_planets
        planet_characters = \
        util.transfer_body_order_conversion.get_mga_character_list_from_list(planet_list)
        print(f'Possible GA planets constrained to {[i for i in planet_list]}')
    else:
        planet_characters = ['Y', 'V', 'E', 'M', 'J', 'S', 'U', 'N']
        planet_list = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
        # Remove excess planets above target planet (except for depature)
        planet_list = core.remove_excess_planets(planet_list, departure_planet,
            arrival_planet) if max_no_of_gas != 0 else [0]
        planet_characters = core.remove_excess_planets(planet_characters, departure_planet, arrival_planet)
        print(f'GA planets limited to planets within {arrival_planet}')


    unique_identifier = '/topology_database'

    # TODO : replace with list of lists
    
    ### Create empty files that are written to later
    if write_results_to_file:
        evaluated_seq_database_file = output_directory + subdirectory + unique_identifier +  '/evaluated_sequences_database.txt'
        sorted_evaluated_seq_database_file = output_directory + subdirectory + '/sorted_evaluated_sequences_database.txt'
        
        if not os.path.exists(output_directory + subdirectory + unique_identifier):
            os.makedirs(output_directory + subdirectory + unique_identifier)

        if not os.path.exists(evaluated_seq_database_file):
            open(evaluated_seq_database_file, 'a+').close()
        else:
            os.remove(evaluated_seq_database_file)

        if not os.path.exists(sorted_evaluated_seq_database_file):
            open(sorted_evaluated_seq_database_file, 'a+').close()
        else:
            os.remove(sorted_evaluated_seq_database_file)

        if leg_exchange:
            leg_database_file = output_directory + subdirectory +  unique_identifier +  '/leg_database.txt'
            if not os.path.exists(leg_database_file):
                open(leg_database_file, 'a+').close()
            else:
                os.remove(leg_database_file)

    # evaluated_sequences_database = [[], []]
    # leg_database = [[], []]
    evaluated_sequences_database = []
    leg_database = [] if leg_exchange else None

    mp.freeze_support()
    cpu_count = os.cpu_count() # not very relevant because differnent machines + asynchronous
    # cpu_count = len(os.sched_getaffinity(0))


    # print(len(planet_list)
    evaluated_sequences_results = 'Sequence, Min Delta V, Mean Delta V, ToF\n'
    mass_dict = {'dmf' : 'Delivery Mass Fraction', 'pmf' : 'Propellant Mass Fraction'}
    if compute_mass:
        evaluated_sequences_results.replace('\n', f', {mass_dict[objectives[0]]}')
    leg_results = 'Leg, Delta V, ToF, #rev\n' if leg_exchange else None

    # border case for max_no_of_gas == 0
    # range_no_of_sequence_recursions = [i for i in range(no_of_sequence_recursions)]
    if max_no_of_gas == 0:
    #     no_of_sequence_recursions = 1
    #     range_no_of_sequence_recursions = [0]
        number_of_sequences_per_planet = [0]

    if fraction_ss_evaluated[0] != 0:
        number_of_sequences_per_planet = []

    # variable definitions
    champions_x = {}
    champions_f = {}
    island_problems = {}
    evaluated_sequences_results_dict = {}
    evaluated_sequences_chars_list = []
    itbs = []
    number_of_islands_array = np.zeros(no_of_sequence_recursions+1, dtype=int) # list of number of islands per recursion
    islands_per_sequence_array = np.zeros(no_of_sequence_recursions+1, dtype=int) # list of number of islands per recursion
    number_of_sequences_array = np.zeros(no_of_sequence_recursions+1, dtype=int) # list of number of islands per recursion
    list_of_lists_of_x_dicts = []# list containing list of dicts of champs per gen
    list_of_lists_of_f_dicts = []
    
###########################################################################
# MGASO Optimisation ###############################################
###########################################################################
    
    combinations_remaining_lambda = lambda planet_list, max_no_of_gas, p : len(planet_list)**(max_no_of_gas-p) + 1 if p == 0 \
        else len(planet_list)**(max_no_of_gas-p)# or no_of_sequence_recursions

    combinations_evaluated_lambda = lambda spp, planet_list, p: spp*len(planet_list) + 1 if p == 0 else spp*len(planet_list)
    # Loop for number of sequence recursions
    # p_bump = False
    for p in range(no_of_sequence_recursions): # gonna be max_no_of_gas
        print('Iteration: ', p, '\n')

        # Projected combinatorial complexity covered
        combinations_remaining = combinations_remaining_lambda(planet_list, max_no_of_gas, p)# or no_of_sequence_recursions

        current_seed = seed[p]

        if fraction_ss_evaluated[p] != 0:
            to_be_evaluated_sequences_current_layer = int(fraction_ss_evaluated[p] * combinations_remaining)
            # print(f'tobeevaluated : {to_be_evaluated_sequences_current_layer}')
            if to_be_evaluated_sequences_current_layer // len(planet_list) == 0:
                number_of_sequences_per_planet.append(1)
            else:
                number_of_sequences_per_planet.append(to_be_evaluated_sequences_current_layer // len(planet_list))
        else:
            fraction_ss_evaluated[p] = (number_of_sequences_per_planet[p]) * len(planet_list) / combinations_remaining
        print(f'Fraction of sequences evaluated in this recursion: {fraction_ss_evaluated[p]}')
        print(f'Absolute number of sequences per target planet evaluated in this recursion : {number_of_sequences_per_planet[p]}')


    
        # Number of sequences to be evaluated in this recursion
        # 1 in the line below because the number of sequences is constant
        combinations_evaluated = combinations_evaluated_lambda(number_of_sequences_per_planet[p], planet_list, p)
        print(f'The combinational coverage that is to be achieved in this layer is {combinations_evaluated} / {combinations_remaining}')

        # Creation of the archipelago
        temp_ptbs, temp_evaluated_sequences_chars, number_of_islands, current_island_problems, archi = \
        core.create_archipelago(p, departure_planet, arrival_planet, free_param_count, pop_size, bounds, max_no_of_gas,
                                number_of_sequences_per_planet, islands_per_sequence, itbs, planet_list, leg_exchange,
                                leg_database=leg_database,
                                manual_base_functions=manual_base_functions, dynamic_bounds=dynamic_bounds,
                                elitist_fraction=elitist_fraction, seed=current_seed, objectives=objectives,
                                evaluated_sequences_chars_list=evaluated_sequences_chars_list, Isp=Isp, m0=Isp,
                                zero_revs=zero_revs, topology_weight=topology_weight)
        if p == 0:
            archi_info = archi

        evaluated_sequences_chars_list += temp_evaluated_sequences_chars
        number_of_islands_array[p] = number_of_islands
        islands_per_sequence_array[p] = islands_per_sequence
        number_of_sequences_array[p] = number_of_islands / islands_per_sequence
        # print(f'number of sequences this layer: {number_of_sequences_array[p]}')
        island_problems[p] = current_island_problems # dict for recursion, list of all isalnds
        # print(f'temp_evaluated_sequences : {temp_evaluated_sequences_chars}')

        list_of_x_dicts, list_of_f_dicts, champions_x[p], \
        champions_f[p], ndf_x, ndf_f = evol.perform_evolution(archi,
                            number_of_islands_array[p],
                            num_gen,
                            objectives)
        # print(list_of_x_dicts, list_of_f_dicts)
        # print(champions_x[p], champions_f[p])

        list_of_lists_of_x_dicts.append(list_of_x_dicts)
        list_of_lists_of_f_dicts.append(list_of_f_dicts)
        # print('Number of islands this iteration', number_of_islands_array[p])
        # print('champ_f_dict_per_gen : ', champ_f_dict_per_gen)
        # print('list_of_f_dicts : ', list_of_f_dicts)

    ### Algorithm for next island generation ###
        itbs, evaluated_sequences_results, evaluated_sequences_database, leg_results, leg_database, delivery_masses = \
        core.determine_itbs(p, evaluated_sequences_results=evaluated_sequences_results,
                            evaluated_sequences_results_dict=evaluated_sequences_results_dict,
                            temp_evaluated_sequences_chars=temp_evaluated_sequences_chars,
                            evaluated_sequences_database=evaluated_sequences_database,
                            leg_database=leg_database, leg_results=leg_results,
                            number_of_islands_array=number_of_islands_array,
                            number_of_sequences_array=number_of_sequences_array,
                            islands_per_sequence_array=islands_per_sequence_array, island_problems=island_problems,
                            champions_x=champions_x, output_directory=output_directory, subdirectory=subdirectory,
                            itbs=itbs, fitness_proportion=fitness_proportion, compute_mass=compute_mass,
                            max_no_of_gas=max_no_of_gas, write_results_to_file=write_results_to_file,
                            no_of_points=no_of_points, planet_chars=planet_characters,
                            fitness_proportion_itbs=fitprop_itbs)

    #################
    # End of p loop #
    #################
    print(f'Initial Target Body Sequence : {itbs}')


###########################################################################
# Post processing #########################################################
###########################################################################

# Saving the trajectories for post-processing
    if write_results_to_file:
        with open(evaluated_seq_database_file, 'w') as file_object:
            # print(evaluated_sequences_database)
            # json.dump(evaluated_sequences_results, file_object)
            file_object.write(evaluated_sequences_results)
            file_object.close()

        if leg_exchange:
            with open(leg_database_file, 'w') as file_object:
                # json.dump(leg_results, file_object)
                file_object.write(leg_results)
                file_object.close()

        # evaluate sorted database
        unsorted_evaluated_sequences_database = evaluated_sequences_database.copy()
        unsorted_evaluated_sequences_database.sort(key=lambda elem : elem[1])
        sorted_evaluated_sequences_database = unsorted_evaluated_sequences_database.copy()

        # sorted_evaluated_sequences_results = 'Sequence, Delta V, ToF, Delivery Mass Fraction\n'
        sorted_evaluated_sequences_results = 'Sequence, Min Delta V, Mean Delta V, ToF\n'
        mass_dict = {'dmf' : 'Delivery Mass Fraction', 'pmf' : 'Propellant Mass Fraction'}
        if compute_mass:
            sorted_evaluated_sequences_results.replace('\n', f', {mass_dict[objectives[0]]}')

        for it, i in enumerate(sorted_evaluated_sequences_database):
            sorted_evaluated_sequences_results += "%s, %d, %d, %d\n" % tuple(i)
            if compute_mass:
                sorted_evaluated_sequences_results.replace('\n', f', {delivery_masses[it] / m0}\n')

        with open(sorted_evaluated_seq_database_file, 'w') as file_object:
            file_object.write(sorted_evaluated_sequences_results)
            file_object.close()

        post.create_files(type_of_optimisation='mgaso',
                            no_of_sequence_recursions=no_of_sequence_recursions,
                            number_of_islands_array=number_of_islands_array,
                            islands_per_sequence_array=islands_per_sequence_array,
                            number_of_sequences_array=number_of_sequences_array,
                            island_problems=island_problems,
                            champions_x=champions_x,
                            champions_f=champions_f,
                            list_of_lists_of_f_dicts=list_of_lists_of_f_dicts,
                            list_of_lists_of_x_dicts=list_of_lists_of_x_dicts,
                            no_of_points=no_of_points,
                            Isp=Isp, 
                            m0=m0,
                            unsorted_evaluated_sequences_database=unsorted_evaluated_sequences_database,
                            output_directory=output_directory,
                            subdirectory=subdirectory,
                            free_param_count=free_param_count,
                            num_gen=num_gen,
                            pop_size=pop_size,
                            cpu_count=cpu_count,
                            bounds=bounds,
                            archi=archi_info,
                            fraction_ss_evaluated=fraction_ss_evaluated,
                            fitprop=fitness_proportion,
                            fitprop_itbs=fitprop_itbs,
                            number_of_sequences_per_planet=number_of_sequences_per_planet,
                            planet_list=planet_list,
                            itbs=itbs)

