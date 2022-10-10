'''
Author: Sean Cowan
Purpose: MSc Thesis
Date Created: 26-07-2022

This module performs the optimization calculations using the help modules from mga-low-thrust-utilities.py and
pygmo-utilities.py
'''

if __name__ == '__main__': #to prevent this code from running if this file is not the source file.
# https://stackoverflow.com/questions/419163/what-does-if-name-main-do

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
    from tudatpy.kernel.astro import time_conversion
    from tudatpy.kernel.numerical_simulation import propagation_setup
    from tudatpy.kernel.numerical_simulation import environment_setup
    from tudatpy.kernel.math import interpolators
    from tudatpy.kernel.astro import element_conversion
    from tudatpy.kernel.trajectory_design import shape_based_thrust
    from tudatpy.kernel.trajectory_design import transfer_trajectory
    from tudatpy.kernel.interface import spice
    spice.load_standard_kernels()
    
    
    from pygmo_problem import MGALowThrustTrajectoryOptimizationProblem
    import mga_low_thrust_utilities as util
    import manual_topology as topo
    
    current_dir = os.getcwd()
    output_directory = current_dir + '/pp_mgso'
    write_results_to_file = True
    
###########################################################################
# General parameters ######################################################
###########################################################################
    
    """
    Number of generations increases the amount of time spent in parallel computing the different islands
    Number of evolutions requires redefining the islands which requires more time on single thread.
    Population size; unknown
    """
    
    julian_day = constants.JULIAN_DAY
    
    seed = 421
    
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

####################################################################
# MGSO Problem Setup ###############################################
####################################################################
    
    
    # mgso General parameters
    max_no_of_gas = 6
    no_of_sequence_recursions = 1
    max_number_of_exchange_generations = 1
    number_of_sequences_per_planet = [1 for _ in range(max_no_of_gas)]
    # number_of_sequences_per_planet = [1 for i in range(max_no_of_gas)]
    
    ## Specific parameters
    departure_planet = "Earth"
    arrival_planet = "Jupiter"
    free_param_count = 0
    num_gen = 2
    pop_size = 100
    no_of_points = 1000
    bounds = [[1000*julian_day, 1, 50*julian_day, -10**6, 0],
            [1200*julian_day, 200, 2000*julian_day, 10**6, 6]]
    # print('Departure date bounds : [%d, %d]' %
    #         (time_conversion.julian_day_to_calendar_date(1000),
    #     time_conversion.julian_day_to_calendar_date(1200)))
    subdirectory = '/island_testing'
    # num_gen = 1
    # pop_size = 100
    # no_of_points = 1000
    # bounds = [[10000*julian_day, 100, 50*julian_day, -10**6, 0],
    #         [10000*julian_day, 100, 2000*julian_day, 10**6, 6]]
    # subdirectory = '/island_testing/'
    
    planet_characters = ['Y', 'V', 'E', 'M', 'J', 'S', 'U', 'N']
    planet_list = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
    
    # Remove excess planets above target planet (except for depature)
    planet_list = topo.manualTopology.remove_excess_planets(planet_list, departure_planet, arrival_planet)
    unique_identifier = '/topology_database'

    evaluated_sequences_database = output_directory + unique_identifier +  '/evaluated_sequences_database.txt'
    separate_leg_database = output_directory + unique_identifier +  '/separate_leg_database.txt'
    
    if not os.path.exists(evaluated_sequences_database):
        open(evaluated_sequences_database, 'w').close()
    if not os.path.exists(separate_leg_database):
        open(separate_leg_database, 'w').close()

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
    
###########################################################################
# MGSO Optimisation ###############################################
###########################################################################
    
    # Loop for number of sequence recursions
    p_bump = False
    for p in range(no_of_sequence_recursions): # gonna be max_no_of_gas
        if p_bump == True:
            p -= 1
            p_bump = False
        print('Iteration: ', p, '\n')
        # Size of random sequence decreases
        current_max_no_of_gas = max_no_of_gas - p
        current_island_problems = []
        temp_evaluated_sequences = []
        temp_predefined_sequences = []
        print('Creating archipelago')
        archi = pg.archipelago(n=0, seed=seed)

        # Add islands with pre-defined sequences
        directtransferbump = 0
        if p == 0:
            number_of_islands = number_of_sequences_per_planet[p]*len(planet_list) + 1
            transfer_body_order = [departure_planet, arrival_planet]
            temp_evaluated_sequences.append(transfer_body_order) 
            temp_predefined_sequences.append([departure_planet])
            print(transfer_body_order)
            island_to_be_added, current_island_problem = \
            topo.manualTopology.create_island(transfer_body_order=transfer_body_order,
                    free_param_count=free_param_count, bounds=bounds, num_gen=num_gen,
                    pop_size=pop_size)
            current_island_problems.append(current_island_problem)
            archi.push_back(island_to_be_added)
            directtransferbump = 1
        else:
            number_of_islands = number_of_sequences_per_planet[p]*len(planet_list)

        print('Number of islands: ', number_of_islands, '\n')
        for i in range(number_of_islands - directtransferbump):
            predefined_sequence = [departure_planet]
            random_sequence = []
            random_sequence = \
                    topo.manualTopology.create_random_transfer_body_order(
                            arrival_planet=arrival_planet, seed=seed, max_no_of_gas=current_max_no_of_gas)

            if p == 0:
                predefined_sequence += [planet_list[i % len(planet_list)]]
            else:
                predefined_sequence += locally_best_sequence + [planet_list[i % len(planet_list)]]

            temp_predefined_sequences.append(predefined_sequence)
            transfer_body_order = predefined_sequence + random_sequence
            temp_evaluated_sequences.append(transfer_body_order) 
            # print(temp_evaluated_sequences)

            print(transfer_body_order)
            island_to_be_added, current_island_problem = \
            topo.manualTopology.create_island(transfer_body_order=transfer_body_order,
                    free_param_count=free_param_count, bounds=bounds, num_gen=num_gen,
                    pop_size=pop_size)
            current_island_problems.append(current_island_problem)
            archi.push_back(island_to_be_added)

        island_problems[p] = current_island_problems
        # print(current_island_problems)
        # Evolve all islands
        for _ in range(max_number_of_exchange_generations): # step between which topology steps are executed
            print('Evolving ..')
            archi.evolve()
        archi.wait_check()
        print('Evolution finished')

    ### Algorithm for next island generation ###

        champions_x[p] = archi.get_champions_x()
        # print('champion_x[p] type', type(champions_x[p]))
        champions_f[p] = archi.get_champions_f()
        # print(champion_f)

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
            # print('delta_v_per_leg type', type(delta_v_per_leg[j]))
            # print('delta_v_per_leg', delta_v_per_leg[j])

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
            current_sequence_leg_results = topo.manualTopology.get_leg_specifics(mga_sequence_characters, champions_x[p][j],
                    delta_v_per_leg[j])
            leg_results += current_sequence_leg_results

            # check auxiliary help for good legs -> SOLVE PICKLE ERROR
            # we have island_problms in a list
            # deltav per leg weighted average based on how far away the transfer is (EM is stricter
            # than YN

        # print(evaluated_sequences_dict)
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

        # increase size of island population if the direct transfer is found to be optimal
        # if locally_best_sequence == []:
        #     number_of_sequences_per_planet[p] *= 3
        #     p_bump =  True

        # print(temp_predefined_sequences)

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
        file_object = open(evaluated_sequences_database, 'r+')
        file_object.write(evaluated_sequences_results)
        file_object.close()

        with open(separate_leg_database, 'r+') as file_object:
            file_object.write(leg_results)
            file_object.close()

    layer_to_analyse = 0
    champions_dict = {}
    champion_fitness_dict = {}
    for i in range(len(champions_x[layer_to_analyse])):
        # print("Champion: ", champions[i])
        mga_low_thrust_problem = island_problems[layer_to_analyse][i]
        # print(champions_x)
        mga_low_thrust_problem.fitness(champions_x[layer_to_analyse][i], post_processing=True) # 2 is one loop

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
            mga_sequence_characters_list[layer_to_analyse*len(champions_x[layer_to_analyse]) + i]
            #evaluated_sequences_dict[p][j][0]

            unique_identifier = "/island_" + str(i) + "/"
            save2txt(state_history, 'state_history.dat', output_directory + subdirectory +
                    unique_identifier)
            save2txt(thrust_acceleration, 'thrust_acceleration.dat', output_directory + subdirectory
                    + unique_identifier)
            save2txt(node_times, 'node_times.dat', output_directory + subdirectory + unique_identifier)
            save2txt(auxiliary_info, 'auxiliary_info.dat', output_directory + subdirectory +
                    unique_identifier)

            champions_dict[i] = champions_x[layer_to_analyse][i]
            champion_fitness_dict[i] = champions_f[layer_to_analyse][i]
    
    if write_results_to_file:
        # print(champions_dict)
        # unique_identifier = "/champions/"
        # file_object = open("%schampions.dat" % (output_directory + unique_identifier), 'a+')
        # file_object.write(champions_x[layer_to_analyse][i])
        # file_object.close("%schampions.dat" % (output_directory + unique_identifier))
        #
        # file_object = open("%schampions_fitness.dat" % (output_directory + unique_identifier), 'a+')
        # file_object.write(champions_f[layer_to_analyse][i])
        # file_object.close("%schampions_fitness.dat" % (output_directory + unique_identifier))
        pass
