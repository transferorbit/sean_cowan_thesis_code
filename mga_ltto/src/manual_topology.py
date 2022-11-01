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
import shutil
import json

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

class manualTopology:

    def __init__(self) -> None:
        pass

    # @staticmethod
    # def create_island(transfer_body_order, free_param_count, bounds, num_gen, pop_size):
    #     mga_low_thrust_object = \
    #     MGALowThrustTrajectoryOptimizationProblem(transfer_body_order=transfer_body_order,
    #             no_of_free_parameters=free_param_count, bounds=bounds)
    #     problem = pg.problem(mga_low_thrust_object)
    #     algorithm = pg.algorithm(pg.sga(gen=num_gen))
    #     return pg.island(algo=algorithm, prob=problem, size=pop_size, udi=pg.mp_island()), mga_low_thrust_object

    @staticmethod
    def create_island(iteration, 
                        transfer_body_order, 
                        free_param_count, 
                        bounds, 
                        Isp,
                        m0,
                        num_gen, 
                        pop_size,
                        leg_exchange, 
                        leg_database, 
                        manual_base_functions=False, 
                        elitist_fraction=0.1,
                        mo_optimisation=False):
        """
        Function that can create an island with completely random individuals or with partially
        predefined individuals that were found to be high fitness
        """
        mga_low_thrust_problem = \
        MGALowThrustTrajectoryOptimizationProblem(transfer_body_order=transfer_body_order,
                no_of_free_parameters=free_param_count, bounds=bounds,
                manual_base_functions=manual_base_functions, mo_optimisation=mo_optimisation, Isp=Isp,
                m0=m0)

        no_of_predefined_individuals = int(pop_size * elitist_fraction) if iteration != 0 else 0
        no_of_random_individuals = pop_size - no_of_predefined_individuals
        pop_size_calc = lambda x, y: x + y
        if not mo_optimisation:
            algorithm = pg.algorithm(pg.sga(gen=num_gen))
        else:
            algorithm = pg.algorithm(pg.nsga2(gen=num_gen))
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

        if leg_exchange:
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
                # print(leg, leg_number)
                if leg_database == []: #exit if 1st iteration where leg_database is empty
                    break
                # print(leg_database)
                leg_data = legDatabaseMechanics.get_leg_data_from_database(leg, leg_database)
                # print(leg_data)
                filtered_legs = legDatabaseMechanics.get_filtered_legs(leg_data,
                        no_of_predefined_individuals)
                pre_design_parameter_vectors = \
                legDatabaseMechanics.get_dpv_from_leg_specifics(pre_design_parameter_vectors,
                        filtered_legs, transfer_body_order, free_param_count, leg_number)
            
                for i in range(no_of_predefined_individuals):
                    population.push_back(pre_design_parameter_vectors[i])
        else:
            population = pg.population(prob=mga_low_thrust_problem, size=pop_size)

        return pg.island(algo=algorithm, pop=population, udi=pg.mp_island()), mga_low_thrust_problem
    
    @staticmethod
    def create_archipelago(iteration,
                            departure_planet,
                            arrival_planet, 
                            free_param_count, 
                            pop_size,
                            bounds,
                            Isp,
                            m0,
                            max_no_of_gas,
                            number_of_sequences_per_planet,
                            itbs,
                            planet_list, 
                            leg_exchange,
                            leg_database,
                            manual_base_functions, 
                            elitist_fraction,
                            seed,
                            mo_optimisation):

        current_max_no_of_gas = max_no_of_gas - iteration
        current_island_problems = []
        temp_evaluated_sequences = []
        temp_ptbs = []

        no_of_possible_planets = len(planet_list)
        combinations_left = no_of_possible_planets**(max_no_of_gas-iteration)# or no_of_sequence_recursions

        print('Creating archipelago')
        print('Leg exchange enabled') if leg_exchange == True else None

        no_of_predefined_individuals = int(pop_size * elitist_fraction) if iteration != 0 else 0
        no_of_random_individuals = pop_size - no_of_predefined_individuals
        print(f'Population size : {pop_size}')
        print(f'Number of random individuals : {no_of_random_individuals}')
        print(f'Number of predefined individuals : {no_of_predefined_individuals}')

        archi = pg.archipelago(n=0, seed=seed)

        # Add islands with pre-defined sequences
        if iteration == 0:
            number_of_islands = number_of_sequences_per_planet[iteration]*(len(planet_list)) + 1
            directtransferbump = 1
        else:
            number_of_islands = number_of_sequences_per_planet[iteration]*(len(planet_list))
            directtransferbump = 0

            # keep number of islands to a maximum if combinatorial space is small enough
            if number_of_islands >= combinations_left:
                print('Number of islands from %i to %i' % (number_of_islands, combinations_left))
                number_of_islands = combinations_left

        print('Number of islands: ', number_of_islands, '\n')
        for i in range(number_of_islands):
            if i == 0 and iteration == 0:
                transfer_body_order = [departure_planet, arrival_planet]
                ptbs = [departure_planet]
            else:
                # [i] so that each island can start with its own predefined
                add_itbs = itbs[(i-directtransferbump) % len(planet_list)] if iteration != 0 else []
                ptbs = [departure_planet] + add_itbs + [planet_list[(i-directtransferbump) % len(planet_list)]]
                # can add seed argument but that does not work yet as intended
                random_sequence = manualTopology.create_random_transfer_body_order(
                        possible_planets=planet_list, max_no_of_gas=current_max_no_of_gas)
                transfer_body_order = ptbs + random_sequence + [arrival_planet]

            temp_ptbs.append(ptbs)
            temp_evaluated_sequences.append(transfer_body_order) 
            print(transfer_body_order)
            island_to_be_added, current_island_problem = \
            manualTopology.create_island(iteration=iteration, transfer_body_order=transfer_body_order,
                    free_param_count=free_param_count, bounds=bounds, Isp=Isp, m0=m0, num_gen=1,
                    pop_size=pop_size, leg_exchange=leg_exchange, leg_database=leg_database,
                    manual_base_functions=manual_base_functions, mo_optimisation=mo_optimisation,
                    elitist_fraction=elitist_fraction)
            current_island_problems.append(current_island_problem)
            archi.push_back(island_to_be_added)

        return temp_ptbs, temp_evaluated_sequences, number_of_islands, current_island_problems, archi

    @staticmethod
    def perform_evolution(archi, 
                            number_of_islands, 
                            num_gen, 
                            mo_optimisation):
        list_of_f_dicts = []
        list_of_x_dicts = []
        if not mo_optimisation:
            # with SO no weird work around champions have to be made
            get_champions_x = lambda archi : (archi.get_champions_x(), None) # 2, no_of_islands 
            get_champions_f = lambda archi : (archi.get_champions_f(), None)
        else:
            current_island_populations = lambda archi : [isl.get_population() for isl in archi]

            def get_champions_x(archi):
                pop_list= []
                pop_f_list= []
                ndf_x = []
                champs_x = []
                for j in range(number_of_islands):
                    pop_list.append(current_island_populations(archi)[j].get_x())
                    pop_f_list.append(current_island_populations(archi)[j].get_f())
                    current_ndf_indices = pg.non_dominated_front_2d(pop_f_list[j]) # determine how to sort
                    ndf_x.append([pop_list[j][i] for i in current_ndf_indices])
                    champs_x.append(ndf_x[j][0]) # j for island, 0 for first (lowest dV) option
                return champs_x, ndf_x

            def get_champions_f(archi):
                pop_list= []
                pop_f_list= []
                ndf_f = []
                champs_f = []
                for j in range(number_of_islands):
                    pop_list.append(current_island_populations(archi)[j].get_x())
                    pop_f_list.append(current_island_populations(archi)[j].get_f())
                    current_ndf_indices = pg.non_dominated_front_2d(pop_f_list[j]) # determine how to sort
                    ndf_f.append([pop_f_list[j][i] for i in current_ndf_indices])
                    champs_f.append(ndf_f[j][0]) # j for island, 0 for first (lowest dV) option

                return champs_f, ndf_f

        for i in range(num_gen): # step between which topology steps are executed
            print('Evolving Gen : %i / %i' % (i, num_gen))
            archi.evolve()
            champs_dict_current_gen = {}
            champ_f_dict_current_gen = {}
            champions_x, ndf_x = get_champions_x(archi)
            champions_f, ndf_f = get_champions_f(archi)
            for j in range(number_of_islands):
                    champs_dict_current_gen[j] = champions_x[j]
                    champ_f_dict_current_gen[j] = champions_f[j]
            list_of_x_dicts.append(champs_dict_current_gen)
            list_of_f_dicts.append(champ_f_dict_current_gen)

            archi.wait_check()


        print('Evolution finished')
        return list_of_x_dicts, list_of_f_dicts, champions_x, champions_f

    @staticmethod
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
                    ptbs_characters = \
                    util.transfer_body_order_conversion.get_mga_characters_from_list(j)
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

            # print('all_values', sequence_delta_v_value_dict)
            ### Get statistics
            quantities = ['max', 'min', 'mean', 'median']
            ptbs_stats = np.zeros((no_of_possible_planets, len(quantities))) 
            for i, j in sequence_delta_v_value_dict.items():
                current_max = np.max(j)
                current_min = np.min(j)
                current_mean = np.mean(j)
                current_median = np.median(j)
                # print(current_max, current_min, current_mean, current_median)
                ptbs_stats[i, :] = np.array([current_max, current_min, current_mean,
                    current_median])
            # print('ptbs_stats', ptbs_stats)

            ### Determine scalarization (what is an optimal combination of statistical quantities
            arg_min = np.argmin(ptbs_stats[:,1]) # take minimum
            local_char = pc[arg_min]
            itbs = [util.transfer_body_order_conversion.get_mga_list_from_characters(local_char) for _ in
                range (no_of_possible_planets)]
            # locally_best_sequence = [pc[ for _ in range(no_of_possible_pl

            return itbs

    @staticmethod
    def create_files(no_of_sequence_recursions,
                        number_of_islands_array,
                        island_problems,
                        champions_x,
                        champions_f,
                        list_of_lists_of_f_dicts,
                        list_of_lists_of_x_dicts,
                        no_of_points,
                        Isp, 
                        m0,
                        unsorted_evaluated_sequences_database,
                        output_directory,
                        subdirectory,
                        free_param_count,
                        num_gen,
                        pop_size,
                        cpu_count,
                        bounds):
        addition = 0
        for layer in range(no_of_sequence_recursions): # 2
            layer_folder = f'/layer_{layer}'
            champions_dict = {}
            champion_fitness_dict = {}
            # for i in range(len(champions_x[layer])):
            for i in range(number_of_islands_array[layer]):
                # print("Champion: ", champions[i])
                mga_low_thrust_problem = island_problems[layer][i]
                mga_low_thrust_problem.fitness(champions_x[layer][i], post_processing=True) # 2 is one loop
        
                # State history
                state_history = \
                mga_low_thrust_problem.transfer_trajectory_object.states_along_trajectory(no_of_points)
        
                # Thrust acceleration
                thrust_acceleration = \
                mga_low_thrust_problem.transfer_trajectory_object.inertial_thrust_accelerations_along_trajectory(no_of_points)

                mass_history, delivery_mass, invalid_trajectory = \
                util.get_mass_propagation(thrust_acceleration, Isp, m0)
            
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
                # print(addition, i)
                # print(unsorted_evaluated_sequences_database[addition + i][0])
                auxiliary_info['MGA Sequence,'] = \
                unsorted_evaluated_sequences_database[addition + i][0]
                auxiliary_info['Maximum thrust,'] = np.max([np.linalg.norm(j[1:]) for _, j in
                    enumerate(thrust_acceleration.items())])
                auxiliary_info['Delivery mass,'] = delivery_mass
                auxiliary_info['Delivery mass fraction,'] = delivery_mass / m0

    
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
                    # print(list_of_lists_of_x_dicts[layer][j][i])
                    current_island_f[j] = list_of_lists_of_f_dicts[layer][j][i]
                    current_island_x[j] = list_of_lists_of_x_dicts[layer][j][i]
                save2txt(current_island_f, 'champ_f_per_gen.dat', output_directory
                        +  subdirectory + layer_folder +  unique_identifier)
                save2txt(current_island_x, 'champs_per_gen.dat', output_directory +
                        subdirectory + layer_folder + unique_identifier)

                champions_dict[i] = champions_x[layer][i]
                champion_fitness_dict[i] = champions_f[layer][i]
            
            #Per layer add the indices
            addition += number_of_islands_array[layer]
        bound_names= ['Departure date [mjd2000]', 'Departure velocity [m/s]', 'Arrival velocity [m/s]',
            'Time of Flight [s]', 'Incoming velocity [m/s]', 'Swingby periapsis [m]', 
            'Free coefficient [-]', 'Number of revolutions [-]']

        optimisation_characteristics = {}
        # optimisation_characteristics['Transfer body order,'] = mga_sequence_characters
        optimisation_characteristics['Free parameter count,'] = free_param_count
        optimisation_characteristics['Number of generations,'] = num_gen
        optimisation_characteristics['Population size,'] = pop_size
        optimisation_characteristics['CPU count,'] = cpu_count
        optimisation_characteristics['Number of islands,'] = (number_of_islands_array[0])
        for j in range(len(bounds[0])):
            for k in range(len(bounds)):
                if k == 0:
                    min = ' LB'
                else:
                    min = ' UB'
                optimisation_characteristics[bound_names[j] + min + ','] = bounds[k][j]
        # optimisation_characteristics['Bounds'] = bounds
        
        # This should be for the statistics, we already save all information per
        unique_identifier = ""
        save2txt(optimisation_characteristics, 'optimisation_characteristics.dat', output_directory +
                subdirectory + unique_identifier)
        
        # with open(evaluated_seq_database_file, 'r+') as file_object:
        #     file_object.write(evaluated_sequences_results)
        #     file_object.close()


class separateLegMechanics:

    def __init__(self, mga_sequence_characters, champion_x, delta_v_per_leg) -> None:
        self.mga_sequence_characters = mga_sequence_characters
        self.champion_x = champion_x
        self.delta_v_per_leg = delta_v_per_leg

        chars = [i for i in self.mga_sequence_characters]
        self.number_of_legs = len(chars) - 1

        dict_of_legs = {}
        for i in range(self.number_of_legs):
            dict_of_legs[chars[i] + chars[i+1]] = i
        self.dict_of_legs = dict_of_legs

    def get_leg_specifics(self, leg_string) -> str:
        """
        This function returns the leg specifics of any given leg string

        Example
        -------
        Input : 'EM'
        Returns : [dV, ToF, #rev]
        """
        index = self.dict_of_legs[leg_string]

        current_dV = self.delta_v_per_leg[index]
        current_tof = self.champion_x[3+index] / 86400
        current_rev = self.champion_x[len(self.champion_x) - self.number_of_legs + index]
        current_leg_results = [leg_string, current_dV, current_tof, current_rev]
        return current_leg_results

    def get_sequence_leg_specifics(self):
        """
        This function returns a list of lists containing the information of whole sequence
        Example
        -------
        Input : 'EMMJ'
        Returns : [[EM, dV, ToF, #rev]
        [MM, dV, ToF, #rev]
        [MJ, dV, ToF, #rev]]
        """
        sequence_results = []
        for leg_string in self.dict_of_legs.keys():
            current_leg_specifics = self.get_leg_specifics(leg_string)
            sequence_results.append(current_leg_specifics)

        return sequence_results


class legDatabaseMechanics:

    def __init__(self):
        pass
        # self.leg_database = leg_data

    @staticmethod
    def get_leg_data_from_database(leg_to_compare, leg_database):
        """
        This function takes the leg_database and creates a list of results specific to that leg. 
        Returns list of leg specific design parameter vectors

        Parameters
        -----------

        leg_database : List[str, float, float, int]
        leg : str

        Returns
        --------

        List[np.array]
        """
        leg_data = []
        for leg_specific_data in leg_database:
            if leg_specific_data[0] == leg_to_compare:
                delta_v = leg_specific_data[1]
                tof = leg_specific_data[2]
                rev = leg_specific_data[3]
                leg_data.append(np.array([delta_v, tof, rev]))

        return leg_data

    @staticmethod
    def get_filtered_legs(leg_specific_database, no_of_predefined_individuals):
        """
        This function takes the dpv variables of a specific leg, and returns the individuals that are input
        into the island.
        Returns list of dpv variables that are to be input into the island design parameter vector

        Parameters
        -----------

        leg_specific_database : List[np.array]
        no_of_predefined_individuals : float

        Returns
        --------

        List[np.array]
        """

        leg_specific_database.sort(key=lambda dpv: dpv[0]) # sort in ascending order

        #limit to amount of predefined individuals
        leg_specific_database = leg_specific_database[:no_of_predefined_individuals] 
        # print(leg_specific_database)

        return leg_specific_database


    @staticmethod
    def get_dpv_from_leg_specifics(pre_dpv, filtered_legs, transfer_body_order, free_param_count,
            leg_count):
        """
        This function takes the filtered dpv variables and inputs them into otherwise random design
        parameter vectors.
        Filtered legs content based on get_sequence_leg_specifics function:
        0 - dV
        1 - ToF
        2 - #rev

        Parameters
        -----------

        filtered_legs : List[np.array]
        transfer_body_order : List[str]
        free_param_count : 2
        populaton : pg.population
        """
        no_of_legs = len(transfer_body_order) - 1
        no_of_gas = len(transfer_body_order) - 2
        # only loop over available leg information, if not legs exist, then just return the pre_dpv
        #the first x pre_dpv vectors are edited.
        for it, leg_info in enumerate(filtered_legs):
            pre_dpv[it][2 + leg_count] = leg_info[1]
            pre_dpv[it][2 + no_of_legs + 2 * no_of_gas + free_param_count * 3 * no_of_legs + leg_count] = leg_info[2]
        return pre_dpv





def run_mgso_optimisation(departure_planet : str,
                            arrival_planet : str,
                            free_param_count : int,
                            Isp : int,
                            m0 : int,
                            num_gen : int,
                            pop_size : int,
                            no_of_points : int,
                            bounds : list,
                            output_directory : str = '',
                            subdirectory : str = '',
                            possible_ga_planets : list = None,
                            max_no_of_gas = 1,
                            no_of_sequence_recursions = 1,
                            elitist_fraction=0.1,
                            number_of_sequences_per_planet : list =  [],
                            seed : int = 421,
                            write_results_to_file=False,
                            manual_base_functions=False,
                            leg_exchange = False,
                            top_x_sequences = 10,
                            mo_optimisation=False):

    # if os.path.exists(output_directory + subdirectory):
    #     shutil.rmtree(output_directory + subdirectory)

    if possible_ga_planets != None:
        planet_list = possible_ga_planets
        planet_characters = \
        util.transfer_body_order_conversion.get_mga_character_list_from_list(planet_list)
        print(f'Possible GA planets constrained to {[i for i in planet_list]}')
    else:
        planet_characters = ['Y', 'V', 'E', 'M', 'J', 'S', 'U', 'N']
        planet_list = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
        # Remove excess planets above target planet (except for depature)
        planet_list = manualTopology.remove_excess_planets(planet_list, departure_planet,
            arrival_planet) if max_no_of_gas != 0 else [0]
        planet_characters = manualTopology.remove_excess_planets(planet_characters, departure_planet, arrival_planet)
        print(f'GA planets limited to planets within {arrival_planet}')
    
    combinations_total = (len(planet_list)+1)**(max_no_of_gas)# or no_of_sequence_recursions
    combinations_evaluated=  number_of_sequences_per_planet[1]*len(planet_list)*no_of_sequence_recursions
    print(f'The combinational coverage that will be achieved {combinations_evaluated} / {combinations_total}')
    unique_identifier = '/topology_database'

    # TODO : replace with list of lists
    
    if write_results_to_file:
        evaluated_seq_database_file = output_directory + subdirectory + unique_identifier +  '/evaluated_sequences_database.txt'
        sorted_evaluated_seq_database_file = output_directory + subdirectory + '/sorted_evaluated_sequences_database.txt'
        separate_leg_database_file = output_directory + subdirectory +  unique_identifier +  '/separate_leg_database.txt'
        
        if not os.path.exists(output_directory + subdirectory + unique_identifier):
            os.makedirs(output_directory + subdirectory + unique_identifier)

        if not os.path.exists(evaluated_seq_database_file):
            open(evaluated_seq_database_file, 'a+').close()
        else:
            os.remove(evaluated_seq_database_file)

        if not os.path.exists(separate_leg_database_file):
            open(separate_leg_database_file, 'a+').close()
        else:
            os.remove(separate_leg_database_file)

        if not os.path.exists(sorted_evaluated_seq_database_file):
            open(sorted_evaluated_seq_database_file, 'a+').close()
        else:
            os.remove(sorted_evaluated_seq_database_file)

    # evaluated_sequences_database = [[], []]
    # separate_leg_database = [[], []]
    evaluated_sequences_database = []
    separate_leg_database = []

    mp.freeze_support()
    cpu_count = os.cpu_count() # not very relevant because differnent machines + asynchronous

    # print(len(planet_list)
    evaluated_sequences_results = 'Sequence, Delta V, ToF, Delivery Mass Fraction \n'
    leg_results = 'Leg, Delta V, ToF, #rev\n'

    # border case for max_no_of_gas == 0
    # range_no_of_sequence_recursions = [i for i in range(no_of_sequence_recursions)]
    if max_no_of_gas == 0:
    #     no_of_sequence_recursions = 1
    #     range_no_of_sequence_recursions = [0]
        number_of_sequences_per_planet = [0]

    # variable definitions
    champions_x = {}
    champions_f = {}
    island_problems = {}
    evaluated_sequences_dict = {}
    itbs = []
    number_of_islands_array = np.zeros(no_of_sequence_recursions+1, dtype=int) # list of number of islands per recursion
    list_of_lists_of_x_dicts = []# list containing list of dicts of champs per gen
    list_of_lists_of_f_dicts = []
    
###########################################################################
# MGSO Optimisation ###############################################
###########################################################################
    
    # Loop for number of sequence recursions
    # p_bump = False
    for p in range(no_of_sequence_recursions): # gonna be max_no_of_gas
        print('Iteration: ', p, '\n')

        # Creation of the archipelago
        temp_ptbs, temp_evaluated_sequences, number_of_islands, current_island_problems, archi = \
        manualTopology.create_archipelago(p,
                                        departure_planet,
                                        arrival_planet, 
                                        free_param_count, 
                                        pop_size,
                                        bounds,
                                        Isp,
                                        m0,
                                        max_no_of_gas,
                                        number_of_sequences_per_planet,
                                        itbs,
                                        planet_list, 
                                        leg_exchange,
                                        separate_leg_database,
                                        manual_base_functions, 
                                        elitist_fraction,
                                        seed,
                                        mo_optimisation)

        number_of_islands_array[p] = number_of_islands
        island_problems[p] = current_island_problems

        list_of_x_dicts, list_of_f_dicts, champions_x[p], \
        champions_f[p] = manualTopology.perform_evolution(archi,
                            number_of_islands_array[p],
                            num_gen,
                            mo_optimisation)
        # print(list_of_x_dicts, list_of_f_dicts)
        # print(champions_x[p], champions_f[p])

        list_of_lists_of_x_dicts.append(list_of_x_dicts)
        list_of_lists_of_f_dicts.append(list_of_f_dicts)
        # print('Number of islands this iteration', number_of_islands_array[p])
        # print('champ_f_dict_per_gen : ', champ_f_dict_per_gen)
        # print('list_of_f_dicts : ', list_of_f_dicts)

    ### Algorithm for next island generation ###

        # Write mga sequences to evaluated sequence database file
        delta_v = {}
        delta_v_per_leg = {}
        tof = {}
        evaluated_sequences_dict[p] = [[] for _ in range(number_of_islands)]

        for j in range(number_of_islands):

            # if mo_optimisation:
            #     champions_x[p][j] = champions_x[p][j][0]
            # define delta v to be used for evaluation of best sequence
            # for MO, the 0 indicates that the best dV is chosen for the database
            island_problems[p][j].fitness(champions_x[p][j], post_processing=True)
            delta_v[j] = island_problems[p][j].transfer_trajectory_object.delta_v
            # print(delta_v)
            delta_v_per_leg[j] = island_problems[p][j].transfer_trajectory_object.delta_v_per_leg
            tof[j]=  island_problems[p][j].transfer_trajectory_object.time_of_flight
            thrust_acceleration = \
            island_problems[p][j].transfer_trajectory_object.inertial_thrust_accelerations_along_trajectory(no_of_points)

            mass_history, delivery_mass, invalid_trajectory = \
            util.get_mass_propagation(thrust_acceleration, Isp, m0)
            # print(delta_v, delta_v_per_leg, tof)

            # Save evaluated sequences to database with extra information
            mga_sequence_characters = \
                    util.transfer_body_order_conversion.get_mga_characters_from_list(
                                    temp_evaluated_sequences[j])
            current_sequence_result = [mga_sequence_characters, delta_v[j], tof[j] / 86400,
                    delivery_mass / m0]
            evaluated_sequences_results += "%s, %d, %d, %f\n" % tuple(current_sequence_result)
            # evaluated_sequences_database[0].append(mga_sequence_characters)
            # evaluated_sequences_database[1].append(current_sequence_result)
            evaluated_sequences_database.append(current_sequence_result)

            evaluated_sequences_dict[p][j] = current_sequence_result

            # Save separate leg information
            current_sequence_leg_mechanics_object = separateLegMechanics(mga_sequence_characters,
                    champions_x[p][j], delta_v_per_leg[j])
            current_sequence_leg_results = \
            current_sequence_leg_mechanics_object.get_sequence_leg_specifics()
            # print(current_sequence_leg_results)
            for leg in current_sequence_leg_results:
                leg_results += "%s, %d, %d, %d\n" % tuple(leg) # only values
                # print(leg_results)
                # separate_leg_database[0].append(current_sequence_leg_results[leg][0])
                # separate_leg_database[1].append(current_sequence_leg_results[leg][1:])
                separate_leg_database.append(leg)

        # print(evaluated_sequences_database, separate_leg_database)
            # check auxiliary help for good legs -> SOLVE PICKLE ERROR
            # we have island_problms in a list
            # deltav per leg weighted average based on how far away the transfer is (EM is stricter
            # than YN
        if max_no_of_gas == 0:
            break

        ### Define ITBS ###
        # print(delta_v, temp_ptbs)
        if p == 0:
            dt_delta_v = delta_v[0] # dt is direct transfer
            dt_sequence = temp_ptbs[0]
            delta_v.pop(0)
            temp_ptbs.pop(0)
            for it in list(delta_v):
                if it == 0:
                    continue
                delta_v[it-1] = delta_v.pop(it)

        print(delta_v, temp_ptbs)
        current_itbs = manualTopology.get_itbs(dv=delta_v, ptbs=temp_ptbs,
            type_of_selection="proportional", dt_tuple=(dt_delta_v, dt_sequence),
            pc=planet_characters, pl=planet_list)
        
        # print('Current Initial Target Body Sequence : ', current_itbs)
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
        with open(evaluated_seq_database_file, 'w') as file_object:
            # print(evaluated_sequences_database)
            # json.dump(evaluated_sequences_results, file_object)
            file_object.write(evaluated_sequences_results)
            file_object.close()

        with open(separate_leg_database_file, 'w') as file_object:
            # json.dump(leg_results, file_object)
            file_object.write(leg_results)
            file_object.close()

        # evaluate sorted database
        unsorted_evaluated_sequences_database = evaluated_sequences_database.copy()
        evaluated_sequences_database.sort(key=lambda elem : elem[1])
        sorted_evaluated_sequences_database = evaluated_sequences_database.copy()
        sorted_evaluated_sequences_results = 'Sequence, Delta V, ToF, Delivery Mass Fraction\n'
        for i in evaluated_sequences_database:
            sorted_evaluated_sequences_results += "%s, %d, %d, %f\n" % tuple(i)

        with open(sorted_evaluated_seq_database_file, 'w') as file_object:
            file_object.write(sorted_evaluated_sequences_results)
            file_object.close()

        manualTopology.create_files(no_of_sequence_recursions,
                            number_of_islands_array,
                            island_problems,
                            champions_x,
                            champions_f,
                            list_of_lists_of_f_dicts,
                            list_of_lists_of_x_dicts,
                            no_of_points,
                            Isp, 
                            m0,
                            unsorted_evaluated_sequences_database,
                            output_directory,
                            subdirectory,
                            free_param_count,
                            num_gen,
                            pop_size,
                            cpu_count,
                            bounds)
