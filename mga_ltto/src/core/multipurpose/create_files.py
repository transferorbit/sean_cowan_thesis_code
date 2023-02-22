'''
Author: Sean Cowan
Purpose: MSc Thesis
Date Created: 31-01-2023

This module includes the create_files function, which can be applied to either ltto or mgaso and saves files in the
format that is used for this thesis
'''

###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General
import numpy as np
import os
import sys

# Tudatpy
from tudatpy.io import save2txt
from tudatpy.kernel import constants

current_dir = os.getcwd()
sys.path.append('/Users/sean/Desktop/tudelft/thesis/code/mga_ltto/src/') # this only works if you run ltto and mgso while in the directory that includes those files
# Local
import core.multipurpose.mga_low_thrust_utilities as util
# If conda environment does not work
# import sys
# sys.path.insert(0, "/Users/sean/Desktop/tudelft/tudat/tudat-bundle/build/tudatpy")

def create_files(type_of_optimisation=None,
                 no_of_sequence_recursions=None,
                 number_of_islands_array=[None],
                 islands_per_sequence_array=[None],
                 number_of_sequences_array=[None],
                 number_of_islands=None,
                 island_problems=None,
                 island_problem=None,
                 champions_x=None,
                 champions_f=None,
                 list_of_lists_of_f_dicts={0:None},
                 list_of_lists_of_x_dicts={0:None},
                 list_of_f_dicts=None,
                 list_of_x_dicts=None,
                 ndf_f=None,
                 ndf_x=None,
                 no_of_points=None,
                 Isp=None, 
                 m0=None,
                 unsorted_evaluated_sequences_database=None,
                 mga_sequence_characters=None,
                 output_directory=None,
                 subdirectory=None,
                 free_param_count=None,
                 num_gen=None,
                 pop_size=None,
                 cpu_count=None,
                 bounds=None,
                 archi=None,
                 compute_mass=False,
                 fraction_ss_evaluated=None,
                 number_of_sequences_per_planet=None,
                 planet_list=None,
                 itbs=None):

    if type_of_optimisation == 'ltto':
        no_of_sequence_recursions = 1

    for layer in range(no_of_sequence_recursions): # 2
        if type_of_optimisation == 'mgaso':
            layer_folder = f'/layer_{layer}'
        else:
            layer_folder = ''
        champions_dict = {}
        champion_fitness_dict = {}
        # for i in range(len(champions_x[layer])):
        for i in range(number_of_islands_array[layer] if type_of_optimisation == 'mgaso' else \
            number_of_islands):
            # print("Champion: ", champions[i])
            mga_low_thrust_problem = island_problems[layer][i] if type_of_optimisation == \
            'mgaso' else island_problem

            mga_low_thrust_problem.fitness(champions_x[layer][i] if type_of_optimisation == \
            'mgaso' else champions_x[i], post_processing=True) # 2 is one loop

            bound_names = mga_low_thrust_problem.bound_names
    
            # State history
            state_history = \
            mga_low_thrust_problem.transfer_trajectory_object.states_along_trajectory(no_of_points)
    
            # Thrust acceleration
            thrust_acceleration = \
            mga_low_thrust_problem.transfer_trajectory_object.inertial_thrust_accelerations_along_trajectory(
                        no_of_points)

            if compute_mass:
                delivery_mass = mga_low_thrust_problem.get_delivery_mass()
        
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
            delta_v_per_node = mga_low_thrust_problem.transfer_trajectory_object.delta_v_per_node
            number_of_legs = mga_low_thrust_problem.transfer_trajectory_object.number_of_legs
            number_of_nodes = mga_low_thrust_problem.transfer_trajectory_object.number_of_nodes
            time_of_flight = mga_low_thrust_problem.transfer_trajectory_object.time_of_flight

            # Aux file per island
            auxiliary_info = {}
            auxiliary_info['Isp,'] = Isp
            auxiliary_info['m0,'] = m0
            auxiliary_info['Number of legs,'] = number_of_legs 
            auxiliary_info['Number of nodes,'] = number_of_nodes 
            auxiliary_info['Total ToF (Days),'] = time_of_flight / 86400.0
            for j in range(number_of_legs):
                auxiliary_info['Delta V for leg %s,'%(j)] = delta_v_per_leg[j]
            for j in range(number_of_nodes):
                auxiliary_info['Delta V for node %s,'%(j)] = delta_v_per_node[j]
            auxiliary_info['Delta V,'] = delta_v 
            auxiliary_info['MGA Sequence,'] = mga_low_thrust_problem.mga_characters
            auxiliary_info['Maximum thrust,'] = np.max([np.linalg.norm(j[1:]) for _, j in
                enumerate(thrust_acceleration.items())])
            if compute_mass:
                auxiliary_info['Delivery mass,'] = delivery_mass
                auxiliary_info['Delivery mass fraction,'] = delivery_mass / m0

            # Get ndf
            if len(mga_low_thrust_problem.objectives) != 1:
                current_ndf_f_list = ndf_f[i]
                current_ndf_f_dict = {}
                for p, q in enumerate(current_ndf_f_list):
                    current_ndf_f_dict[p] = q


            unique_identifier = "/islands/island_" + str(i) + "/"
            if len(mga_low_thrust_problem.objectives) != 1:
                save2txt(current_ndf_f_dict, 'pareto_front.dat', output_directory + subdirectory +
                        layer_folder + unique_identifier)
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
                current_island_f[j] = list_of_lists_of_f_dicts[layer][j][i] if \
                type_of_optimisation == 'mgaso' else list_of_f_dicts[j][i]
                current_island_x[j] = list_of_lists_of_x_dicts[layer][j][i] if \
                type_of_optimisation == 'mgaso' else list_of_x_dicts[j][i]
            save2txt(current_island_f, 'champ_f_per_gen.dat', output_directory
                    +  subdirectory + layer_folder +  unique_identifier)
            save2txt(current_island_x, 'champs_per_gen.dat', output_directory +
                    subdirectory + layer_folder + unique_identifier)

            if layer == (no_of_sequence_recursions-1):
                champions_dict[i] = champions_x[layer][i] if type_of_optimisation == \
                'mgaso' else champions_x[i]
                champion_fitness_dict[i] = champions_f[layer][i] if type_of_optimisation == \
                'mgaso' else champions_f[i]

    optimisation_characteristics = {}
    # optimisation_characteristics['Transfer body order,'] = mga_sequence_characters
    optimisation_characteristics['No of points,'] = no_of_points
    optimisation_characteristics['Free parameter count,'] = free_param_count
    optimisation_characteristics['Number of generations,'] = num_gen
    optimisation_characteristics['Population size,'] = pop_size
    optimisation_characteristics['CPU count,'] = cpu_count
    optimisation_characteristics['Isp,'] = Isp
    optimisation_characteristics['m0,'] = m0
    optimisation_characteristics['Manual base functions,'] = mga_low_thrust_problem.manual_base_functions
    optimisation_characteristics['Zero revs,'] = mga_low_thrust_problem.zero_revs
    if type_of_optimisation == 'mgaso':
        for j in range(no_of_sequence_recursions):
            optimisation_characteristics[f'Number of islands - Layer {j},'] = number_of_islands_array[j] 
        for j in range(no_of_sequence_recursions):
            optimisation_characteristics[f'Islands per sequence - Layer {j},'] = islands_per_sequence_array[j]
        for j in range(no_of_sequence_recursions):
            optimisation_characteristics[f'Number of sequences - Layer {j},'] = number_of_sequences_array[j]
        for j in range(no_of_sequence_recursions):
            optimisation_characteristics[f'Fraction of sequences - Layer {j},'] = fraction_ss_evaluated[j]
        for j in range(no_of_sequence_recursions):
            optimisation_characteristics[f'Number of sequences per planet - Layer {j},'] = \
            number_of_sequences_per_planet[j]
        optimisation_characteristics['Possible GA planets,'] = ' '.join(planet_list)
        optimisation_characteristics['Initial Target Body Sequence'] = ' '.join(itbs)
    else:
        optimisation_characteristics['Number of islands,'] = number_of_islands 
        

    optimisation_characteristics['Dynamic time_of_flight,'] = mga_low_thrust_problem.dynamic_bounds['time_of_flight']
    optimisation_characteristics['Dynamic orbit_ori_angle,'] = mga_low_thrust_problem.dynamic_bounds['orbit_ori_angle']
    optimisation_characteristics['Dynamic swingby_outofplane,'] = \
    mga_low_thrust_problem.dynamic_bounds['swingby_outofplane']
    optimisation_characteristics['Dynamic swingby_inplane,'] = \
    mga_low_thrust_problem.dynamic_bounds['swingby_inplane']
    optimisation_characteristics['Dynamic shaping_function,'] = \
    mga_low_thrust_problem.dynamic_bounds['swingby_inplane']
    optimisation_characteristics[f'Objective 1,'] = mga_low_thrust_problem.objectives[0]
    if len(mga_low_thrust_problem.objectives) > 1:
        optimisation_characteristics[f'Objective 2,'] = mga_low_thrust_problem.objectives[1]
    optimisation_characteristics['Topology Info,'] = archi.get_topology().get_extra_info().replace("\n", "").strip()
    optimisation_characteristics['Algorithm Info,'] = archi[0].get_algorithm().get_extra_info().replace("\n", "").strip()
    for j in range(len(bounds[0])):
        for k in range(len(bounds)):
            if k == 0:
                min = ' LB'
            else:
                min = ' UB'
            optimisation_characteristics[bound_names[j] + min + ','] = bounds[k][j]
    if mga_low_thrust_problem.manual_tof_bounds != None:
        it = 0
        for i in range(len(mga_low_thrust_problem.manual_tof_bounds[0])):
            optimisation_characteristics[
                    f'Manual ToF bounds Leg {it} - {mga_low_thrust_problem.legstrings[i]} LB,'] = \
                    mga_low_thrust_problem.manual_tof_bounds[0][i]
            optimisation_characteristics[
                    f'Manual ToF bounds Leg {it} - {mga_low_thrust_problem.legstrings[i]} UB,'] = \
                    mga_low_thrust_problem.manual_tof_bounds[1][i]
            it += 1

    unique_identifier = "/champions/"
    if type_of_optimisation == 'ltto':
        save2txt(champion_fitness_dict, 'champion_fitness.dat', output_directory + subdirectory +
                unique_identifier)
        save2txt(champions_dict, 'champions.dat', output_directory + subdirectory +
                unique_identifier)
    unique_identifier = ""
    save2txt(optimisation_characteristics, 'optimisation_characteristics.dat', output_directory +
            subdirectory + unique_identifier)
    
