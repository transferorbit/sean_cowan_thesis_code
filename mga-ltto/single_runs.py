'''
Author: Sean Cowan
Purpose: MSc Thesis
Date Created: Unknown

This module runs single runs with certain design parameter vector in order to analyse specific
outcomes for specific parameters.
'''
###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import numpy as np
import os
import multiprocessing as mp

# Tudatpy imports
import tudatpy
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.astro import time_conversion

import mga_low_thrust_utilities as util
from pygmo_problem import MGALowThrustTrajectoryOptimizationProblem

current_dir = os.getcwd()
output_directory = current_dir + '/pp_singleruns'
write_results_to_file = True
julian_day = constants.JULIAN_DAY

# transfer_body_order = ["Earth", "Mars"]
# transfer_body_order = ["Earth", "Mars", "Jupiter"]
no_of_points=500

# dpv from file
# dir = 'single_sequence_optimisation/test_optimization_results/champions/'
# design_parameter_vectors = np.loadtxt(dir + 'champions.dat')

# start_epoch = 725803200.000000
# tof = 1.5*constants.JULIAN_YEAR
# final_time = start_epoch+tof
# free_coefficients = [ -38852.8007796046,
#                       12759.6520758622 ,
#                       -31425.1033837461,
#                       -54221.2080529588,
#                       -9658.99274172873,
#                       6519.19424919116]
# design_parameter_vector = np.array([start_epoch, 0.0, tof, free_coefficients[0],
#     free_coefficients[1], free_coefficients[2], free_coefficients[3], free_coefficients[4],
#     free_coefficients[5], 0.0])

# transfer_body_order = ["Earth", "Mars", "Venus", "Venus", "Jupiter"]
# design_parameter_vector = np.array([793293955.8355129, 0.0, 100037080.39505903, 24027527.630540058,
#     41415785.19754335, 75680409.92923114, 769.608647936012, 3116.227712819472, 1682.9766433373895,
#     2.7405905638652768, 8.649480070012665, 6.332816720721828, -1970.0, -3373.0, -1933.0, -2111.0,
#     129.0, -8323.0, -3463.0, 2150.0, 8385.0, -7920.0, -9419.0, -5465.0, -794.0, -3013.0, 7457.0,
#     -1115.0, -3748.0, -7115.0, 3578.0, 8028.0, -3360.0, 1090.0, -6401.0, -5072.0, 2.0, 0.0, 2.0,
#     0.0])
# mga_low_thrust_problem = \
# MGALowThrustTrajectoryOptimizationProblem(transfer_body_order=transfer_body_order,
#         no_of_free_parameters=2)
# mga_sequence_characters = util.transfer_body_order_conversion.get_mga_characters_from_list(
#         transfer_body_order)

# transfer_body_order = ["Earth", "Mars", "Earth", "Jupiter"]
# design_parameter_vector = np.array([825888649.2573633, 0.0, 78190321.22839531, 33290229.75152632,
#     95076967.72102702, 46.6838926237002, 3074.131807921915, 5.565627663196208, 8.1171557568468,
#     139.0, 6131.0, -2622.0, -6300.0, -299.0, -1731.0, 289.0, -1641.0, 6984.0, 3438.0, 1587.0,
#     1885.0, -9917.0, -8114.0, -2667.0, 5548.0, 7637.0, -9544.0, 1.0, 0.0, 0.0])
#
# mga_low_thrust_problem = \
# MGALowThrustTrajectoryOptimizationProblem(transfer_body_order=transfer_body_order,
#         no_of_free_parameters=2)
# mga_sequence_characters = util.transfer_body_order_conversion.get_mga_characters_from_list(
#         transfer_body_order)

transfer_body_order = ["Earth", "Mars", "Jupiter", "Saturn"]
design_parameter_vector=  np.array([432637767.0582297, 0.0, 37617329.41815525, 114454633.09842438,
    127173280.30168964, 1822.2404419599536, 8.387164987550776, 8.565347514555263,
    6.7719118763726485, 815.0, 6658.0, -8652.0, 6169.0, -7799.0, -7583.0, -9202.0, 6315.0, -9730.0,
    -1006.0, -6841.0, 1749.0, 6079.0, -2752.0, -8547.0, 2211.0, -6901.0, 7974.0, 0.0, 0.0, 0.0])
mga_low_thrust_problem = \
MGALowThrustTrajectoryOptimizationProblem(transfer_body_order=transfer_body_order,
        no_of_free_parameters=2)
mga_sequence_characters = util.transfer_body_order_conversion.get_mga_characters_from_list(
        transfer_body_order)

# design_parameter_vector = np.array([9558.896403441706, 0.0, 904.9805697730939, 385.30358508711015,
#     1100.4278671415163, 1183.979987813652, 3074.131807921915, 5.565627663196208, 8.129202206701256,
#     -804.0, 5232.0, -3612.0, -6300.0, -299.0, -1731.0, -2587.0, -6648.0, 6984.0, 3438.0, -3043.0,
#     -1027.0, -8283.0, -8114.0, -2667.0, 7128.0, 2473.0, -1471.0, 1.0, 0.0, 0.0])

#0fp that recreates the result
# design_parameter_vector = np.array([10025.0, 0.0, 1050.0, 2.0])
# mga_sequence_characters = util.transfer_body_order_conversion.get_mga_characters_from_list(
#         transfer_body_order)
# mga_low_thrust_problem = \
# MGALowThrustTrajectoryOptimizationProblem(transfer_body_order=transfer_body_order,
#         no_of_free_parameters=0)
print(time_conversion.julian_day_to_calendar_date(time_conversion.modified_julian_day_to_julian_day(10025.0
    + 51544.5)))


# index_list = [0, 2, 3, 4]
# for i in index_list:
#     design_parameter_vector[i] *= julian_day
unique_identifier = '/tudat_example_EMJS'

mga_low_thrust_problem.fitness(design_parameter_vector, post_processing=True)

### Post processing ###
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
print(delta_v, time_of_flight)

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
    auxiliary_info['Maximum thrust,'] = np.max([np.linalg.norm(j[1:]) for _, j in
        enumerate(thrust_acceleration.items())])

    save2txt(state_history, 'state_history.dat', output_directory + unique_identifier)
    save2txt(thrust_acceleration, 'thrust_acceleration.dat', output_directory + unique_identifier)
    save2txt(node_times, 'node_times.dat', output_directory + unique_identifier)
    save2txt(auxiliary_info, 'auxiliary_info.dat', output_directory + unique_identifier)
