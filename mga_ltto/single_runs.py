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
import sys

# Tudatpy imports
import tudatpy
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.astro import time_conversion

sys.path.append('../mga_ltto/src/')
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

# 0 fp
# transfer_body_order = ["Earth", "Mars"]
# design_parameter_vector =np.array([1199.988419679548, 0.0, 282.9013364921598, 906.5959220524712, 0.0, 0.0])
# mga_low_thrust_problem = \
# MGALowThrustTrajectoryOptimizationProblem(transfer_body_order=transfer_body_order,
#         no_of_free_parameters=0)
# mga_sequence_characters = util.transfer_body_order_conversion.get_mga_characters_from_list(
#         transfer_body_order)

#EMJ 2fp
# Isp = 3000
# m0 = 1300
# transfer_body_order = ["Earth", "Mars", "Jupiter"]
# design_parameter_vector = np.array([ 900525711.786143, 0.0, 69181163.50817049, 101650320.36332642,
#     2541.500771560418, 9.575103719627307, 1607.0, 8074.0, -8768.0, -1224.0, -3123.0, 5325.0,
#     -9147.0, -4187.0, -9772.0, -5821.0, 2269.0, 2448.0, 1.0, 0.0])
# mga_low_thrust_problem = \
# MGALowThrustTrajectoryOptimizationProblem(transfer_body_order=transfer_body_order,
#         no_of_free_parameters=2)
# mga_sequence_characters = util.transfer_body_order_conversion.get_mga_characters_from_list(
#         transfer_body_order)

# 2fp
# design_parameter_vector=  np.array([1197.3079260152658, 0.0, 262.41762550207926, 1105.3748080547718,
#     7727.0,7189.0, -6947.0, 8690.0, 9236.0, -9382.0, -9968.0, 3883.0, -8402.0, -6677.0, 2385.0,
#     -84.0, 0.0, 0.0])
# transfer_body_order=  ['Earth', 'Mars', 'Mercury', 'Mercury', 'Jupiter']
# design_parameter_vector = np.array([798144938.5175248, 0.0, 34438826.5744251, 72151028.85329694,
#     102918049.60361323, 90408666.78799471, 96.25866185034238, 4082.5018127125654,
#     2105.4589341361825, 10.064195086589267, 2.6415739555067135, 2.856660323144302, 4570.0, 4659.0,
#     3271.0, 2083.0, 2374.0, -2007.0, -2344.0, 7582.0, -116.0, 6927.0, 6251.0, 3491.0, -8115.0,
#     -2350.0, -4790.0, -4855.0, 8799.0, 5799.0, 994.0, -6532.0, 6399.0, -195.0, 3851.0, 876.0, 1.0,
#     1.0, 0.0, 0.0])
#
# mga_sequence_characters = util.transfer_body_order_conversion.get_mga_characters_from_list(
#         transfer_body_order)
# mga_low_thrust_problem = \
# MGALowThrustTrajectoryOptimizationProblem(transfer_body_order=transfer_body_order,
#         no_of_free_parameters=2)

#0fp that recreates the result
transfer_body_order = ["Earth", "Mars"]
design_parameter_vector = np.array([10024.5, 0.0, 0.0, 1050.0, 2.0])
mga_sequence_characters = util.transfer_body_order_conversion.get_mga_characters_from_list(
        transfer_body_order)
mga_low_thrust_problem = \
MGALowThrustTrajectoryOptimizationProblem(transfer_body_order=transfer_body_order,
        no_of_free_parameters=0, manual_base_functions=False, mo_optimisation=False)
print(time_conversion.julian_day_to_calendar_date(time_conversion.modified_julian_day_to_julian_day(10025.0
    + 51544.5)))


index_list = [0, 3]
for i in index_list:
    design_parameter_vector[i] *= julian_day
unique_identifier = '/EM_verification'

mga_low_thrust_problem.fitness(design_parameter_vector, post_processing=True)

### Post processing ###
state_history = \
mga_low_thrust_problem.transfer_trajectory_object.states_along_trajectory(no_of_points)
# print(state_history)

# Thrust acceleration
thrust_acceleration = \
mga_low_thrust_problem.transfer_trajectory_object.inertial_thrust_accelerations_along_trajectory(no_of_points)
# print(thrust_acceleration)
# mass_history, delivery_mass = util.get_mass_propagation(thrust_acceleration, Isp, m0)
# print(m0, delivery_mass)

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
