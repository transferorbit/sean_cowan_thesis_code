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

import mga_low_thrust_utilities as util
from pygmo_problem import MGALowThrustTrajectoryOptimizationProblem

current_dir = os.getcwd()
output_directory = current_dir + '/pp_singleruns'
write_results_to_file = False
julian_day = constants.JULIAN_DAY

transfer_body_order = ["Earth", "Mars"]
no_of_points=500
# mga_sequence_characters = util.transfer_body_order_conversion.get_mga_characters_from_list(
#         transfer_body_order)

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
# design_parameter_vector = np.array([695310227.667289, 0.0, 77203061.5420745, -4539.0, 7924.0,
#     -7830.0, -9585.0, 17.0, 41.0, 1.0])
mga_low_thrust_problem = \
MGALowThrustTrajectoryOptimizationProblem(transfer_body_order=transfer_body_order,
        no_of_free_parameters=2)

# 2 fp
# design_parameter_vector = np.array([86400000.0, 100.0, 32784995.97082778, 0, 0, 0, 0, 0, 0, 0.0])
# mga_low_thrust_problem = \
# MGALowThrustTrajectoryOptimizationProblem(transfer_body_order=transfer_body_order,
#         no_of_free_parameters=2)

mga_low_thrust_problem.fitness(design_parameter_vector, post_processing=True)
delta_v = mga_low_thrust_problem.transfer_trajectory_object.delta_v
delta_v_per_leg = mga_low_thrust_problem.transfer_trajectory_object.delta_v_per_leg
tof = mga_low_thrust_problem.transfer_trajectory_object.time_of_flight

state_history = \
mga_low_thrust_problem.transfer_trajectory_object.states_along_trajectory(no_of_points)

# Thrust acceleration
thrust_acceleration = \
mga_low_thrust_problem.transfer_trajectory_object.inertial_thrust_accelerations_along_trajectory(no_of_points)
# dpv = mga_low_thrust_problem.get_design_parameter_vector()
print(delta_v, tof, delta_v_per_leg)#, dpv)
unique_identifier = '/2fp'
if write_results_to_file:
    save2txt(state_history, 'state_history.dat', output_directory + unique_identifier)
    save2txt(thrust_acceleration, 'thrust_acceleration.dat', output_directory + unique_identifier)

