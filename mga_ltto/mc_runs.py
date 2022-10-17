'''
Author: Sean Cowan
Purpose: MSc Thesis
Date Created: 03-10-2022

This modules performs a monte carlo analysis on design parameters.
'''
###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import numpy as np
import os
import random

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

#################################################################
# MC RUNS #######################################################
#################################################################

# 2 fp
# design_parameter_vector = np.array([86400000.0, 100.0, 32784995.97082778, 0, 0, 0, 0, 0, 0, 0.0])

mga_low_thrust_problem = \
MGALowThrustTrajectoryOptimizationProblem(transfer_body_order=transfer_body_order,
        no_of_free_parameters=2)

dpv_param_count = 10
no_of_mc_points = 1000
dpv_array = np.zeros((no_of_mc_points, dpv_param_count))
fitness_array = np.zeros((no_of_mc_points, 2)) #delta v, tof
state_hist_dict = {}
thrust_acceleration_dict = {}
no_of_invalid_points = 0

for i in range(no_of_mc_points):
    fixed_parameters = np.array([86400000.0, 100.0, 32784995.97082778,])
    # random_dpv = (np.random.random_sample(size=(6, 1)) - 1/2) * 2e6 # for range between -1e6 and 1e6
    random_dpv = np.array([random.uniform(-10**3, 10**3) for _ in range(6)]) # for range between -1e6 and 1e6
    # random_dpv = np.array([random.gauss(0, 1e5) for _ in range(6)]) # for range between -1e6 and 1e6
    revolutions = np.array([0.0])
    design_parameter_vector = np.append(fixed_parameters, np.append(random_dpv, revolutions))
    dpv_array[i, :] = design_parameter_vector

    try:
        mga_low_thrust_problem.fitness(design_parameter_vector, post_processing=True)
        delta_v = mga_low_thrust_problem.transfer_trajectory_object.delta_v
        # delta_v_per_leg = mga_low_thrust_problem.transfer_trajectory_object.delta_v_per_leg
        tof = mga_low_thrust_problem.transfer_trajectory_object.time_of_flight
        fitness_array[i, :] = np.array([delta_v, tof])

        # state_history = \
        # mga_low_thrust_problem.transfer_trajectory_object.states_along_trajectory(no_of_points)
        # thrust_acceleration = \
        # mga_low_thrust_problem.transfer_trajectory_object.inertial_thrust_accelerations_along_trajectory(no_of_points)
        # state_hist_dict[i]=  state_history
        # thrust_acceleration_dict[i]=  thrust_acceleration
    except Exception as inst:
        fitness_array[i, :] = np.array([np.nan, np.nan])
        no_of_invalid_points += 1

print(no_of_invalid_points)
# unique_identifier = '/2fp'
# if write_results_to_file:
#     save2txt(state_history, 'state_history.dat', output_directory + unique_identifier)
#     save2txt(thrust_acceleration, 'thrust_acceleration.dat', output_directory + unique_identifier)

