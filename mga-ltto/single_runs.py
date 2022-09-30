'''
Author: Sean Cowan
Purpose: MSc Thesis
Date Created: 26-07-2022

This module performs the optimization calculations using the help modules from mga-low-thrust-utilities.py and
pygmo-utilities.py
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
write_results_to_file = True
julian_day = constants.JULIAN_DAY

transfer_body_order = ["Earth", "Mars", "Jupiter"]
mga_sequence_characters = util.transfer_body_order_conversion.get_mga_characters_from_list(
        transfer_body_order)

# dpv from file
dir = 'single_sequence_optimisation/test_optimization_results/champions/'
design_parameter_vectors = np.loadtxt(dir + 'champions.dat')

# dpv self defined
# design_parameter_vectors = np.array([])

design_parameter_vectors = design_parameter_vectors[:,1:]
# print(design_parameter_vectors)

mga_low_thrust_problem = \
MGALowThrustTrajectoryOptimizationProblem(transfer_body_order=transfer_body_order)
mga_low_thrust_problem.fitness(design_parameter_vectors[0], post_processing=True)
delta_v = mga_low_thrust_problem.transfer_trajectory_object.delta_v
delta_v_per_leg = mga_low_thrust_problem.transfer_trajectory_object.delta_v_per_leg
tof = mga_low_thrust_problem.transfer_trajectory_object.time_of_flight
# dpv = mga_low_thrust_problem.get_design_parameter_vector()
print(delta_v, tof, delta_v_per_leg)#, dpv)

