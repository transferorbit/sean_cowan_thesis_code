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
import pygmo as pg
import multiprocessing as mp

# Tudatpy imports
import tudatpy
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.math import interpolators
from tudatpy.kernel.trajectory_design import shape_based_thrust
from tudatpy.kernel.trajectory_design import transfer_trajectory

# import mga_low_thrust_utilities as mga_util
import pygmo_island as isl
from pygmo_problem import MGALowThrustTrajectoryOptimizationProblem
from error_vectors import design_vector_list
# print(design_vector_list)

current_dir = os.getcwd()
write_results_to_file = True
subdirectory = '/test_optimization_results/'

mga_low_thrust_problem = MGALowThrustTrajectoryOptimizationProblem(no_of_free_parameters=1)

# Single runs from the list
# mga_low_thrust_problem.fitness(design_vector_list[6])

# All 'problematic' design vectors run in a loop
for i in range(len(design_vector_list)):
    mga_low_thrust_problem.fitness(design_vector_list[i])



