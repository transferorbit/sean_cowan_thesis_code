'''
Author: Sean Cowan
Purpose: MSc Thesis
Date Created: 02-08-2022

This module performs post processing actions to visualize the optimized results.
'''
###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################


# General imports
import mga_low_thrust_utilities as mga_util

from trajectory3d import trajectory_3d

data_directory = "test_optimization_results/island_0/"

mga_util.hodographic_shaping_visualisation(dir=data_directory, trajectory_function=mga_util.trajectory_3d)


