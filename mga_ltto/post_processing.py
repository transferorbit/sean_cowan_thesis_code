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
# import sys
# sys.path.insert(0, "/Users/sean/Desktop/tudelft/tudat/tudat-bundle/build/tudatpy")

from tudatpy.kernel.interface import spice
spice.load_standard_kernels()
import mga_low_thrust_utilities as mga_util
from trajectory3d import trajectory_3d

data_directory = "pp_singleruns/EM_verification/"
# data_directory = "pp_ltto/EMJ_checknode_2fp/island_0/"
mga_util.hodographic_shaping_visualisation(dir=data_directory, trajectory_function=mga_util.trajectory_3d)

# data_directory= "verification/verification_results/minlp_rastrigin/island_0/"
# mga_util.objective_per_generation_visualisation(dir=data_directory)
