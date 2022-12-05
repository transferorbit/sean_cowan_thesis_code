'''
Author: Sean Cowan
Purpose: MSc Thesis
Date Created: 02-08-2022

This module performs post processing actions to visualize the optimized results.
'''
###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

import sys
import os
import matplotlib.pyplot as plt

# General imports
# import sys
# sys.path.insert(0, "/Users/sean/Desktop/tudelft/tudat/tudat-bundle/build/tudatpy")

current_dir = os.getcwd()
sys.path.append(current_dir) # this only works if you run ltto and mgso while in the directory that includes those files
import src.mga_low_thrust_utilities as util
from src.trajectory3d import trajectory_3d

# util.pareto_front(dir=data_directory, pmf_as_obj=True) # only if MO of course
# util.hodographic_shaping_visualisation(dir=data_directory, quiver=True, projection='xy')
# util.thrust_propagation(dir=data_directory)
# util.objective_per_generation_visualisation(dir=data_directory)


data_directory = "pp_ltto/comp_aa_0fp/EJ_g300p1200_test30/islands/"
util.objective_per_generation_visualisation(dir_of_dir=data_directory)
data_directory = "pp_ltto/comp_aa_1fp/EJ_g300p1200_test26/islands/"
util.objective_per_generation_visualisation(dir_of_dir=data_directory)
data_directory = "pp_ltto/comp_aa_2fp/EJ_g300p1200_test34/islands/"
util.objective_per_generation_visualisation(dir_of_dir=data_directory)
# util.hodographic_shaping_visualisation(dir=data_directory, quiver=False)
# util.thrust_propagation(dir=data_directory)


plt.show()
