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

### TRAJECTORY VISUALIZATION ###
data_directory = "pp_ltto/EVEMJ_cpu4gen150pop300np50/islands/island_0/"
util.hodographic_shaping_visualisation(dir=data_directory, trajectory_function=util.trajectory_3d)
data_directory = "pp_ltto/EVEMJ_cpu4gen150pop300np50/islands/island_1/"
util.hodographic_shaping_visualisation(dir=data_directory, trajectory_function=util.trajectory_3d)

### PARETO FRONT ###
# data_directory = "pp_ltto/EVEMJ_cpu4gen200pop300dsf0/islands/island_0/"
# util.pareto_front(dir=data_directory) # only if MO of course
# data_directory = "pp_ltto/EVEMJ_cpu4gen150pop300fp2_plot/islands/island_0/"
# util.pareto_front(dir=data_directory) # only if MO of course

### OBJECTIVE PER GEN ###
# data_directory= "verification/verification_results/minlp_rastrigin/island_0/"
# util.objective_per_generation_visualisation(dir=data_directory)

### THRUST ACCELERATION PROFILE ###
# data_directory = "pp_ltto/EVEMJ_cpu8gen150pop300np200/islands/island_0/"
# util.thrust_propagation(dir=data_directory)
plt.show()
