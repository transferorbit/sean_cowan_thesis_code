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

# data_directory = "pp_mgso/morante_maxga3sr2_spp5gen150pop300/layer_0/islands/island_1/"
# data_directory = "pp_ltto/EVEMJ_verification_gen150pop300/islands/island_3/"
# util.hodographic_shaping_visualisation(dir=data_directory, trajectory_function=util.trajectory_3d)

data_directory = "pp_ltto/EVEMJ_cpu4gen200pop300dsf0/islands/island_0/"
util.pareto_front(dir=data_directory) # only if MO of course
data_directory = "pp_ltto/EVEMJ_cpu4gen200pop300dsf1/islands/island_0/"
util.pareto_front(dir=data_directory) # only if MO of course
plt.show()

# data_directory= "verification/verification_results/minlp_rastrigin/island_0/"
# util.objective_per_generation_visualisation(dir=data_directory)
