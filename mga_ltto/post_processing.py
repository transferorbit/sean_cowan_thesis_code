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

# data_directory = "pp_ltto/EEMJ_gs_ddate/EEMJ_g300p1200_test88/islands/"
# util.pareto_front(dir=data_directory, pmf_as_obj=True) # only if MO of course
# util.hodographic_shaping_visualisation(dir=data_directory, quiver=True, projection='xy')
# util.thrust_propagation(dir=data_directory)
# util.objective_per_generation_visualisation(dir=data_directory, no_of_islands=24)


# data_directory = "pp_ltto/ddate_gs/EEMJ_lb62800_ub63200_test101/"
# util.get_scattered_objectives(data_directory)

data_directory = "pp_ltto/EEMJ_topocomp_24_repeat/EEMJ_g300p1200_test119/islands/"
util.objective_per_generation_visualisation(dir_of_dir=data_directory, no_of_islands=24)
data_directory = "pp_ltto/EEMJ_topocomp_24_repeat/EEMJ_g300p1200_test122/islands/"
util.objective_per_generation_visualisation(dir_of_dir=data_directory, no_of_islands=24)
data_directory = "pp_ltto/EEMJ_topocomp_24_repeat/EEMJ_g300p1200_test124/islands/"
util.objective_per_generation_visualisation(dir_of_dir=data_directory, no_of_islands=24)
data_directory = "pp_ltto/EEMJ_topocomp_24_repeat/EEMJ_g300p1200_test126/islands/"
util.objective_per_generation_visualisation(dir_of_dir=data_directory, no_of_islands=24)



plt.show()
