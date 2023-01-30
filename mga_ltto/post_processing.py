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
import numpy as np

# General imports
# import sys
# sys.path.insert(0, "/Users/sean/Desktop/tudelft/tudat/tudat-bundle/build/tudatpy")


current_dir = os.getcwd()
sys.path.append(current_dir) # this only works if you run ltto and mgaso while in the directory that includes those files
import src.mga_low_thrust_utilities as util
from src.trajectory3d import trajectory_3d
import src.manual_topology as topo

# Different types of pp
# data_directory = "pp_ltto/EEMJ_gs_ddate/EEMJ_g300p1200_test88/islands/"
# util.pareto_front(dir=data_directory, pmf_as_obj=True) # only if MO of course
# util.hodographic_shaping_visualisation(dir=data_directory, quiver=True, projection='xy')
# util.thrust_propagation(dir=data_directory)
# util.objective_per_generation_visualisation(dir=data_directory, no_of_islands=24)

# quantity='fc_axes'
# util.get_stats(dir_of_dir="pp_ltto/EEEMJ_fpbounds/EEEMJ_fpb5e3_test199/",
#                quantity_to_analyse=quantity, plot_quantity=True, title=2, save_fig=True, bins=20)
# util.get_stats(dir_of_dir="pp_ltto/EEEMJ_fpbounds/EEEMJ_fpb1e4_test198/",
#                quantity_to_analyse=quantity, plot_quantity=True, title=2, save_fig=True, bins=20)
# util.get_stats(dir_of_dir="pp_ltto/EEEMJ_fpbounds/EEEMJ_fpb5e4_test197/",
#                quantity_to_analyse=quantity, plot_quantity=True, title=2, save_fig=True, bins=20)
# util.get_stats(dir_of_dir="pp_ltto/EEEMJ_fpbounds/EEEMJ_fpb1e5_test196/",
#                quantity_to_analyse=quantity, plot_quantity=True, title=2, save_fig=True, bins=20)

# data_directory = "pp_ltto/ddate_gs_60/EJ_lb62000_ub62400_test93/"
# util.get_scattered_objectives(data_directory, add_local_optimisation=True, no_of_islands=24, save_fig=True)
# data_directory = "pp_ltto/ddate_gs_60/EJ_lb62400_ub62800_test94/"
# util.get_scattered_objectives(data_directory, add_local_optimisation=True, no_of_islands=24, save_fig=True)
# data_directory = "pp_ltto/ddate_gs_60/EJ_lb62800_ub63200_test95/"
# util.get_scattered_objectives(data_directory, add_local_optimisation=True, no_of_islands=15, save_fig=True)
#
data_directory = "pp_ltto/EN_testing/EN_test164/"
util.get_scattered_objectives(dir_of_dir=data_directory, add_local_optimisation=True, no_of_islands=24, save_fig=True)
data_directory = "pp_ltto/EN_testing/EJN_test165/"
util.get_scattered_objectives(dir_of_dir=data_directory, add_local_optimisation=True, no_of_islands=24, save_fig=True)
data_directory = "pp_ltto/EN_testing/EMJN_test168/"
util.get_scattered_objectives(dir_of_dir=data_directory, add_local_optimisation=True, no_of_islands=24, save_fig=True)
data_directory = "pp_ltto/EN_testing/EEMJN_test169/"
util.get_scattered_objectives(dir_of_dir=data_directory, add_local_optimisation=True, no_of_islands=24, save_fig=True)
data_directory = "pp_ltto/EN_testing/EEEMJN_test172/"
util.get_scattered_objectives(dir_of_dir=data_directory, add_local_optimisation=True, no_of_islands=24, save_fig=True)


# if __name__ == "__main__":
#     data_directory = "pp_ltto/EN_testing/EN_test164/"
#     topo.perform_local_optimisation(dir_of_dir=data_directory, no_of_islands=24, max_eval=2000, set_verbose=False,
#                                     print_results=False)
#     data_directory = "pp_ltto/EN_testing/EJN_test165/"
#     topo.perform_local_optimisation(dir_of_dir=data_directory, no_of_islands=24, max_eval=2000, set_verbose=False,
#                                     print_results=False)
#     data_directory = "pp_ltto/EN_testing/EMJN_test168/"
#     topo.perform_local_optimisation(dir_of_dir=data_directory, no_of_islands=24, max_eval=2000, set_verbose=False,
#                                     print_results=False)
#     data_directory = "pp_ltto/EN_testing/EEMJN_test169/"
#     topo.perform_local_optimisation(dir_of_dir=data_directory, no_of_islands=24, max_eval=2000, set_verbose=False,
#                                     print_results=False)
#     data_directory = "pp_ltto/EN_testing/EEEMJN_test172/"
#     topo.perform_local_optimisation(dir_of_dir=data_directory, no_of_islands=24, max_eval=2000, set_verbose=False,
#                                     print_results=False)
