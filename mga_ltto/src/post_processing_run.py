'''
Author: Sean Cowan
Purpose: MSc Thesis
Date Created: 02-08-2022

This module performs post processing actions to visualize the optimized results.
'''
###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General
import sys
import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

# plt.rcParams['figure.dpi'] = 250
# plt.rcParams['savefig.dpi'] = 250
plt.rcParams['font.size'] = 6
plt.rcParams['font.family'] = "Arial"
plt.rcParams['figure.figsize'] = (12, 12)
#
# Local
from misc.trajectory3d import trajectory_3d
import misc.post_processing_utilities as post

# General imports
# import sys
# sys.path.insert(0, "/Users/sean/Desktop/tudelft/tudat/tudat-bundle/build/tudatpy")

# current_dir = os.getcwd()
# sys.path.append(current_dir) # this only works if you run ltto and mgaso while in the directory that includes those files

# Different types of pp
# data_directory = "pp_ltto/EEMJ_gs_ddate/EEMJ_g300p1200_test88/islands/island_0/"
# post.pareto_front(dir=data_directory, pmf_as_obj=True) # only if MO of course
# post.hodographic_shaping_visualisation(dir=data_directory, quiver=True, projection='xy')
# post.thrust_propagation(dir=data_directory)
# post.objective_per_generation_visualisation(dir=data_directory, no_of_islands=24)


# data_directory = "pp_mgaso/fan2021fast_gs/"
# post.mgaso_scatter(data_directory, fitprop_values=[1.0, 0.75,0.5, 0.25, 0.0], frac_values=[0.1, 0.3])

# quantity='fc_axes'
# post.get_stats(dir_of_dir="pp_ltto/EEEMJ_fpbounds/EEEMJ_fpb5e3_test199/",
#                quantity_to_analyse=quantity, plot_quantity=True, title=2, save_fig=True, bins=20)
# post.get_stats(dir_of_dir="pp_ltto/EEEMJ_fpbounds/EEEMJ_fpb1e4_test198/",
#                quantity_to_analyse=quantity, plot_quantity=True, title=2, save_fig=True, bins=20)
# post.get_stats(dir_of_dir="pp_ltto/EEEMJ_fpbounds/EEEMJ_fpb5e4_test197/",
#                quantity_to_analyse=quantity, plot_quantity=True, title=2, save_fig=True, bins=20)
# post.get_stats(dir_of_dir="pp_ltto/EEEMJ_fpbounds/EEEMJ_fpb1e5_test196/",
#                quantity_to_analyse=quantity, plot_quantity=True, title=2, save_fig=True, bins=20)

# data_directory = "pp_ltto/ddate_gs_60/EJ_lb62000_ub62400_test93/"
# post.get_scattered_objectives(data_directory, add_local_optimisation=True, no_of_islands=24, save_fig=True)
# data_directory = "pp_ltto/ddate_gs_60/EJ_lb62400_ub62800_test94/"
# post.get_scattered_objectives(data_directory, add_local_optimisation=True, no_of_islands=24, save_fig=True)
# data_directory = "pp_ltto/ddate_gs_60/EJ_lb62800_ub63200_test95/"
# post.get_scattered_objectives(data_directory, add_local_optimisation=True, no_of_islands=15, save_fig=True)
#
# data_directory = "pp_ltto/EN_testing/EN_test164/"
# post.get_scattered_objectives(dir_of_dir=data_directory, add_local_optimisation=True, no_of_islands=24, save_fig=True)
# data_directory = "pp_ltto/EN_testing/EJN_test165/"
# post.get_scattered_objectives(dir_of_dir=data_directory, add_local_optimisation=True, no_of_islands=24, save_fig=True)
# data_directory = "pp_ltto/EN_testing/EMJN_test168/"
# post.get_scattered_objectives(dir_of_dir=data_directory, add_local_optimisation=True, no_of_islands=24, save_fig=True)
# data_directory = "pp_ltto/EN_testing/EEMJN_test169/"
# post.get_scattered_objectives(dir_of_dir=data_directory, add_local_optimisation=True, no_of_islands=24, save_fig=True)
# data_directory = "pp_ltto/EN_testing/EEEMJN_test172/"
# post.get_scattered_objectives(dir_of_dir=data_directory, add_local_optimisation=True, no_of_islands=24, save_fig=True)

# data_directory = "pp_validation/bepi_data_test5/bepi_state.dat"
# hs_directory = "pp_validation/bepi_hs_test5/islands/island_0/state_history.dat"
# post.compare_data_to_hs(data_file=data_directory, hs_file=hs_directory)
# post.thrust_propagation(dir=data_directory)

# plt.show()

# if __name__ == "__main__":
#     data_directory = "pp_ltto/EN_testing/EN_test164/"
#     post.perform_local_optimisation(dir_of_dir=data_directory, no_of_islands=24, max_eval=2000, set_verbose=False,
#                                     print_results=False)
#     data_directory = "pp_ltto/EN_testing/EJN_test165/"
#     post.perform_local_optimisation(dir_of_dir=data_directory, no_of_islands=24, max_eval=2000, set_verbose=False,
#                                     print_results=False)
#     data_directory = "pp_ltto/EN_testing/EMJN_test168/"
#     post.perform_local_optimisation(dir_of_dir=data_directory, no_of_islands=24, max_eval=2000, set_verbose=False,
#                                     print_results=False)
#     data_directory = "pp_ltto/EN_testing/EEMJN_test169/"
#     post.perform_local_optimisation(dir_of_dir=data_directory, no_of_islands=24, max_eval=2000, set_verbose=False,
#                                     print_results=False)
#     data_directory = "pp_ltto/EN_testing/EEEMJN_test172/"
#     post.perform_local_optimisation(dir_of_dir=data_directory, no_of_islands=24, max_eval=2000, set_verbose=False,
#                                     print_results=False)
