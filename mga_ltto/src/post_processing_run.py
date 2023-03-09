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
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = "Arial"
plt.rcParams['figure.figsize'] = (12,8)
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

dir1 = "pp_mgaso/mgaso_2rec_customtop_g300p1200/mgaso_2rec_seed1/"
dir2 = "pp_mgaso/mgaso_2rec_customtop_g300p1200/mgaso_2rec_seed2/"
# dir3 = "pp_mgaso/mgaso_2rec_customtop_g300p1200/mgaso_2rec_seed3/"
dir3 = None
post.mgaso_scatter_multi(dir1, dir2, dir3, fitprop_values=[1.0, 0.5], fitprop_itbs_values=[1.0, 0.5], 
                         frac_values=[0.1, 0.4, 0.7], save_fig=False, title='2rec_customtop_g300p1200')

# Iteration 3 custom topology verification
# data_directory = "pp_mgaso/mgaso_test129/"
# post.objective_per_generation_visualisation(dir=data_directory, no_of_islands=70, title='topoNoneg200p800rec1',
#                                             seq=0, ips=14, num_gen=200, save_fig=True)
# data_directory = "pp_mgaso/mgaso_test128/"
# post.objective_per_generation_visualisation(dir=data_directory, no_of_islands=70, title='topo01g200p800rec1',
#                                             seq=0, ips=14, num_gen=200, save_fig=True)
# plt.show()

# # 60 day grid search ltto 
# dir1 = "pp_ltto/ddate_gs_60/EJ_lb62000_ub62400_test93/"
# dir3 = "pp_ltto/ddate_gs_60/EJ_lb62800_ub63200_test95/"
# dir2 = "pp_ltto/ddate_gs_60/EJ_lb62400_ub62800_test94/"
# dir1 = "pp_ltto/ddate_gs_60/EMJ_lb61600_ub62000_test96/"
# dir2 = "pp_ltto/ddate_gs_60/EMJ_lb62000_ub62400_test97/"
# dir3 = "pp_ltto/ddate_gs_60/EMJ_lb62400_ub62800_test98/"
# dir1 = "pp_ltto/ddate_gs_60/EEMJ_lb61200_ub61600_test99/"
# dir2 = "pp_ltto/ddate_gs_60/EEMJ_lb61600_ub62000_test100/"
# dir3 = "pp_ltto/ddate_gs_60/EEMJ_lb62000_ub62400_test101/"
# dir1 = "pp_ltto/ddate_gs_60/EEEMJ_lb61600_ub62000_test102/"
# dir2 = "pp_ltto/ddate_gs_60/EEEMJ_lb62000_ub62400_test103/"
# dir3 = "pp_ltto/ddate_gs_60/EEEMJ_lb62400_ub62800_test104/"
# post.get_scattered_objectives_extended(dir1, dir2, dir3, add_local_optimisation=True, no_of_islands=24, save_fig=True)
# plt.show()

# 400 day grid search ltto 
# dir1 = "pp_ltto/ddate_gs_400/EJ_lb62000_ub62400_test105/"
# dir2 = "pp_ltto/ddate_gs_400/EJ_lb62400_ub62800_test106/"
# dir3 = "pp_ltto/ddate_gs_400/EJ_lb62800_ub63200_test107/"
# dir1 = "pp_ltto/ddate_gs_400/EMJ_lb61600_ub62000_test108/"
# dir2 = "pp_ltto/ddate_gs_400/EMJ_lb62000_ub62400_test109/"
# dir3 = "pp_ltto/ddate_gs_400/EMJ_lb62400_ub62800_test110/"
# dir1 = "pp_ltto/ddate_gs_400/EEMJ_lb61200_ub61600_test111/"
# dir2 = "pp_ltto/ddate_gs_400/EEMJ_lb61600_ub62000_test112/"
# dir3 = "pp_ltto/ddate_gs_400/EEMJ_lb62000_ub62400_test113/"
# dir1 = "pp_ltto/ddate_gs_400/EEEMJ_lb61600_ub62000_test114/"
# dir2 = "pp_ltto/ddate_gs_400/EEEMJ_lb62000_ub62400_test115/"
# dir3 = "pp_ltto/ddate_gs_400/EEEMJ_lb62400_ub62800_test116/"
# post.get_scattered_objectives_extended(dir1, dir2, dir3, add_local_optimisation=False, no_of_islands=24, save_fig=True)
# plt.show()

# get_stats
# plt.rcParams['figure.figsize'] = (8,8)
# plt.rcParams['font.size'] = 18
# quantity='fc_axes'
# # post.get_stats(dir_of_dir="pp_ltto/EEEMJ_popsizegen_gs_24islands/EEEMJ_pop1200gen300_test195/",
# #                quantity_to_analyse=quantity, plot_quantity=True, title=2, save_fig=True, bins=20)
# # post.get_stats(dir_of_dir="pp_ltto/EEEMJ_fpbounds/EEEMJ_fpb1e4_test198/",
# #                quantity_to_analyse=quantity, plot_quantity=True, title=2, save_fig=True, bins=20)
# # post.get_stats(dir_of_dir="pp_ltto/EEEMJ_fpbounds/EEEMJ_fpb5e4_test197/",
# #                quantity_to_analyse=quantity, plot_quantity=True, title=2, save_fig=True, bins=20)
# post.get_stats(dir_of_dir="pp_ltto/EEEMJ_fpbounds/EEEMJ_fpb1e5_test196/",
#                quantity_to_analyse=quantity, plot_quantity=True, title=2, save_fig=True, bins=20)
# plt.show()


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

# data_directory = "pp_validation/bepi_data_test20/bepi_state.dat"
# hs_directory = "pp_validation/bepi_hs_test36/islands/island_0/state_history.dat"
# hs_dir = "pp_validation/bepi_hs_test37/islands/island_0/"
# # post.compare_data_to_hs(data_file=data_directory, hs_file=hs_directory)
# post.thrust_propagation(dir=hs_dir, mass=4100)
# plt.show()

# data_directory = "pp_validation/bepi_data_test5/bepi_state.dat"
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
