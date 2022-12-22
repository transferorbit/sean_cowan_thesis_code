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
sys.path.append(current_dir) # this only works if you run ltto and mgso while in the directory that includes those files
import src.mga_low_thrust_utilities as util
from src.trajectory3d import trajectory_3d
import src.manual_topology as topo

# data_directory = "pp_ltto/EEMJ_gs_ddate/EEMJ_g300p1200_test88/islands/"
# util.pareto_front(dir=data_directory, pmf_as_obj=True) # only if MO of course
# util.hodographic_shaping_visualisation(dir=data_directory, quiver=True, projection='xy')
# util.thrust_propagation(dir=data_directory)
# util.objective_per_generation_visualisation(dir=data_directory, no_of_islands=24)



# data_directory = "pp_ltto/ddate_gs_60/EMJ_lb61600_ub62000_test96/"
# util.get_scattered_objectives(data_directory, add_local_optimisation=True, no_of_islands=24)
# data_directory = "pp_ltto/ddate_gs_60/EMJ_lb62000_ub62400_test97/"
# util.get_scattered_objectives(data_directory, add_local_optimisation=True, no_of_islands=24)
# data_directory = "pp_ltto/ddate_gs_60/EMJ_lb62400_ub62800_test98/"
# util.get_scattered_objectives(data_directory, add_local_optimisation=True, no_of_islands=24)

data_directory = "pp_ltto/EN_testing/EEEMJN_test172/islands/"
util.objective_per_generation_visualisation(dir_of_dir=data_directory, no_of_islands=24, title=2)
# data_directory = "pp_ltto/EN_testing/EMJN_test168/islands/"
# util.objective_per_generation_visualisation(dir_of_dir=data_directory, no_of_islands=24, title=2)
# data_directory = "pp_ltto/EN_testing/EEMJN_test169/islands/"
# util.objective_per_generation_visualisation(dir_of_dir=data_directory, no_of_islands=24, title=2)
# data_directory = "pp_ltto/EN_testing/EEMJN_test170/islands/"
# util.objective_per_generation_visualisation(dir_of_dir=data_directory, no_of_islands=24, title=2)
# data_directory = "pp_ltto/EN_testing/EJN_test166/islands/"
# util.objective_per_generation_visualisation(dir_of_dir=data_directory, no_of_islands=24, title=2)


# if __name__ == "__main__":
#     data_directory = "pp_ltto/ddate_gs_60/EMJ_lb61600_ub62000_test96/61600_61660/"
#     topo.perform_local_optimisation(dir_of_dir=data_directory, no_of_islands=24, max_eval=2000, set_verbose=False,
#                                     print_results=False)
#     data_directory = "pp_ltto/ddate_gs_60/EMJ_lb61600_ub62000_test96/61660_61720/"
#     topo.perform_local_optimisation(dir_of_dir=data_directory, no_of_islands=24, max_eval=2000, set_verbose=False,
#                                     print_results=False)
#     data_directory = "pp_ltto/ddate_gs_60/EMJ_lb61600_ub62000_test96/61720_61780/"
#     topo.perform_local_optimisation(dir_of_dir=data_directory, no_of_islands=24, max_eval=2000, set_verbose=False,
#                                     print_results=False)
#     data_directory = "pp_ltto/ddate_gs_60/EMJ_lb61600_ub62000_test96/61780_61840/"
#     topo.perform_local_optimisation(dir_of_dir=data_directory, no_of_islands=24, max_eval=2000, set_verbose=False,
#                                     print_results=False)
#     data_directory = "pp_ltto/ddate_gs_60/EMJ_lb61600_ub62000_test96/61840_61900/"
#     topo.perform_local_optimisation(dir_of_dir=data_directory, no_of_islands=24, max_eval=2000, set_verbose=False,
#                                     print_results=False)
#     data_directory = "pp_ltto/ddate_gs_60/EMJ_lb61600_ub62000_test96/61900_61960/"
#     topo.perform_local_optimisation(dir_of_dir=data_directory, no_of_islands=24, max_eval=2000, set_verbose=False,
#                                     print_results=False)
#     data_directory = "pp_ltto/ddate_gs_60/EMJ_lb61600_ub62000_test96/61960_62020/"
#     topo.perform_local_optimisation(dir_of_dir=data_directory, no_of_islands=24, max_eval=2000, set_verbose=False,
#                                     print_results=False)
#
#     data_directory = "pp_ltto/ddate_gs_60/EMJ_lb62000_ub62400_test97/62000_62060/"
#     topo.perform_local_optimisation(dir_of_dir=data_directory, no_of_islands=24, max_eval=2000, set_verbose=False,
#                                     print_results=False)
#     data_directory = "pp_ltto/ddate_gs_60/EMJ_lb62000_ub62400_test97/62060_62120/"
#     topo.perform_local_optimisation(dir_of_dir=data_directory, no_of_islands=24, max_eval=2000, set_verbose=False,
#                                     print_results=False)
#     data_directory = "pp_ltto/ddate_gs_60/EMJ_lb62000_ub62400_test97/62120_62180/"
#     topo.perform_local_optimisation(dir_of_dir=data_directory, no_of_islands=24, max_eval=2000, set_verbose=False,
#                                     print_results=False)
#     data_directory = "pp_ltto/ddate_gs_60/EMJ_lb62000_ub62400_test97/62180_62240/"
#     topo.perform_local_optimisation(dir_of_dir=data_directory, no_of_islands=24, max_eval=2000, set_verbose=False,
#                                     print_results=False)
#     data_directory = "pp_ltto/ddate_gs_60/EMJ_lb62000_ub62400_test97/62240_62300/"
#     topo.perform_local_optimisation(dir_of_dir=data_directory, no_of_islands=24, max_eval=2000, set_verbose=False,
#                                     print_results=False)
#     data_directory = "pp_ltto/ddate_gs_60/EMJ_lb62000_ub62400_test97/62300_62360/"
#     topo.perform_local_optimisation(dir_of_dir=data_directory, no_of_islands=24, max_eval=2000, set_verbose=False,
#                                     print_results=False)
#     data_directory = "pp_ltto/ddate_gs_60/EMJ_lb62000_ub62400_test97/62360_62420/"
#     topo.perform_local_optimisation(dir_of_dir=data_directory, no_of_islands=24, max_eval=2000, set_verbose=False,
#                                     print_results=False)
#
#     data_directory = "pp_ltto/ddate_gs_60/EMJ_lb62400_ub62800_test98/62400_62460/"
#     topo.perform_local_optimisation(dir_of_dir=data_directory, no_of_islands=24, max_eval=2000, set_verbose=False,
#                                     print_results=False)
#     data_directory = "pp_ltto/ddate_gs_60/EMJ_lb62400_ub62800_test98/62460_62520/"
#     topo.perform_local_optimisation(dir_of_dir=data_directory, no_of_islands=24, max_eval=2000, set_verbose=False,
#                                     print_results=False)
#     data_directory = "pp_ltto/ddate_gs_60/EMJ_lb62400_ub62800_test98/62520_62580/"
#     topo.perform_local_optimisation(dir_of_dir=data_directory, no_of_islands=24, max_eval=2000, set_verbose=False,
#                                     print_results=False)
#     data_directory = "pp_ltto/ddate_gs_60/EMJ_lb62400_ub62800_test98/62580_62640/"
#     topo.perform_local_optimisation(dir_of_dir=data_directory, no_of_islands=24, max_eval=2000, set_verbose=False,
#                                     print_results=False)
#     data_directory = "pp_ltto/ddate_gs_60/EMJ_lb62400_ub62800_test98/62640_62700/"
#     topo.perform_local_optimisation(dir_of_dir=data_directory, no_of_islands=24, max_eval=2000, set_verbose=False,
#                                     print_results=False)
#     data_directory = "pp_ltto/ddate_gs_60/EMJ_lb62400_ub62800_test98/62700_62760/"
#     topo.perform_local_optimisation(dir_of_dir=data_directory, no_of_islands=24, max_eval=2000, set_verbose=False,
#                                     print_results=False)
#     data_directory = "pp_ltto/ddate_gs_60/EMJ_lb62400_ub62800_test98/62760_62820/"
#     topo.perform_local_optimisation(dir_of_dir=data_directory, no_of_islands=24, max_eval=2000, set_verbose=False,
#                                     print_results=False)



plt.show()
