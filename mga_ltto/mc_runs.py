'''
Author: Sean Cowan
Purpose: MSc Thesis
Date Created: 03-10-2022

This modules performs a monte carlo analysis on design parameters.
'''
###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import numpy as np
import os
import random
import sys
import matplotlib.pyplot as plt

# Tudatpy imports
from tudatpy.kernel import constants

current_dir = os.getcwd()
sys.path.append(current_dir) # this only works if you run ltto and mgso while in the directory that includes those files

# import mga_low_thrust_utilities as util
from src.pygmo_problem import MGALowThrustTrajectoryOptimizationProblem
from src.date_conversion import dateConversion

output_directory = current_dir + '/pp_singleruns'
julian_day = constants.JULIAN_DAY

no_of_points=100
free_param_count=2
manual_tof_bounds = None
dynamic_shaping_functions = False
dynamic_bounds = True
write_results_to_file = True
manual_base_functions = False
zero_revs = True
objectives = ['pmf', 'tof'] #dv, tof, pmf, dmf

transfer_body_order = ["Earth", "Venus", "Earth", "Mars", "Jupiter"]
Isp = 3200
m0 = 1300
dep_date_lb = dateConversion(calendar_date='2009, 8, 1').date_to_mjd()
dep_date_ub = dateConversion(calendar_date='2012, 4, 27').date_to_mjd()
bounds = [[dep_date_lb, 0, 0, 10, 100, 2e5, -10**4, 0],
    [dep_date_ub, 1925, 500, 10000, 10000, 2e7, 10**4, 6]]


#################################################################
# MC RUNS #######################################################
#################################################################


mga_low_thrust_problem = \
MGALowThrustTrajectoryOptimizationProblem(transfer_body_order=transfer_body_order,
          no_of_free_parameters=free_param_count, 
          manual_base_functions=manual_base_functions, 
          objectives=objectives, 
          Isp=Isp,
          m0=m0,
          no_of_points=no_of_points,
          zero_revs=zero_revs,
          dynamic_shaping_functions=dynamic_shaping_functions)

dpv = np.array([941736686.9579611, 1999.9999995570101, 5104.867172316973, 8977607.24797409,
                13565136.927156938, 27479395.34453047, 41926238.55471664, 1319.006091716405,
                6152.304362108564, 889.3643746824156, 6206296.91871, 160066.732009,
                16002170.238, -4539.0, 3013.0, 9965.0, -5532.0, -8039.0, -3313.0, -9980.0,
                9054.0, -6823.0, 4454.0, 4423.0, 8715.0, 9995.0, -9775.0, -9767.0, 8990.0, 1470.0,
                715.0, -9805.0, -9793.0, -7162.0, 8158.0, -8419.0, 7691.0, 0.0, 0.0, 0.0, 0.0])

no_of_legs = len(transfer_body_order) - 1
no_of_gas = len(transfer_body_order) - 2
mc_type = 'all'
number_of_params = 2 * (no_of_gas) # sp and iv
no_of_runs = 2000
dpv_array = np.zeros((no_of_runs, number_of_params))
no_of_invalid_points = 0
if mc_type=='onebyone':
    pmf_array = np.zeros((no_of_runs, number_of_params))
    dv_array = np.zeros((no_of_runs, number_of_params))
else:
    pmf_array = np.zeros((no_of_runs, 1))
    dv_array = np.zeros((no_of_runs, 1))
state_hist_dict = {}
thrust_acceleration_dict = {}


mc_run = True
plot = True
if mc_run and mc_type=='onebyone':
    for k in range(number_of_params):
        parameter_all_runs = np.random.uniform(bounds[0][4 if k < number_of_params // 2 else 5], \
                                               bounds[1][4 if k < number_of_params // 2 else 5], no_of_runs)

        for i in range(no_of_runs):
    
            design_parameter_vector = dpv.copy()
            design_parameter_vector[3 + no_of_legs + k] = parameter_all_runs[i]
            dpv_array[:, k] = parameter_all_runs
    
            try:
                mga_low_thrust_problem.fitness(design_parameter_vector, post_processing=True)
                delta_v = mga_low_thrust_problem.transfer_trajectory_object.delta_v
                # delta_v_per_leg = mga_low_thrust_problem.transfer_trajectory_object.delta_v_per_leg
                tof = mga_low_thrust_problem.transfer_trajectory_object.time_of_flight
                # print(mga_low_thrust_problem.transfer_trajectory_object)
                # print(mga_low_thrust_problem.Isp)
                # print(mga_low_thrust_problem.m0)
                objectives = mga_low_thrust_problem.get_objectives().copy()
                pmf_array[i, k] = objectives[0]
    
                # state_history = \
                # mga_low_thrust_problem.transfer_trajectory_object.states_along_trajectory(no_of_points)
                # thrust_acceleration = \
                # mga_low_thrust_problem.transfer_trajectory_object.inertial_thrust_accelerations_along_trajectory(no_of_points)
                # state_hist_dict[i]=  state_history
                # thrust_acceleration_dict[i]=  thrust_acceleration
            except Exception as inst:
                pmf_array[i, k] = np.nan
                no_of_invalid_points += 1

if mc_run and mc_type == 'all':
    for k in range(number_of_params):
        parameter_all_runs = np.random.uniform(bounds[0][4 if k < number_of_params // 2 else 5], \
                                               bounds[1][4 if k < number_of_params // 2 else 5], no_of_runs)
        dpv_array[:, k] = parameter_all_runs

    for i in range(no_of_runs):

        design_parameter_vector = dpv.copy()
        design_parameter_vector[3 + no_of_legs:3 + no_of_legs + number_of_params] = dpv_array[i, :]

        try:
            mga_low_thrust_problem.fitness(design_parameter_vector, post_processing=True)
            delta_v = mga_low_thrust_problem.transfer_trajectory_object.delta_v
            # delta_v_per_leg = mga_low_thrust_problem.transfer_trajectory_object.delta_v_per_leg
            tof = mga_low_thrust_problem.transfer_trajectory_object.time_of_flight
            # print(mga_low_thrust_problem.transfer_trajectory_object)
            # print(mga_low_thrust_problem.Isp)
            # print(mga_low_thrust_problem.m0)
            objectives = mga_low_thrust_problem.get_objectives().copy()
            pmf_array[i, :] = objectives[0]
            dv_array[i, :] = delta_v

            # state_history = \
            # mga_low_thrust_problem.transfer_trajectory_object.states_along_trajectory(no_of_points)
            # thrust_acceleration = \
            # mga_low_thrust_problem.transfer_trajectory_object.inertial_thrust_accelerations_along_trajectory(no_of_points)
            # state_hist_dict[i]=  state_history
            # thrust_acceleration_dict[i]=  thrust_acceleration
        except Exception as inst:
            pmf_array[i, :] = np.array([np.nan, np.nan])
            no_of_invalid_points += 1

if plot:
    design_variable_names = {0: 'V incoming velocity',
                           1: 'E incoming velocity',
                           2: 'M incoming velocity',
                           3: 'V swingby periapsis',
                           4: 'E swingby periapsis',
                           5: 'M swingby periapsis'}
    objective_names = ['Propellant Mass Fraction', 'Distance']

    for obj in range(1): #number of objectives

        fig, axs = plt.subplots(2, 3, figsize=(14, 8))
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        fig.suptitle('Monte Carlo - one-by-one - Objective: %s - Scaling: Constrained Distance'
                     %(objective_names[obj]))
        for ax_index, ax in enumerate(axs.flatten()):
            ax.ticklabel_format(useOffset=False)
            cs = ax.scatter(dpv_array[:, ax_index], pmf_array[:, ax_index] if mc_type=='onebyone'
                            else pmf_array[:], s=2, c=dv_array[:,ax_index] if mc_type=='onebyone'
                            else dv_array[:])
            cbar = fig.colorbar(cs, ax=ax)
            cbar.ax.set_ylabel(r'$\Delta V$')
            ax.grid()
            ax.set_ylabel('%s [rad]'%(objective_names[obj]))
            ax.set_xlabel(design_variable_names[ax_index])
plt.show()
