'''
Author: Sean Cowan
Purpose: MSc Thesis
Date Created: 03-10-2022

This modules performs a monte carlo analysis on design parameters for the all angle Problem class
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
sys.path.append(current_dir) # this only works if you run ltto and mgaso while in the directory that includes those files

# import mga_low_thrust_utilities as util
from src.pygmo_problem import MGALowThrustTrajectoryOptimizationProblem
from src.date_conversion import dateConversion

output_directory = current_dir + '/pp_singleruns'
julian_day = constants.JULIAN_DAY
no_of_points=100

bound_names= ['Departure date [mjd2000]', 'Departure velocity [m/s]', 'Departure in-plane angle [rad]', 
              'Departure out-of-plane angle [rad]', 'Arrival velocity [m/s]', 'Arrival in-plane angle [rad]', 
              'Arrival out-of-plane angle [rad]', 'Time of Flight [s]', 'Incoming velocity [m/s]', 
              'Swingby periapsis [m]', 'Orbit orientation angle [rad]', 'Swingby in-plane Angle [rad]', 
              'Swingby out-of-plane angle [rad]', 'Free coefficient [-]', 'Number of revolutions [-]']

manual_tof_bounds = None
dynamic_shaping_functions = False
dynamic_bounds = False
write_results_to_file = True
manual_base_functions = False
zero_revs = False
objectives = ['dv'] #dv, tof, pmf, dmf

transfer_body_order = ["Earth", "Mars", "Jupiter"]
free_param_count=2

Isp = 3200
m0 = 1300
jd2000_dep_date_lb = 61420 - 51544.5
jd2000_dep_date_ub = 63382 - 51544.5

departure_date = (jd2000_dep_date_lb, jd2000_dep_date_ub)
# departure_date=  (11263.51 - 100, 11263.51 + 100) #from island_0 EM_gen150
departure_velocity = (0, 0)
departure_inplane_angle = (0, 0)
departure_outofplane_angle = (0, 0)
arrival_velocity = (0, 3000)
arrival_inplane_angle = (0, 2 * np.pi)
arrival_outofplane_angle = (-np.pi / 4, np.pi / 4)
time_of_flight = (100, 4500)
incoming_velocity = (0, 5000)
swingby_periapsis = (2e5, 2e8)
orbit_ori_angle = (0, 2 * np.pi)
swingby_inplane_angle = (0, 2 * np.pi)
swingby_outofplane_angle = (-np.pi / 4, np.pi / 4)
free_coefficient = (-3e4, 3e4)
number_of_revs = (0, 2)
Isp = 3200 #guess
m0 = 1300 #guess
manual_tof_bounds = [[100, 500], [1000, 4000]]

bounds = [[departure_date[0], departure_velocity[0], arrival_velocity[0], time_of_flight[0],
           incoming_velocity[0], swingby_periapsis[0], free_coefficient[0], number_of_revs[0]], 
          [departure_date[1], departure_velocity[1], arrival_velocity[1], time_of_flight[1],
           incoming_velocity[1], swingby_periapsis[1], free_coefficient[1], number_of_revs[1]]]


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

dpv = np.array([918466644.0595107, 0.0, 0.0, 33848923.031928234, 95659980.40849155,
                682.324024017103, 7427748.992780652, -21256.760764444123, 12931.363490562802,
                -20335.891606953475, -26274.083230499542, 24330.702672446605, -22388.64308993075,
                23057.16796329622, 10533.80907344227, -793.6374730770622, 25407.13856662316,
                18544.402594118554, -5880.084251002834, 0.0, 0.0])

no_of_legs = len(transfer_body_order) - 1
no_of_gas = len(transfer_body_order) - 2
mc_type = 'all'
number_of_params = 2 * (no_of_gas) # sp and iv
number_of_params = 1
no_of_runs = 1000
no_of_invalid_points = 0

bound_indices = [3, 4, 5, 7]

def get_dpv_indices(bound_indices, transfer_body_order):
    """
    Inputs
    ------
    bound_indices : list(int), the list of bound indices that are supposed to be analysed with MC
    transfer_body_order : list(str), the transfer body order for no_of_legs and no_of_gas

    Returns
    -------
    bound_index_dict : dict, iteration count and bound index value
    indices : list(int), list of dpv indices to be analyzed
    """
    no_of_legs = len(transfer_body_order) - 1
    no_of_gas = len(transfer_body_order) - 2

    indices = []
    bound_index_list = []

    for it, index in enumerate(bound_indices):
        if index in [3, 6, 7]:
            if index == 3:
                for i in range(no_of_legs):
                    indices += [3 + i]
                    bound_index_list += [index]
            if index == 6:
                for i in range(6 * no_of_legs):
                    indices += [3 + no_of_legs + 2 * no_of_gas + i]
                    bound_index_list += [index]
            if index == 7:
                for i in range(no_of_legs):
                    indices += [3 + no_of_legs + 2 * no_of_gas + 6 * no_of_legs + i]
                    bound_index_list += [index]
        elif index in [4, 5]:
            if index == 4:
                for i in range(no_of_gas):
                    indices += [3 + no_of_legs + i]
                    bound_index_list += [index]
            if index == 5:
                for i in range(no_of_gas):
                    indices += [3 + no_of_legs + no_of_gas + i]
                    bound_index_list += [index]
        else:
            indices += [index]
            bound_index_list += [index]

    return bound_index_list, indices


# def get_specific_bounds(parameter_number, bound_indexes : list(int), bounds):
#
#     """
#     Input
#     ------
#     parameter_number : int, determines what index is currently going to be changed
#     bound_indexes : list(int), says what indexes of 
#     """
#
#     for it, bound_lb in enumerate(bounds[0]):
#         bound_ub = bounds[1][it]
#         for bi in bound_indexes:
#             if it == bi && parameter_number < :
#                 current_bound_index
#
#     return bounds[0][current_bound_index], bounds[1][current_bound_index]
#
# bound_index_dict, indices = get_dpv_indices(bound_indices, transfer_body_order)
# print(indices)
# print(number_of_params)

mc_run = True
plot = True

if mc_run and mc_type=='onebyone':
    bound_index_list, indices = get_dpv_indices(bound_indices, transfer_body_order)
    obj1_array = np.zeros((no_of_runs, len(indices)))
    obj2_array = np.zeros((no_of_runs, len(indices)))
    dpv_array = np.zeros((no_of_runs, len(indices)))

    for k, index in enumerate(indices):
        print(f"{index} - {bound_names[bound_index_list[k]]}")
        lower_bound, upper_bound = bounds[0][bound_index_list[k]], bounds[1][bound_index_list[k]]
        parameter_all_runs = np.random.uniform(lower_bound, upper_bound, no_of_runs)

        for i in range(no_of_runs):
    
            design_parameter_vector = dpv.copy()
            design_parameter_vector[indices[k]] = parameter_all_runs[i]
            dpv_array[:, k] = parameter_all_runs
    
            try:
                mga_low_thrust_problem.fitness(design_parameter_vector, post_processing=True)
                delta_v = mga_low_thrust_problem.transfer_trajectory_object.delta_v
                tof = mga_low_thrust_problem.transfer_trajectory_object.time_of_flight

                obj1_array[i, k] = delta_v
    
            except Exception as inst:
                obj1_array[i, k] = np.nan
                no_of_invalid_points += 1


if mc_run and mc_type == 'all':
    bound_index_list, indices = get_dpv_indices(bound_indices, transfer_body_order)
    obj1_array = np.zeros(no_of_runs)
    obj2_array = np.zeros(no_of_runs)
    dpv_array = np.zeros((no_of_runs, len(indices)))

    for k, index in enumerate(indices):
        print(f"{index} - {bound_names[bound_index_list[k]]}")
        lower_bound, upper_bound = bounds[0][bound_index_list[k]], bounds[1][bound_index_list[k]]
        parameter_all_runs = np.random.uniform(lower_bound, upper_bound, no_of_runs)

        dpv_array[:, k] = parameter_all_runs

    for i in range(no_of_runs):

        design_parameter_vector = dpv.copy()
        for (index, replacement) in zip(indices, dpv_array[i, :]):
            design_parameter_vector[index] = replacement

        try:
            mga_low_thrust_problem.fitness(design_parameter_vector, post_processing=True)
            delta_v = mga_low_thrust_problem.transfer_trajectory_object.delta_v
            tof = mga_low_thrust_problem.transfer_trajectory_object.time_of_flight

            obj1_array[i] = delta_v

        except Exception as inst:
            obj1_array[i] = np.nan
            no_of_invalid_points += 1


if plot:
    objective_names = [r'$\Delta V$ [m/s]', 'Distance [m]']
    obj_arrays = [obj1_array, obj2_array]

    for obj in [0]: #number of objectives

        it2 = 0
        for it, bound in enumerate(bound_indices):
            current_bound_name = bound_names[bound]

            if bound in [3, 7]:
                fig, axs = plt.subplots(1, no_of_legs, figsize=(14, 8))
            elif bound in [6]:
                fig, axs = plt.subplots(3, 2, figsize=(14, 8)) # 6 free coefficients
            elif bound in [4, 5]:
                fig, axs = plt.subplots(1, no_of_gas, figsize=(14, 8))
            else:
                fig, axs = plt.subplots(1, 1, figsize=(14, 8))


            plt.subplots_adjust(wspace=0.5, hspace=0.5)
            fig.suptitle(
            f"MC - {mc_type} - Obj: {objective_names[obj]} - Current Bound = {current_bound_name}"
            )
            axs = axs.flatten() if type(axs) == np.ndarray else np.array([axs])
            for ax_index, ax in enumerate(axs):
                if ax_index == len(indices):
                    break
                ax.ticklabel_format(useOffset=False)
                cs = ax.scatter(dpv_array[:, it2], obj_arrays[obj][:, it2] if mc_type=='onebyone'
                                else obj_arrays[obj], s=2)#, c=obj_arrays[obj][:,ax_index])
                # cbar = fig.colorbar(cs, ax=ax)
                # cbar.ax.set_ylabel(objective_names[obj])
                ax.grid()
                ax.set_ylabel(objective_names[obj])
                ax.set_xlabel(f"Variable : {bound_names[bound_index_list[it2]]} - Leg : {ax_index}")
                ax.set_yscale('log')
                it2 += 1

plt.show()

