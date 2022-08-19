"""
Copyright (c) 2010-2022, Delft University of Technology
All rights reserved

This file is part of the Tudat. Redistribution and use in source and
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.

TUDATPY EXAMPLE APPLICATION: Mulitple Gravity Assist and Deep Space Maneuver transfers
FOCUS:                       Analysis of MGA-DSM transfer trajectories
"""

###############################################################################
# TUDATPY EXAMPLE APPLICATION: MGA-DSM transfers                   ############
###############################################################################

""" ABSTRACT.

This example demonstrates how Multiple Gravity Assist (MGA) transfer trajectories 
 with, or without, Deep Space Maneuvers (DSM) can be simulated. Both an example with 
 and an example without DSMs is provided. 
In addition, these example show how the result, such as Delta V and Time of Flight
 values can be retrieved from the transfer object.

"""


###############################################################################
# IMPORT STATEMENTS ###########################################################
###############################################################################
# import build_directory

import numpy as np
import matplotlib.pyplot as plt
from tudatpy.kernel.trajectory_design import transfer_trajectory
from tudatpy.kernel.trajectory_design import shape_based_thrust
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.util import result2array
from tudatpy.kernel import constants
from tudatpy.kernel.math import root_finders

########################################################################################################################
def mga_without_dsm():

    print("######################### MGA with unpowered legs")

    ###########################################################################
    # DEFINE TRANSFER SETTINGS ################################################
    ###########################################################################

    # Simplified bodies
    bodies = environment_setup.create_simplified_system_of_bodies()
    central_body = 'Sun'

    # Define order of bodies (nodes) for gravity assists
    transfer_body_order = [
        'Earth', 'Venus', 'Venus', 'Earth',  'Jupiter',  'Saturn']

    # Define departure and insertion orbit
    departure_semi_major_axis = np.inf
    departure_eccentricity = 0.

    arrival_semi_major_axis = 1.0895e8 / 0.02
    arrival_eccentricity = 0.98

    # Define type of leg between bodies
    leg_type = transfer_trajectory.unpowered_unperturbed_leg_type

    ###########################################################################
    # CREATE TRANSFER SETTINGS AND OBJECT #####################################
    ###########################################################################

    # Define trajectory settings
    transfer_leg_settings, transfer_node_settings = transfer_trajectory.mga_settings_unpowered_unperturbed_legs(
        transfer_body_order,
        departure_orbit=(departure_semi_major_axis, departure_eccentricity),
        arrival_orbit=(arrival_semi_major_axis, arrival_eccentricity))

    # Create transfer calculation object
    transfer_trajectory_object = transfer_trajectory.create_transfer_trajectory(
        bodies,
        transfer_leg_settings,
        transfer_node_settings,
        transfer_body_order,
        central_body)

    ###########################################################################
    # DEFINE TRANSFER PARAMETERS ##############################################
    ###########################################################################

    # Define times at each node
    julian_day = constants.JULIAN_DAY
    node_times = list()
    node_times.append((-789.8117 - 0.5) * julian_day)
    node_times.append(node_times[0] + 158.302027105278 * julian_day)
    node_times.append(node_times[1] + 449.385873819743 * julian_day)
    node_times.append(node_times[2] + 54.7489684339665 * julian_day)
    node_times.append(node_times[3] + 1024.36205846918 * julian_day)
    node_times.append(node_times[4] + 4552.30796805542 * julian_day)

    # Define free parameters per leg (now: none)
    leg_free_parameters = list()
    for i in range(len(transfer_body_order)-1):
        leg_free_parameters.append(np.zeros(0))

    # Define free parameters per node (now: none)
    node_free_parameters = list()
    for i in range(len(transfer_body_order)):
        node_free_parameters.append(np.zeros(0))

    ###########################################################################
    # EVALUATE TRANSFER #######################################################
    ###########################################################################

    # Evaluate transfer with given parameters
    transfer_trajectory_object.evaluate(node_times, leg_free_parameters, node_free_parameters)

    # Extract and print computed Delta V and time of flight
    print('Delta V [m/s]: ', transfer_trajectory_object.delta_v)
    print('Time of flight [day]: ', transfer_trajectory_object.time_of_flight / julian_day)
    print()
    print('Delta V per leg [m/s] : ', transfer_trajectory_object.delta_v_per_leg)
    print('Delta V per node [m/s] : ', transfer_trajectory_object.delta_v_per_node)

    transfer_trajectory.print_parameter_definitions(transfer_leg_settings, transfer_node_settings)

    # Extract state and thrust acceleration history
    state_history = transfer_trajectory_object.states_along_trajectory(500)
    thrust_acceleration_history = transfer_trajectory_object.inertial_thrust_accelerations_along_trajectory(500)

    # Plot state history
    # fly_by_states = np.array([state_history[node_times[i]] for i in range(len(node_times))])
    # state_history = result2array(state_history)
    # au = 1.5e11

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(state_history[:, 1] / au, state_history[:, 2] / au, state_history[:, 3] / au)
    # ax.scatter(fly_by_states[0, 0] / au, fly_by_states[0, 1] / au, fly_by_states[0, 2] / au, color='blue', label='Earth departure')
    # ax.scatter(fly_by_states[1, 0] / au, fly_by_states[1, 1] / au, fly_by_states[1, 2] / au, color='brown', label='Venus fly-by')
    # ax.scatter(fly_by_states[2, 0] / au, fly_by_states[2, 1] / au, fly_by_states[2, 2] / au, color='brown', label='Venus fly-by')
    # ax.scatter(fly_by_states[3, 0] / au, fly_by_states[3, 1] / au, fly_by_states[3, 2] / au, color='green', label='Earth fly-by')
    # ax.scatter(fly_by_states[4, 0] / au, fly_by_states[4, 1] / au, fly_by_states[4, 2] / au, color='peru', label='Jupiter fly-by')
    # ax.scatter(fly_by_states[5, 0] / au, fly_by_states[5, 1] / au, fly_by_states[5, 2] / au, color='red', label='Saturn arrival')
    # ax.scatter([0], [0], [0], color='orange', label='Sun')
    # ax.set_xlabel('x wrt Sun [AU]')
    # ax.set_ylabel('y wrt Sun [AU]')
    # ax.set_zlabel('z wrt Sun [AU]')
    # ax.set_xlim([-10.5, 2.5])
    # ax.set_ylim([-8.5, 4.5])
    # ax.set_zlim([-6.5, 6.5])
    # ax.legend(bbox_to_anchor=[1.15, 1])
    # plt.show()

    return 0

########################################################################################################################
def mga_with_dsm( ):

    print("######################### MGA with velocity-based DSM legs")

    ###########################################################################
    # DEFINE TRANSFER SETTINGS ################################################
    ###########################################################################

    # Simplified bodies
    bodies = environment_setup.create_simplified_system_of_bodies()

    # Define order of bodies (nodes)
    transfer_body_order = ['Earth', 'Earth', 'Venus', 'Venus',  'Mercury']

    # Define type of leg between bodies
    leg_type = transfer_trajectory.dsm_velocity_based_leg_type

    ###########################################################################
    # CREATE TRANSFER SETTINGS AND OBJECT #####################################
    ###########################################################################

    # Define type of leg between bodies
    transfer_leg_settings, transfer_node_settings = transfer_trajectory.mga_settings_dsm_velocity_based_legs(
        transfer_body_order,
        departure_orbit=(np.inf, 0.0),
        arrival_orbit=(np.inf, 0.0))

    # Create transfer calculation object
    transfer_trajectory_object = transfer_trajectory.create_transfer_trajectory(
        bodies,
        transfer_leg_settings,
        transfer_node_settings,
        transfer_body_order,
        'Sun')

    ###########################################################################
    # DEFINE TRANSFER PARAMETERS ##############################################
    ###########################################################################

    # Define times at each node
    julian_day = constants.JULIAN_DAY
    node_times = list()
    node_times.append((1171.64503236 - 0.5) * julian_day)
    node_times.append(node_times[0] + 399.999999715 * julian_day)
    node_times.append(node_times[1] + 178.372255301 * julian_day)
    node_times.append(node_times[2] + 299.223139512 * julian_day)
    node_times.append(node_times[3] + 180.510754824 * julian_day)

    # Define free parameters per leg
    leg_free_parameters = list()
    leg_free_parameters.append(np.array([0.234594654679]))
    leg_free_parameters.append(np.array([0.0964769387134]))
    leg_free_parameters.append(np.array([0.829948744508]))
    leg_free_parameters.append(np.array([0.317174785637]))

    # Define free parameters per node
    node_free_parameters = list()
    node_free_parameters.append(np.array([1408.99421278, 0.37992647165 * 2.0 * 3.14159265358979, np.arccos(2.0 * 0.498004040298 - 1.0) - 3.14159265358979 / 2.0]))
    node_free_parameters.append(np.array([1.80629232251 * 6.378e6, 1.35077257078, 0.0]))
    node_free_parameters.append(np.array([3.04129845698 * 6.052e6, 1.09554368115, 0.0]))
    node_free_parameters.append(np.array([1.10000000891 * 6.052e6, 1.34317576594, 0.0]))
    node_free_parameters.append(np.array([]))

    ###########################################################################
    # EVALUATE TRANSFER #######################################################
    ###########################################################################

    # Evaluate transfer with given parameters
    transfer_trajectory_object.evaluate( node_times, leg_free_parameters, node_free_parameters)

    # Extract and print computed Delta V and time of flight
    print('Delta V [m/s]: ', transfer_trajectory_object.delta_v)
    print('Time of flight [day]: ', transfer_trajectory_object.time_of_flight / julian_day)
    print()
    print('Delta V per leg [m/s] : ', transfer_trajectory_object.delta_v_per_leg)
    print('Delta V per node [m/s] : ', transfer_trajectory_object.delta_v_per_node)

    transfer_trajectory.print_parameter_definitions(transfer_leg_settings, transfer_node_settings)

    # Extract state and thrust acceleration history
    state_history = transfer_trajectory_object.states_along_trajectory(500)
    thrust_acceleration_history = transfer_trajectory_object.inertial_thrust_accelerations_along_trajectory(500)

    # fly_by_states = np.array([state_history[node_times[i]] for i in range(len(node_times))])
    # state_history = result2array(state_history)
    # au = 1.5e11

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(state_history[:, 1] / au, state_history[:, 2] / au)
    # ax.scatter(fly_by_states[0, 0] / au, fly_by_states[0, 1] / au, color='blue', label='Earth departure')
    # ax.scatter(fly_by_states[1, 0] / au, fly_by_states[1, 1] / au, color='green', label='Earth fly-by')
    # ax.scatter(fly_by_states[2, 0] / au, fly_by_states[2, 1] / au, color='brown', label='Venus fly-by')
    # ax.scatter(fly_by_states[3, 0] / au, fly_by_states[3, 1] / au, color='brown')
    # ax.scatter(fly_by_states[4, 0] / au, fly_by_states[4, 1] / au, color='grey', label='Mercury arrival')
    # ax.scatter([0], [0], color='orange', label='Sun')
    # ax.set_xlabel('x wrt Sun [AU]')
    # ax.set_ylabel('y wrt Sun [AU]')
    # ax.set_aspect('equal')
    # ax.legend(bbox_to_anchor=[1, 1])
    # plt.show()

    return 0

def mga_with_spherical_shaping():

    print("######################### MGA with spherical shaping legs")

    ###########################################################################
    # DEFINE TRANSFER SETTINGS ################################################
    ###########################################################################

    # Simplified bodies
    bodies = environment_setup.create_simplified_system_of_bodies()
    central_body = 'Sun'

    # Define order of bodies (nodes) for gravity assists
    transfer_body_order = ['Earth', 'Mars', 'Earth']

    # Define departure and insertion orbit
    departure_semi_major_axis = np.inf
    departure_eccentricity = 0.0

    arrival_semi_major_axis = np.inf
    arrival_eccentricity = 0.0

    number_of_revolutions = [1,1]
    root_finder_settings = root_finders.bisection(1.0e-6, np.nan, np.nan, 30)
    lower_bound_free_coefficient = 1.0e-6
    upper_bound_free_coefficient = 1.0e-1

    creation_mode = 1

    ###########################################################################
    # CREATE TRANSFER SETTINGS AND OBJECT #####################################
    ###########################################################################
    # Create transfer legs and nodes settings

    print("Used creation mode: ", creation_mode)

    # Manually create transfer leg and node settins
    if creation_mode == 0:

        transfer_leg_settings = []
        transfer_leg_settings.append( transfer_trajectory.spherical_shaping_leg(
            root_finder_settings, lower_bound_free_coefficient, upper_bound_free_coefficient) )
        transfer_leg_settings.append( transfer_trajectory.spherical_shaping_leg(
            root_finder_settings, lower_bound_free_coefficient, upper_bound_free_coefficient) )

        transfer_node_settings = []
        transfer_node_settings.append( transfer_trajectory.departure_node(departure_semi_major_axis, departure_eccentricity) )
        transfer_node_settings.append( transfer_trajectory.swingby_node() )
        transfer_node_settings.append( transfer_trajectory.capture_node(arrival_semi_major_axis, arrival_eccentricity) )

    # Automatically create transfer leg and node settings for spherical shaping
    elif creation_mode == 1:
        transfer_leg_settings, transfer_node_settings = transfer_trajectory.mga_settings_spherical_shaping_legs(
            body_order=transfer_body_order,
            root_finder_settings=root_finder_settings,
            departure_orbit=(departure_semi_major_axis, departure_eccentricity),
            arrival_orbit=(arrival_semi_major_axis, arrival_eccentricity),
            lower_bound_free_coefficient=lower_bound_free_coefficient,
            upper_bound_free_coefficient=upper_bound_free_coefficient)


    # Create transfer calculation object
    transfer_trajectory_object = transfer_trajectory.create_transfer_trajectory(
        bodies,
        transfer_leg_settings,
        transfer_node_settings,
        transfer_body_order,
        central_body)

    ###########################################################################
    # DEFINE TRANSFER PARAMETERS ##############################################
    ###########################################################################

    transfer_trajectory.print_parameter_definitions(transfer_leg_settings, transfer_node_settings)

    # Define times at each node
    julian_day = constants.JULIAN_DAY
    node_times = list()
    node_times.append( 8174.5 * julian_day)
    node_times.append(node_times[0] + 580.0 * julian_day)
    node_times.append(node_times[1] + 560.0 * julian_day)

    # Define free parameters per leg
    leg_free_parameters = list()
    for i in range(len(transfer_body_order)-1):
        leg_free_parameters.append([number_of_revolutions[i]])

    # Define free parameters per node
    node_free_parameters = list()
    node_free_parameters.append(np.zeros(3))
    node_free_parameters.append([10, 0.0, 0.0, 65000.0e3, 1.3, 10.0])
    node_free_parameters.append(np.zeros(3))

    ###########################################################################
    # EVALUATE TRANSFER #######################################################
    ###########################################################################

    # Evaluate transfer with given parameters
    transfer_trajectory_object.evaluate(node_times, leg_free_parameters, node_free_parameters)

    # Extract and print computed Delta V and time of flight
    print('Delta V [m/s]: ', transfer_trajectory_object.delta_v)
    print('Time of flight [day]: ', transfer_trajectory_object.time_of_flight / julian_day)
    print()
    print('Delta V per leg [m/s] : ', transfer_trajectory_object.delta_v_per_leg)
    print('Delta V per node [m/s] : ', transfer_trajectory_object.delta_v_per_node)

    # Extract state and thrust acceleration history
    state_history = transfer_trajectory_object.states_along_trajectory(500)
    thrust_acceleration_history = transfer_trajectory_object.inertial_thrust_accelerations_along_trajectory(500)

    return 0

def mga_with_hodographic_shaping():

    print("\n"+"######################### MGA with hodographic shaping legs")

    ###########################################################################
    # DEFINE TRANSFER SETTINGS ################################################
    ###########################################################################

    # Simplified bodies
    bodies = environment_setup.create_simplified_system_of_bodies()
    central_body = 'Sun'

    # Define order of bodies (nodes) for gravity assists
    transfer_body_order = ['Earth', 'Mars', 'Earth']

    # Define departure and insertion orbit
    departure_semi_major_axis = np.inf
    departure_eccentricity = 0.0

    arrival_semi_major_axis = np.inf
    arrival_eccentricity = 0.0

    number_of_revolutions = [1,1]
    julian_day = constants.JULIAN_DAY
    time_of_flight = np.array([580.0, 560.0 ]) * julian_day
    departure_date = 8174.5 * julian_day

    creation_mode = 2

    ###########################################################################
    # CREATE TRANSFER SETTINGS AND OBJECT #####################################
    ###########################################################################
    # Create transfer legs and nodes settings

    print("Used creation mode: ", creation_mode)

    # Manually create transfer leg and node settings
    if creation_mode == 0:

        transfer_leg_settings = []
        for i in range(len(transfer_body_order)-1):
            radial_velocity_functions = shape_based_thrust.recommended_radial_hodograph_functions(time_of_flight[i])
            normal_velocity_functions = shape_based_thrust.recommended_normal_hodograph_functions(time_of_flight[i])
            axial_velocity_functions = shape_based_thrust.recommended_axial_hodograph_functions(time_of_flight[i], number_of_revolutions[i])

            transfer_leg_settings.append( transfer_trajectory.hodographic_shaping_leg(
                radial_velocity_functions, normal_velocity_functions, axial_velocity_functions) )

        transfer_node_settings = []
        transfer_node_settings.append( transfer_trajectory.departure_node(departure_semi_major_axis, departure_eccentricity) )
        transfer_node_settings.append( transfer_trajectory.swingby_node() )
        transfer_node_settings.append( transfer_trajectory.capture_node(arrival_semi_major_axis, arrival_eccentricity) )

    # Automatically create transfer leg and node settings for hodographic shaping
    elif creation_mode == 1:

        radial_velocity_function_components_per_leg = []
        normal_velocity_function_components_per_leg = []
        axial_velocity_function_components_per_leg = []
        for i in range(len(transfer_body_order)-1):
            radial_velocity_functions = shape_based_thrust.recommended_radial_hodograph_functions(time_of_flight[i])
            radial_velocity_function_components_per_leg.append(radial_velocity_functions)
            normal_velocity_functions = shape_based_thrust.recommended_normal_hodograph_functions(time_of_flight[i])
            normal_velocity_function_components_per_leg.append(normal_velocity_functions)
            axial_velocity_functions = shape_based_thrust.recommended_axial_hodograph_functions(time_of_flight[i], number_of_revolutions[i])
            axial_velocity_function_components_per_leg.append(axial_velocity_functions)

        transfer_leg_settings, transfer_node_settings = transfer_trajectory.mga_settings_hodographic_shaping_legs(
            body_order=transfer_body_order,
            radial_velocity_function_components_per_leg=radial_velocity_function_components_per_leg,
            normal_velocity_function_components_per_leg=normal_velocity_function_components_per_leg,
            axial_velocity_function_components_per_leg=axial_velocity_function_components_per_leg,
            departure_orbit=(departure_semi_major_axis, departure_eccentricity),
            arrival_orbit=(arrival_semi_major_axis, arrival_eccentricity) )

    # Automatically create transfer leg and node settings for hodographic shaping with recommended velocity functions
    elif creation_mode == 2:
        transfer_leg_settings, transfer_node_settings = transfer_trajectory.mga_settings_hodographic_shaping_legs_with_recommended_functions(
            body_order=transfer_body_order,
            time_of_flight_per_leg=time_of_flight,
            number_of_revolutions_per_leg=number_of_revolutions,
            departure_orbit=(departure_semi_major_axis, departure_eccentricity),
            arrival_orbit=(arrival_semi_major_axis, arrival_eccentricity) )


    # Create transfer calculation object
    transfer_trajectory_object = transfer_trajectory.create_transfer_trajectory(
        bodies,
        transfer_leg_settings,
        transfer_node_settings,
        transfer_body_order,
        central_body)

    ###########################################################################
    # DEFINE TRANSFER PARAMETERS ##############################################
    ###########################################################################

    transfer_trajectory.print_parameter_definitions(transfer_leg_settings, transfer_node_settings)

    # Define times at each node
    node_times = list()
    node_times.append( departure_date )
    node_times.append(node_times[0] + time_of_flight[0] )
    node_times.append(node_times[1] + time_of_flight[1] )

    # Define free parameters per leg
    leg_free_parameters = list()
    for i in range(len(transfer_body_order)-1):
        leg_free_parameters.append([number_of_revolutions[i]])

    # Define free parameters per node
    node_free_parameters = list()
    node_free_parameters.append(np.zeros(3))
    node_free_parameters.append([10, 0.0, 0.0, 65000.0e3, 1.3, 10.0])
    node_free_parameters.append(np.zeros(3))

    ###########################################################################
    # EVALUATE TRANSFER #######################################################
    ###########################################################################

    # Evaluate transfer with given parameters
    transfer_trajectory_object.evaluate(node_times, leg_free_parameters, node_free_parameters)

    # Extract and print computed Delta V and time of flight
    print('Delta V [m/s]: ', transfer_trajectory_object.delta_v)
    print('Time of flight [day]: ', transfer_trajectory_object.time_of_flight / julian_day)
    print()
    print('Delta V per leg [m/s] : ', transfer_trajectory_object.delta_v_per_leg)
    print('Delta V per node [m/s] : ', transfer_trajectory_object.delta_v_per_node)

    # Extract state and thrust acceleration history
    state_history = transfer_trajectory_object.states_along_trajectory(500)
    thrust_acceleration_history = transfer_trajectory_object.inertial_thrust_accelerations_along_trajectory(500)

    fly_by_states = np.array([state_history[node_times[i]] for i in range(len(node_times))])
    state_history = result2array(state_history)
    au = 1.5e11

########################
    # Plot the transfer
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111, projection='3d')
    # Plot the trajectory from the state history
    ax.plot(state_history[:, 1] / au, state_history[:, 2] / au, state_history[:, 3] / au)
    ## Plot the position of the nodes
    #ax.scatter(fly_by_states[0, 0] / au, fly_by_states[0, 1] / au, fly_by_states[0, 2] / au, color='blue', label='Earth departure')
    #ax.scatter(fly_by_states[1, 0] / au, fly_by_states[1, 1] / au, fly_by_states[1, 2] / au, color='brown', label='Venus fly-by')
    #ax.scatter(fly_by_states[2, 0] / au, fly_by_states[2, 1] / au, fly_by_states[2, 2] / au, color='brown', label='Venus fly-by')
    #ax.scatter(fly_by_states[3, 0] / au, fly_by_states[3, 1] / au, fly_by_states[3, 2] / au, color='green', label='Earth fly-by')
    #ax.scatter(fly_by_states[4, 0] / au, fly_by_states[4, 1] / au, fly_by_states[4, 2] / au, color='peru', label='Jupiter fly-by')
    #ax.scatter(fly_by_states[5, 0] / au, fly_by_states[5, 1] / au, fly_by_states[5, 2] / au, color='red', label='Saturn arrival')
    # Plot the position of the Sun
    #ax.scatter([0], [0], [0], color='orange', label='Sun')
    # Add axis labels and limits
    ax.set_xlabel('x wrt Sun [AU]')
    ax.set_ylabel('y wrt Sun [AU]')
    ax.set_zlabel('z wrt Sun [AU]')
    ax.set_xlim([-10.5, 2.5])
    ax.set_ylim([-8.5, 4.5])
    ax.set_zlim([-6.5, 6.5])
    # Put legend on the right
    ax.legend(bbox_to_anchor=[1.15, 1])
    plt.tight_layout()
    plt.show()
########################

    return 0

def mga_with_mixed_legs():

    print("\n"+"######################### MGA with mixed high-/low- thrust legs")

    ###########################################################################
    # DEFINE TRANSFER SETTINGS ################################################
    ###########################################################################

    # Simplified bodies
    bodies = environment_setup.create_simplified_system_of_bodies()
    central_body = 'Sun'

    # Define order of bodies (nodes) for gravity assists
    transfer_body_order = ['Earth', 'Mars', 'Earth', "Venus"]

    # Define departure and insertion orbit
    departure_semi_major_axis = np.inf
    departure_eccentricity = 0.0

    arrival_semi_major_axis = np.inf
    arrival_eccentricity = 0.0

    number_of_revolutions = 2
    julian_day = constants.JULIAN_DAY
    time_of_flight = np.array([260.0, 750.0, 300.0]) * julian_day
    departure_date = 6708 * julian_day

    ###########################################################################
    # CREATE TRANSFER SETTINGS AND OBJECT #####################################
    ###########################################################################
    # Create transfer legs and nodes settings

    # Manually create transfer leg settings and node settings
    transfer_leg_settings = []

    transfer_leg_settings.append( transfer_trajectory.unpowered_leg() )

    radial_velocity_functions = shape_based_thrust.recommended_radial_hodograph_functions(time_of_flight[1])
    normal_velocity_functions = shape_based_thrust.recommended_normal_hodograph_functions(time_of_flight[1])
    axial_velocity_functions = shape_based_thrust.recommended_axial_hodograph_functions(time_of_flight[1], number_of_revolutions)
    transfer_leg_settings.append( transfer_trajectory.hodographic_shaping_leg(
        radial_velocity_functions, normal_velocity_functions, axial_velocity_functions) )

    transfer_leg_settings.append( transfer_trajectory.unpowered_leg() )

    # Manually create transfer node settings

    transfer_node_settings = []
    transfer_node_settings.append( transfer_trajectory.departure_node(departure_semi_major_axis, departure_eccentricity) )
    transfer_node_settings.append( transfer_trajectory.swingby_node() )
    transfer_node_settings.append( transfer_trajectory.swingby_node() )
    transfer_node_settings.append( transfer_trajectory.capture_node(arrival_semi_major_axis, arrival_eccentricity) )


    # Create transfer calculation object
    transfer_trajectory_object = transfer_trajectory.create_transfer_trajectory(
        bodies,
        transfer_leg_settings,
        transfer_node_settings,
        transfer_body_order,
        central_body)

    ###########################################################################
    # DEFINE TRANSFER PARAMETERS ##############################################
    ###########################################################################

    transfer_trajectory.print_parameter_definitions(transfer_leg_settings, transfer_node_settings)

    # Define times at each node
    node_times = list()
    node_times.append( departure_date )
    node_times.append(node_times[0] + time_of_flight[0] )
    node_times.append(node_times[1] + time_of_flight[1] )
    node_times.append(node_times[2] + time_of_flight[2] )

    # Define free parameters per leg
    leg_free_parameters = list()
    leg_free_parameters.append([])
    leg_free_parameters.append([number_of_revolutions])
    leg_free_parameters.append([])

    # Define free parameters per node
    swingby_periapsis1 = 6.5e5 * 1e3
    swingby_node_deltaV1 = 10.0
    swingby_rotation_angle1 = 0.0
    swingby_periapsis2 = 42.0e6
    swingby_node_deltaV2 = 0.0
    swingby_rotation_angle2 = 0.0

    node_free_parameters = list()
    node_free_parameters.append(np.zeros(3))
    node_free_parameters.append([swingby_periapsis1, swingby_rotation_angle1, swingby_node_deltaV1])
    node_free_parameters.append([swingby_periapsis2, swingby_rotation_angle2, swingby_node_deltaV2])
    node_free_parameters.append(np.zeros(3))

    ###########################################################################
    # EVALUATE TRANSFER #######################################################
    ###########################################################################

    # Evaluate transfer with given parameters
    transfer_trajectory_object.evaluate(node_times, leg_free_parameters, node_free_parameters)

    # Extract and print computed Delta V and time of flight
    print('Delta V [m/s]: ', transfer_trajectory_object.delta_v)
    print('Time of flight [day]: ', transfer_trajectory_object.time_of_flight / julian_day)
    print()
    print('Delta V per leg [m/s] : ', transfer_trajectory_object.delta_v_per_leg)
    print('Delta V per node [m/s] : ', transfer_trajectory_object.delta_v_per_node)

    # Extract state and thrust acceleration history
    state_history = transfer_trajectory_object.states_along_trajectory(500)
    thrust_acceleration_history = transfer_trajectory_object.inertial_thrust_accelerations_along_trajectory(500)

    return 0

if __name__ == "__main__":
    mga_without_dsm()
    mga_with_dsm()
    mga_with_spherical_shaping()
    mga_with_hodographic_shaping()
    mga_with_mixed_legs()
