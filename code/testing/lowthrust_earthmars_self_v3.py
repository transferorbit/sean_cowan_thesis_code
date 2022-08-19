import faulthandler
faulthandler.enable()

import numpy as np
import matplotlib.pyplot as plt
from trajectory3d import trajectory_3d
from tudatpy.kernel.trajectory_design import transfer_trajectory
from tudatpy.kernel.trajectory_design import shape_based_thrust
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.util import result2array
from tudatpy.kernel import constants
from tudatpy.kernel.math import root_finders

def mga_with_hodographic_shaping():

    print("\n"+"######################### MGA with hodographic shaping legs")

    ###########################################################################
    # DEFINE TRANSFER SETTINGS ################################################
    ###########################################################################

    # Simplified bodies
    bodies = environment_setup.create_simplified_system_of_bodies()
    central_body = 'Sun'

    # Define order of bodies (nodes) for gravity assists
    #transfer_body_order = ['Earth', 'Mars']
    #transfer_body_order = ['Earth', 'Earth', 'Mars']
    transfer_body_order = ['Earth', 'Venus', 'Venus', 'Earth', 'Jupiter',
            'Saturn']

    # Define departure and insertion orbit
    departure_semi_major_axis = np.inf
    departure_eccentricity = 0.0

    #arrival_semi_major_axis = np.inf
    #arrival_eccentricity = 0.0
    arrival_semi_major_axis = 1.0895e8 / 0.02
    arrival_eccentricity = 0.98

    number_of_revolutions = [0, 0, 0, 0, 0]
    julian_day = constants.JULIAN_DAY
    time_of_flight = np.array([158.302027105278, 449.385873819743, 54.7489684339665, 1024.36205846918, 4552.30796805542]) * julian_day
    departure_date = (-789.8117 - 0.5)  * julian_day
    tof = time_of_flight[0]
    tof = time_of_flight[1] - time_of_flight[0]
    #node_times.append( ( -789.8117 - 0.5 ) * julian_day )
    #node_times.append( node_times[ 0 ] + 158.302027105278 * julian_day )
    #node_times.append( node_times[ 1 ] + 449.385873819743 * julian_day )
    #node_times.append( node_times[ 2 ] + 54.7489684339665 * julian_day )
    #node_times.append( node_times[ 3 ] + 1024.36205846918 * julian_day )
    #node_times.append( node_times[ 4 ] + 4552.30796805542 * julian_day )

    creation_mode = 0

    ###########################################################################
    # CREATE TRANSFER SETTINGS AND OBJECT #####################################
    ###########################################################################
    # Create transfer legs and nodes settings

    print("Used creation mode: ", creation_mode)
    frequency = 2.0 * np.pi / tof
    scale_factor = 1.0 / tof

    # Manually create transfer leg and node settings
    if creation_mode == 0:
        transfer_leg_settings = []
        for i in range(len(transfer_body_order)-1):

            radial_velocity_functions = shape_based_thrust.recommended_radial_hodograph_functions(time_of_flight[i])
            radial_velocity_functions.append(shape_based_thrust.hodograph_power_sine(
                exponent=1.0,
                frequency=0.5 * frequency,
                scale_factor=scale_factor))
            radial_velocity_functions.append(shape_based_thrust.hodograph_scaled_power_cosine(
                exponent=1.0,
                frequency=0.5 * frequency,
                scale_factor=scale_factor))

            normal_velocity_functions = shape_based_thrust.recommended_normal_hodograph_functions(time_of_flight[i])
            normal_velocity_functions.append(shape_based_thrust.hodograph_power_sine(
                exponent=1.0,
                frequency=0.5 * frequency,
                scale_factor=scale_factor))
            normal_velocity_functions.append(shape_based_thrust.hodograph_scaled_power_cosine(
                exponent=1.0,
                frequency=0.5 * frequency,
                scale_factor=scale_factor))

            axial_velocity_functions = shape_based_thrust.recommended_axial_hodograph_functions(time_of_flight[i], number_of_revolutions[i])
            axial_velocity_functions.append(shape_based_thrust.hodograph_power_sine(
                exponent=1.0,
                frequency=0.5 * frequency,
                scale_factor=scale_factor))
            axial_velocity_functions.append(shape_based_thrust.hodograph_scaled_power_cosine(
                exponent=1.0,
                frequency=0.5 * frequency,
                scale_factor=scale_factor))

            transfer_leg_settings.append( transfer_trajectory.hodographic_shaping_leg(
                radial_velocity_functions, normal_velocity_functions, axial_velocity_functions) )

        transfer_node_settings = []
        transfer_node_settings.append( transfer_trajectory.departure_node(departure_semi_major_axis, departure_eccentricity) )
        transfer_node_settings.append( transfer_trajectory.swingby_node() )
        transfer_node_settings.append( transfer_trajectory.swingby_node() )
        transfer_node_settings.append( transfer_trajectory.swingby_node() )
        transfer_node_settings.append( transfer_trajectory.swingby_node() )
        transfer_node_settings.append( transfer_trajectory.capture_node(arrival_semi_major_axis, arrival_eccentricity) )

    # Automatically create transfer leg and node settings for hodographic shaping
    elif creation_mode == 1: #USe this for extra

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
    node_times.append(node_times[2] + time_of_flight[2] )
    node_times.append(node_times[3] + time_of_flight[3] )
    node_times.append(node_times[4] + time_of_flight[4] )

    # Define free parameters per leg
    leg_free_parameters = list()
    for i in range(len(transfer_body_order)-1):
        leg_free_parameters.append([number_of_revolutions[i], 1, 1])

    # Define free parameters per node
    node_free_parameters = list()
    node_free_parameters.append(np.zeros(3))
    node_free_parameters.append([10, 0.0, 0.0, 65000.0e3, 1.3, 0.0])
    node_free_parameters.append([10, 0.0, 0.0, 65000.0e3, 1.3, 0.0])
    node_free_parameters.append([10, 0.0, 0.0, 65000.0e3, 1.3, 0.0])
    node_free_parameters.append([10, 0.0, 0.0, 65000.0e3, 1.3, 0.0])
    node_free_parameters.append(np.zeros(3))
    #node_free_parameters.append(np.zeros(0))

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
    print(type(state_history))
    thrust_acceleration_history = transfer_trajectory_object.inertial_thrust_accelerations_along_trajectory(500)

    fly_by_states = np.array([state_history[node_times[i]] for i in range(len(node_times))])
    # state_history = result2array(state_history)
    au = 1.5e11

########################
    # Plot the transfer
########################
    fig, ax = trajectory_3d(
        state_history,
        vehicles_names=["Spacecraft"],
        central_body_name="SSB",
        spice_bodies=["Earth", "Venus", "Jupiter", "Saturn"],
        frame_orientation= 'ECLIPJ2000',
    )
    # Change the size of the figure
    ax.scatter(fly_by_states[0, 0] , fly_by_states[0, 1] , fly_by_states[0,
        2] , marker='o', color='blue', label='Earth departure')
    ax.scatter(fly_by_states[1, 0] , fly_by_states[1, 1] , fly_by_states[1,
        2] , marker='o', color='brown', label='Mars fly-by')

    fig.set_size_inches(8, 8)
    # Show the plot
    plt.show()
#    fig = plt.figure(figsize=(8,5))
#    ax = fig.add_subplot(111, projection='3d')
#    # Plot the trajectory from the state history
#    ax.plot(state_history[:, 1] / au, state_history[:, 2] / au, state_history[:, 3] / au)
#    ## Plot the position of the nodes
#    #ax.scatter(fly_by_states[0, 0] / au, fly_by_states[0, 1] / au, fly_by_states[0, 2] / au, color='blue', label='Earth departure')
#    #ax.scatter(fly_by_states[1, 0] / au, fly_by_states[1, 1] / au, fly_by_states[1, 2] / au, color='brown', label='Venus fly-by')
#    #ax.scatter(fly_by_states[2, 0] / au, fly_by_states[2, 1] / au, fly_by_states[2, 2] / au, color='brown', label='Venus fly-by')
#    #ax.scatter(fly_by_states[3, 0] / au, fly_by_states[3, 1] / au, fly_by_states[3, 2] / au, color='green', label='Earth fly-by')
#    #ax.scatter(fly_by_states[4, 0] / au, fly_by_states[4, 1] / au, fly_by_states[4, 2] / au, color='peru', label='Jupiter fly-by')
#    #ax.scatter(fly_by_states[5, 0] / au, fly_by_states[5, 1] / au, fly_by_states[5, 2] / au, color='red', label='Saturn arrival')
#    # Plot the position of the Sun
#    #ax.scatter([0], [0], [0], color='orange', label='Sun')
#    # Add axis labels and limits
#    ax.set_xlabel('x wrt Sun [AU]')
#    ax.set_ylabel('y wrt Sun [AU]')
#    ax.set_zlabel('z wrt Sun [AU]')
#    ax.set_xlim([-10.5, 2.5])
#    ax.set_ylim([-8.5, 4.5])
#    ax.set_zlim([-6.5, 6.5])
#    # Put legend on the right
#    ax.legend(bbox_to_anchor=[1.15, 1])
#    plt.tight_layout()
#    plt.show()
########################

    return 0

