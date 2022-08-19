import numpy as np
import matplotlib.pyplot as plt
#from tudatpy.plotting import trajectory_3d
from trajectory3d import trajectory_3d
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.trajectory_design import transfer_trajectory
from tudatpy.kernel.trajectory_design import shape_based_thrust
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.util import result2array
from tudatpy.kernel import constants
from tudatpy.kernel.math import root_finders

#trajectory_parameters = [570727221.2273525 / constants.JULIAN_DAY,
#                         37073942.58665284 / constants.JULIAN_DAY,
#                         0,
#                         2471.19649906354,
#                         4207.587982407276,
#                         5594.040587888714,
#                         8748.139268525232,
#                         3449.838496679572]

trajectory_parameters = [570727221.2273525 / constants.JULIAN_DAY,
                         37073942.58665284 / constants.JULIAN_DAY,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0]
print("\n"+"######################### MGA with hodographic shaping legs")

###########################################################################
# DEFINE TRANSFER SETTINGS ################################################
###########################################################################

# Simplified bodies
#bodies = environment_setup.create_simplified_system_of_bodies()
#central_body = 'Sun'
spice_interface.load_standard_kernels()


# Define settings for celestial bodies
bodies_to_create = ['Earth',
                    'Mars',
                    'Sun']
# Define coordinate system
global_frame_origin = 'SSB'
global_frame_orientation = 'ECLIPJ2000'
# Create body settings
body_settings = environment_setup.get_default_body_settings(bodies_to_create,
                                                            global_frame_origin,
                                                            global_frame_orientation)
# Create bodies
bodies = environment_setup.create_system_of_bodies(body_settings)

# Define order of bodies (nodes) for gravity assists
transfer_body_order = ['Earth', 'Mars']

number_of_revolutions = [0]
julian_day = constants.JULIAN_DAY
#time_of_flight = [trajectory_parameters[0] * julian_day]
#departure_date = trajectory_parameters[1] * julian_day
time_of_flight = np.array([580.0]) * julian_day
departure_date = 8174.5 * julian_day
central_body_gravitational_parameter = 1.3284e20 # (sum of Sun and Jupiter)

# Define coordinate system
global_frame_origin = 'SSB'
global_frame_orientation = 'ECLIPJ2000'

# Define departure and insertion orbit
keplerian_state_Earth =  environment_setup.ephemeris.keplerian_from_spice(
                                            transfer_body_order[0],
                                            departure_date,
                                            central_body_gravitational_parameter,
                                            global_frame_origin,
                                            global_frame_orientation )

keplerian_state_Mars = environment_setup.ephemeris.keplerian_from_spice(
                                            transfer_body_order[1],
                                            departure_date,
                                            central_body_gravitational_parameter,
                                            global_frame_origin,
                                            global_frame_orientation )

#departure_semi_major_axis = keplerian_state_Earth[0]
#departure_eccentricity = keplerian_state_Earth[1]
#
#arrival_semi_major_axis = keplerian_state_Mars[0]
#arrival_eccentricity = keplerian_state_Mars[1]
departure_semi_major_axis = np.inf
departure_eccentricity = 0.0

arrival_semi_major_axis = np.inf
arrival_eccentricity = 0.0


creation_mode = 2 # 0 is used for testing

###########################################################################
# CREATE TRANSFER SETTINGS AND OBJECT #####################################
###########################################################################
# Create transfer legs and nodes settings

print("Used creation mode: ", creation_mode)

# Manually create transfer leg and node settings
if creation_mode == 0:
    transfer_leg_settings = []
    radial_velocity_functions = shape_based_thrust.recommended_radial_hodograph_functions(time_of_flight[0])
    normal_velocity_functions = shape_based_thrust.recommended_normal_hodograph_functions(time_of_flight[0])
    axial_velocity_functions = shape_based_thrust.recommended_axial_hodograph_functions(time_of_flight[0],
            number_of_revolutions[0])

    transfer_leg_settings = [transfer_trajectory.hodographic_shaping_leg(
        radial_velocity_functions, normal_velocity_functions,
        axial_velocity_functions)]

    transfer_node_settings = []
    transfer_node_settings.append( transfer_trajectory.departure_node(departure_semi_major_axis, departure_eccentricity) )
    transfer_node_settings.append( transfer_trajectory.capture_node(arrival_semi_major_axis, arrival_eccentricity) )

# Automatically create transfer leg and node settings for hodographic shaping with recommended velocity functions
elif creation_mode == 2:
    transfer_leg_settings, transfer_node_settings = transfer_trajectory.mga_settings_hodographic_shaping_legs_with_recommended_functions(
        body_order=transfer_body_order,
        time_of_flight_per_leg=time_of_flight,
        number_of_revolutions_per_leg=number_of_revolutions,
        departure_orbit=(departure_semi_major_axis, departure_eccentricity),
        arrival_orbit=(arrival_semi_major_axis, arrival_eccentricity) )

elif creation_mode == 3:

    def get_trajectory_time_of_flight(trajectory_parameters: list) -> float:
        return trajectory_parameters[1] * constants.JULIAN_DAY


    def get_trajectory_initial_time(trajectory_parameters: list,
                                    buffer_time: float = 0.0) -> float:
        return trajectory_parameters[0] * constants.JULIAN_DAY + buffer_time


    def get_trajectory_final_time(trajectory_parameters: list,
                                  buffer_time: float = 0.0) -> float:
        # Get initial time
        initial_time = get_trajectory_initial_time(trajectory_parameters)
        return initial_time + get_trajectory_time_of_flight(trajectory_parameters) - buffer_time

    ## Radial
    def get_radial_velocity_shaping_functions(trajectory_parameters: list,
                                              frequency: float,
                                              scale_factor: float,
                                              time_of_flight: float,
                                              number_of_revolutions: int) -> tuple:
        # Retrieve default methods (lowest-order in Gondelach and Noomen, 2015)
        radial_velocity_shaping_functions = shape_based_thrust.recommended_radial_hodograph_functions(time_of_flight)
        # Add degrees of freedom (highest-order in Gondelach and Noomen, 2015)
        radial_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_sine(
            exponent=1.0,
            frequency=0.5 * frequency,
            scale_factor=scale_factor))
        radial_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_cosine(
            exponent=1.0,
            frequency=0.5 * frequency,
            scale_factor=scale_factor))
        # Set free parameters
        free_coefficients = trajectory_parameters[3:5]
        return (radial_velocity_shaping_functions,
            free_coefficients)

    ##Normal
    def get_normal_velocity_shaping_functions(trajectory_parameters: list,
                                              frequency: float,
                                              scale_factor: float,
                                              time_of_flight: float,
                                              number_of_revolutions: int) -> tuple:
        # Retrieve default methods (lowest-order in Gondelach and Noomen, 2015)
        normal_velocity_shaping_functions = shape_based_thrust.recommended_normal_hodograph_functions(time_of_flight)
        # Add degrees of freedom (highest-order in Gondelach and Noomen, 2015)
        normal_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_sine(
            exponent=1.0,
            frequency=0.5 * frequency,
            scale_factor=scale_factor))
        normal_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_cosine(
            exponent=1.0,
            frequency=0.5 * frequency,
            scale_factor=scale_factor))
        # Set free parameters
        free_coefficients = trajectory_parameters[5:7]
        return (normal_velocity_shaping_functions,
                free_coefficients)


    ##Axial
    def get_normal_velocity_shaping_functions(trajectory_parameters: list,
                                              frequency: float,
                                              scale_factor: float,
                                              time_of_flight: float,
                                              number_of_revolutions: int) -> tuple:

        # Retrieve default methods (lowest-order in Gondelach and Noomen, 2015)
        normal_velocity_shaping_functions = shape_based_thrust.recommended_normal_hodograph_functions(time_of_flight)
        # Add degrees of freedom (highest-order in Gondelach and Noomen, 2015)
        normal_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_sine(
            exponent=1.0,
            frequency=0.5 * frequency,
            scale_factor=scale_factor))
        normal_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_cosine(
            exponent=1.0,
            frequency=0.5 * frequency,
            scale_factor=scale_factor))
        # Set free parameters
        free_coefficients = trajectory_parameters[5:7]
        return (normal_velocity_shaping_functions,
                free_coefficients)

    initial_time = get_trajectory_initial_time(trajectory_parameters)
    time_of_flight = get_trajectory_time_of_flight(trajectory_parameters)
    final_time = get_trajectory_final_time(trajectory_parameters)
    # Number of revolutions
    number_of_revolutions = int(trajectory_parameters[2])
    # Compute relevant frequency and scale factor for shaping functions
    frequency = 2.0 * np.pi / time_of_flight
    scale_factor = 1.0 / time_of_flight

    radial_velocity_shaping_functions, radial_free_coefficients = get_radial_velocity_shaping_functions(
        trajectory_parameters,
        frequency,
        scale_factor,
        time_of_flight,
        number_of_revolutions)
    normal_velocity_shaping_functions, normal_free_coefficients = get_normal_velocity_shaping_functions(
        trajectory_parameters,
        frequency,
        scale_factor,
        time_of_flight,
        number_of_revolutions)
    axial_velocity_shaping_functions, axial_free_coefficients = get_axial_velocity_shaping_functions(
        trajectory_parameters,
        frequency,
        scale_factor,
        time_of_flight,
        number_of_revolutions)
    # Retrieve boundary conditions and central body gravitational parameter
    initial_state = bodies.get_body('Earth').state_in_base_frame_from_ephemeris(initial_time)
    final_state = bodies.get_body('Mars').state_in_base_frame_from_ephemeris(final_time)
    gravitational_parameter = bodies.get_body('Sun').gravitational_parameter

    hodographic_shaping_object = shape_based_thrust.HodographicShapingLeg(initial_state,
                                                                   final_state,
                                                                   time_of_flight,
                                                                   gravitational_parameter,
                                                                   number_of_revolutions,
                                                                   radial_velocity_shaping_functions,
                                                                   normal_velocity_shaping_functions,
                                                                   axial_velocity_shaping_functions,
                                                                   radial_free_coefficients,
                                                                   normal_free_coefficients,
                                                                   axial_free_coefficients)

    # Create shaping object
    shaping_object = create_hodographic_shaping_object(trajectory_parameters,
                                                       bodies)
    # Define current hodograph time
    hodograph_time = epoch - get_trajectory_initial_time(trajectory_parameters)

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

transfer_trajectory.print_parameter_definitions(transfer_leg_settings, transfer_node_settings)

# Define times at each node
node_times = list()
node_times.append( departure_date )
for i in range(len(transfer_body_order)-1):
    node_times.append(node_times[i] + time_of_flight[i])

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
#state_history = result2array(state_history)
au = 1.5e11

########################
# Plot the transfer
########################
fig, ax = trajectory_3d(
    state_history,
    vehicles_names=["Spacecraft"],
    central_body_name="SSB",
    spice_bodies=["Earth", "Mars"],
    frame_orientation= 'ECLIPJ2000',
    linestyles=["dotted", "dashed", "solid"],
    colors=["blue", "green", "grey"]
)
# Change the size of the figure
#ax.scatter(fly_by_states[0, 0] , fly_by_states[0, 1] , fly_by_states[0,
#    2] , marker='o', color='blue', label='Earth departure')
#ax.scatter(fly_by_states[1, 0] , fly_by_states[1, 1] , fly_by_states[1,
#    2] , marker='o', color='brown', label='Mars fly-by')

fig.set_size_inches(8, 8)
# Show the plot
plt.show()

#
#fig = plt.figure(figsize=(8,5))
#ax = fig.add_subplot(111, projection='3d')
## Plot the trajectory from the state history
#ax.plot(state_history[:, 1] / au, state_history[:, 2] / au, state_history[:, 3] / au)
### Plot the position of the nodes
#ax.scatter(fly_by_states[0, 0] / au, fly_by_states[0, 1] / au, fly_by_states[0, 2] / au, color='blue', label='Earth departure')
#ax.scatter(fly_by_states[1, 0] / au, fly_by_states[1, 1] / au, fly_by_states[1, 2] / au, color='brown', label='Mars fly-by')
##ax.scatter(fly_by_states[2, 0] / au, fly_by_states[2, 1] / au, fly_by_states[2, 2] / au, color='brown', label='Earth arrival')
##ax.scatter(fly_by_states[3, 0] / au, fly_by_states[3, 1] / au, fly_by_states[3, 2] / au, color='green', label='Earth fly-by')
##ax.scatter(fly_by_states[4, 0] / au, fly_by_states[4, 1] / au, fly_by_states[4, 2] / au, color='peru', label='Jupiter fly-by')
##ax.scatter(fly_by_states[5, 0] / au, fly_by_states[5, 1] / au, fly_by_states[5, 2] / au, color='red', label='Saturn arrival')
## Plot the position of the Sun
#ax.scatter([0], [0], [0], color='orange', label='Sun')
## Add axis labels and limits
#ax.set_xlabel('x wrt Sun [AU]')
#ax.set_ylabel('y wrt Sun [AU]')
#ax.set_zlabel('z wrt Sun [AU]')
##ax.set_aspect('auto')
#ax.set_xlim([-1.5, 1.5])
#ax.set_ylim([-1.5, 1.5])
#ax.set_zlim([-1.5, 1.5])
## Put legend on the right
#ax.legend(bbox_to_anchor=[1.15, 1])
#plt.tight_layout()
#plt.show()
#

