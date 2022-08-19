###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import numpy as np
import os

# Tudatpy imports
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.math import interpolators
from tudatpy.kernel.trajectory_design import shape_based_thrust
from tudatpy.kernel.trajectory_design import transfer_trajectory

###########################################################################
# DEFINE GLOBAL SETTINGS ##################################################
###########################################################################

# Load spice kernels
spice_interface.load_standard_kernels()
trajectory_parameters = [570727221.2273525 / constants.JULIAN_DAY,
                         37073942.58665284 / constants.JULIAN_DAY,
                         0,
                         2471.19649906354,
                         4207.587982407276,
                         -5594.040587888714,
                         8748.139268525232,
                         -3449.838496679572]

write_results_to_file = True
# Get path of current directory
current_dir = os.path.dirname(__file__)

###########################################################################
# DEFINE SIMULATION SETTINGS ##############################################
###########################################################################

# Fixed parameters
minimum_mars_distance = 5.0E7
# Time since 'departure from Earth CoM' at which propagation starts (and similar
# for arrival time)
time_buffer = 30.0 * constants.JULIAN_DAY

###################
###################

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

###################
###################

initial_propagation_time = get_trajectory_initial_time(trajectory_parameters,
        time_buffer)

###########################################################################
# CREATE ENVIRONMENT ######################################################
###########################################################################

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

###########################################################################
# WRITE RESULTS FOR SEMI-ANALYTICAL METHOD ################################
###########################################################################

# Time settings
initial_time = get_trajectory_initial_time(trajectory_parameters)
time_of_flight = get_trajectory_time_of_flight(trajectory_parameters)
final_time = get_trajectory_final_time(trajectory_parameters)
# Number of revolutions
number_of_revolutions = int(trajectory_parameters[2])
# Compute relevant frequency and scale factor for shaping functions
frequency = 2.0 * np.pi / time_of_flight
scale_factor = 1.0 / time_of_flight

def get_radial_velocity_shaping_functions(trajectory_parameters: list,
                                          frequency: float,
                                          scale_factor: float,
                                          time_of_flight: float,
                                          number_of_revolutions: int) -> tuple:
    # Retrieve default methods (lowest-order in Gondelach and Noomen, 2015)
    radial_velocity_shaping_functions = shape_based_thrust.recommended_radial_hodograph_functions(time_of_flight)
    # Add degrees of freedom (highest-order in Gondelach and Noomen, 2015)
    #radial_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_sine(
    #    exponent=1.0,
    #    frequency=0.5 * frequency,
    #    scale_factor=scale_factor))
    #radial_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_cosine(
    #    exponent=1.0,
    #    frequency=0.5 * frequency,
    #    scale_factor=scale_factor))
    ## Set free parameters
    #free_coefficients = trajectory_parameters[3:5]
    #return (radial_velocity_shaping_functions,
    #        free_coefficients)
    return radial_velocity_shaping_functions


def get_normal_velocity_shaping_functions(trajectory_parameters: list,
                                          frequency: float,
                                          scale_factor: float,
                                          time_of_flight: float,
                                          number_of_revolutions: int) -> tuple:
    # Retrieve default methods (lowest-order in Gondelach and Noomen, 2015)
    normal_velocity_shaping_functions = shape_based_thrust.recommended_normal_hodograph_functions(time_of_flight)
    # Add degrees of freedom (highest-order in Gondelach and Noomen, 2015)
    #normal_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_sine(
    #    exponent=1.0,
    #    frequency=0.5 * frequency,
    #    scale_factor=scale_factor))
    #normal_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_cosine(
    #    exponent=1.0,
    #    frequency=0.5 * frequency,
    #    scale_factor=scale_factor))
    ## Set free parameters
    #free_coefficients = trajectory_parameters[5:7]
    #return (normal_velocity_shaping_functions,
    #        free_coefficients)
    return normal_velocity_shaping_functions


def get_axial_velocity_shaping_functions(trajectory_parameters: list,
                                         frequency: float,
                                         scale_factor: float,
                                         time_of_flight: float,
                                         number_of_revolutions: int) -> tuple:
    # Retrieve default methods (lowest-order in Gondelach and Noomen, 2015)
    axial_velocity_shaping_functions = shape_based_thrust.recommended_axial_hodograph_functions(
        time_of_flight,
        number_of_revolutions)
    # Add degrees of freedom (highest-order in Gondelach and Noomen, 2015)
    #exponent = 4.0
    #axial_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_cosine(
    #    exponent=exponent,
    #    frequency=(number_of_revolutions + 0.5) * frequency,
    #    scale_factor=scale_factor ** exponent))
    #axial_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_sine(
    #    exponent=exponent,
    #    frequency=(number_of_revolutions + 0.5) * frequency,
    #    scale_factor=scale_factor ** exponent))
    ## Set free parameters
    #free_coefficients = trajectory_parameters[7:9]
    #return (axial_velocity_shaping_functions,
    #        free_coefficients)
    return axial_velocity_shaping_functions

# Retrieve shaping functions and free parameters
radial_velocity_shaping_functions = get_radial_velocity_shaping_functions(
    trajectory_parameters,
    frequency,
    scale_factor,
    time_of_flight,
    number_of_revolutions)
normal_velocity_shaping_functions = get_normal_velocity_shaping_functions(
    trajectory_parameters,
    frequency,
    scale_factor,
    time_of_flight,
    number_of_revolutions)
axial_velocity_shaping_functions = get_axial_velocity_shaping_functions(
    trajectory_parameters,
    frequency,
    scale_factor,
    time_of_flight,
    number_of_revolutions)

# Retrieve boundary conditions and central body gravitational parameter
initial_state = bodies.get_body('Earth').state_in_base_frame_from_ephemeris(initial_time)
final_state = bodies.get_body('Mars').state_in_base_frame_from_ephemeris(final_time)
gravitational_parameter = bodies.get_body('Sun').gravitational_parameter

# Create and return shape-based method
hodographic_shaping_object = transfer_trajectory.HodographicShapingLeg(initial_state,
                                                                   final_state,
                                                                   time_of_flight,
                                                                   gravitational_parameter,
                                                                   number_of_revolutions,
                                                                   radial_velocity_shaping_functions,
                                                                   normal_velocity_shaping_functions,
                                                                   axial_velocity_shaping_functions)#,
#                                                                   radial_free_coefficients,
#                                                                   normal_free_coefficients,
#                                                                   axial_free_coefficients)

# Prepares output path
if write_results_to_file:
    output_path = current_dir + '/SimulationOutput/HodographicSemiAnalytical/'
else:
    output_path = None

# get_hodographic_shape function
start_time = 0.0
final_time = get_trajectory_time_of_flight(trajectory_parameters)
# Set number of data points
number_of_data_points = 10000
# Compute step size
step_size = (final_time - start_time) / (number_of_data_points - 1)
# Create epochs vector
epochs = np.linspace(start_time,
                     final_time,
                     number_of_data_points)

trajectory_shape = hodographic_shaping_object.get_trajectory(epochs)

save2txt(trajectory_shape, 'hodographic_trajectory.dat', output_path)

###########################################################################
# RUN SIMULATION FOR VARIOUS SETTINGS #####################################
###########################################################################

def get_hodograph_thrust_acceleration_settings(trajectory_parameters: list,
                                               bodies: tudatpy.kernel.numerical_simulation.environment.SystemOfBodies,
                                               specific_impulse: float) \
        -> tudatpy.kernel.trajectory_design.shape_based_thrust.HodographicShaping:

    # Create shaping object
    shaping_object = create_hodographic_shaping_object(trajectory_parameters,
                                                       bodies)
    # Compute offset, which is the time since J2000 (when t=0 for tudat) at which the simulation starts
    # N.B.: this is different from time_buffer, which is the delay between the start of the hodographic
    # trajectory and the beginning of the simulation
    time_offset = get_trajectory_initial_time(trajectory_parameters)
    # Create specific impulse lambda function
    specific_impulse_function = lambda t: specific_impulse
    # Return acceleration settings
    return transfer_trajectory.get_low_thrust_acceleration_settings(shaping_object,
                                                                    bodies,
                                                                    'Vehicle',
                                                                    specific_impulse_function,
                                                                    time_offset)

hodograph_time = epoch - get_trajectory_initial_time(trajectory_parameters)
hodographic_shaping_states = hodographic_shaping_object.get_state(hodograph_time)

