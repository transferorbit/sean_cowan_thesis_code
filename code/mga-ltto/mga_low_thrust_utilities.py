'''
Author: Sean Cowan
Purpose: MSc Thesis
Date Created: 26-07-2022

This module provides functions that assist in setting up the mga low-thrust
trajectories. These functions will be called by optimization-runs.py when
performing the actual optimization.
'''
###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import numpy as np

# Tudatpy imports
import tudatpy
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.math import interpolators
from tudatpy.kernel.trajectory_design import shape_based_thrust
from tudatpy.kernel.trajectory_design import transfer_trajectory

from tudatpy.kernel.interface import spice

spice.load_standard_kernels()

###########################################################################
# HODOGRAPH-SPECIFIC FUNCTIONS ############################################
###########################################################################

class transfer_body_order_conversion:
    
    def __init__(self):
        self.body_dict = {0: "Null",
                1: "Mercury",
                2: "Venus",
                3: "Earth",
                4: "Mars",
                5: "Jupiter",
                6: "Saturn",
                7: "Uranus",
                8: "Neptune"}

    def get_transfer_body_integers(self, transfer_body_list: list, strip=True) -> np.ndarray:
        """
        Input : ["Venus", "Venus", "Earth", "Mercury", "Null", "Null", "Null", "Null"]
        Ouput : np.array([2, 2, 3, 5, 0, 0, 0, 0])
        """
        #transfer_body_list = transfer_body_list[1:-1]

        body_values = list(self.body_dict.values())
        body_keys = list(self.body_dict.keys())
        body_list = []

        for i, body in enumerate(transfer_body_list):
            for j in range(len(self.body_dict.values())):
                if body == body_values[j] and strip and body != "Null":
                    body_list.append(body_keys[j])
                elif body == body_values[j] and strip == False:
                    body_list.append(body_keys[j])

        body_array = np.array(body_list)

        return body_array

    def get_transfer_body_list(self, transfer_body_integers: np.ndarray, strip=True) -> list:
        """
        Input : np.array([2, 2, 3, 5, 0, 0, 0, 0])
        Output : ["Venus", "Venus", "Earth", "Mercury"]
        """

        body_values = list(self.body_dict.values())
        body_keys = list(self.body_dict.keys())
        body_list = []

        for i, j in enumerate(transfer_body_integers):
            for k in range(len(self.body_dict.values())):
                if j == body_keys[k] and strip and j != 0:
                    body_list.append(body_values[k])
                elif j == body_keys[k] and strip == False:
                    body_list.append(body_values[k])

        return body_list
    
    def get_mga_sequence(self, bodylist: list) -> str():

        char_list = []
        for i in bodylist:
            chars = [char for char in i]
            char_list.append(chars[0])
        mga_sequence = ""
        for j in char_list:
            mga_sequence += j

        return mga_sequence


def get_low_thrust_transfer_object(transfer_body_order : list,
                                        time_of_flight : np.ndarray,
                                        departure_elements : tuple,
                                        target_elements : tuple,
                                        bodies,
                                        central_body,
                                        no_of_free_parameters: int = 0,
                                        number_of_revolutions: np.ndarray = np.zeros(7, dtype=int),
                                        frequency: float = 1e-6,
                                        scale_factor: float = 1e-6) -> transfer_trajectory.TransferTrajectory:
    """
    Provides the transfer settings required to construct a hodographic shaping mga trajectory.
    frequency = 2.0 * np.pi / tof
    scale_factor = 1.0 / tof

    """


    # departure and target things
    departure_semi_major_axis = np.inf
    departure_eccentricity = 0
    target_semi_major_axis = np.inf
    target_eccentricity = 0
    
    transfer_leg_settings = []
    for i in range(len(transfer_body_order)-1):
        radial_velocity_functions = get_radial_velocity_shaping_functions(time_of_flight[i],
                                            number_of_revolutions=number_of_revolutions[i],
                                            no_of_free_parameters=no_of_free_parameters)
        normal_velocity_functions = get_normal_velocity_shaping_functions(time_of_flight[i],
                                            number_of_revolutions=number_of_revolutions[i],
                                            no_of_free_parameters=no_of_free_parameters)
        axial_velocity_functions = get_axial_velocity_shaping_functions(time_of_flight[i],
                                            number_of_revolutions=number_of_revolutions[i],
                                            no_of_free_parameters=no_of_free_parameters)

        transfer_leg_settings.append( transfer_trajectory.hodographic_shaping_leg( 
            radial_velocity_functions, normal_velocity_functions, axial_velocity_functions) )

    transfer_node_settings = []
    transfer_node_settings.append( transfer_trajectory.departure_node(departure_semi_major_axis, departure_eccentricity) )
    for i in range(len(transfer_body_order)-2):
        transfer_node_settings.append( transfer_trajectory.swingby_node() )
    transfer_node_settings.append( transfer_trajectory.capture_node(target_semi_major_axis, target_eccentricity) )

    # transfer_trajectory.print_parameter_definitions(transfer_leg_settings, transfer_node_settings)
    transfer_trajectory_object = transfer_trajectory.create_transfer_trajectory(
        bodies,
        transfer_leg_settings,
        transfer_node_settings,
        transfer_body_order,
        central_body)


    return transfer_trajectory_object


def get_node_times(transfer_body_order: list,
                    departure_date: float,
                    time_of_flight: np.ndarray) -> list:
    """
    Forms array of node times used for the 'evaluate' function of the transferTrajectory class.

    Returns
    --------
    list[float]

    """
    node_times=  []
    node_times.append(departure_date)
    for i in range(len(time_of_flight)):
        node_times.append(node_times[i] + time_of_flight[i])

    return node_times

def get_leg_free_parameters(free_parameters: np.ndarray,
                            transfer_body_order: list,
                            number_of_revolutions: np.array = np.zeros(1)) -> list:
    """
    Forms list of leg free parameters based on free_parameter vector

    Returns
    -------
    list[list[float]]

    """
    # free_parameters = np.ones(len(free_parameters))
    
    number_of_revolutions = np.zeros(len(transfer_body_order)-1)
    leg_free_parameters = list()
    for i in range(len(transfer_body_order)-1):
        leg_parameters = list()
        leg_parameters.append(number_of_revolutions[i])
        # for j in free_parameters:
        #     leg_parameters.append(j)
        leg_free_parameters.append(leg_parameters)

    return leg_free_parameters

def get_node_free_parameters(transfer_body_order: list, swingby_periapses: np.ndarray,
        incoming_velocities: np.ndarray, departure_velocity: float = 0, arrival_velocity: float = 0) -> list:
    """
    velocity magnitude
    velocity in-plane angle
    velocity out-of-plane angle
    
    velocity magnitude
    velocity in-plane angle
    velocity out-of-plane angle
    swingby periapsis
    swingby orbit-orientation rotation
    swingby delta V

    velocity magnitude
    velocity in-plane angle
    velocity out-of-plane angle
    """
    
    # swingby_delta_v = 0

    node_free_parameters = list()

    # Departure node
    node_free_parameters.append(np.array([departure_velocity, 0, 0]))

    # Swingby nodes
    for i in range(len(transfer_body_order)-2):
        node_parameters = list()
        node_parameters.append(incoming_velocities[i])
        node_parameters.append(0)
        node_parameters.append(0)
        node_parameters.append(swingby_periapses[i])
        node_parameters.append(0)
        node_parameters.append(0)

        node_free_parameters.append(node_parameters)

    # Arrival node
    node_free_parameters.append(np.array([arrival_velocity, 0, 0]))
    
    return node_free_parameters

def get_radial_velocity_shaping_functions(time_of_flight: float,
                                             no_of_free_parameters: int = 2,
                                             frequency: float = np.pi,
                                             scale_factor: float = 1,
                                             number_of_revolutions: int = 0) -> list:

    """
    Retrieves the radial velocity shaping functions (lowest and highest order in Gondelach and Noomen, 2015) and returns
    them together with the free coefficients.

    Parameters
    ----------
    time_of_flight: float
        Time of flight of the trajectory.
    no_of_free_parameters : list
        The number of free parameters to determine what base functions to include
    frequency: float
        Frequency of the highest-order methods.
    scale_factor: float
        Scale factor of the highest-order methods.
    number_of_revolutions: int
        Number of revolutions around the Sun (currently unused).

    Returns
    -------
    list
        Composed of the radial velocity shaping functions
    """

    # Retrieve default methods (lowest-order in Gondelach and Noomen, 2015)
    radial_velocity_shaping_functions = shape_based_thrust.recommended_radial_hodograph_functions(time_of_flight)
    # Add degrees of freedom (highest-order in Gondelach and Noomen, 2015)
    if no_of_free_parameters > 0:
        radial_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_sine(
            exponent=1.0,
            frequency=0.5 * frequency,
            scale_factor=scale_factor))
    if no_of_free_parameters > 1:
        radial_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_cosine(
            exponent=1.0,
            frequency=0.5 * frequency,
            scale_factor=scale_factor))
    if no_of_free_parameters > 2:
        radial_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_cosine(
            exponent=1.0,
            frequency=0.5 * frequency,
            scale_factor=scale_factor))

    return radial_velocity_shaping_functions

def get_normal_velocity_shaping_functions(time_of_flight: float,
                                             no_of_free_parameters: int = 2,
                                             frequency: float = np.pi,
                                             scale_factor: float = 1,
                                             number_of_revolutions: int = 0): # -> \
                                            # list(shape_based_thrust.BaseFunctionHodographicShaping):

    """
    Retrieves the normal velocity shaping functions (lowest and highest order in Gondelach and Noomen, 2015) and returns
    them together with the free coefficients.

    Parameters
    ----------
    time_of_flight: float
        Time of flight of the trajectory.
    no_of_free_parameters : list
        The number of free parameters to determine what base functions to include
    frequency: float
        Frequency of the highest-order methods.
    scale_factor: float
        Scale factor of the highest-order methods.
    number_of_revolutions: int
        Number of revolutions around the Sun (currently unused).

    Returns
    -------
    tuple
        A tuple composed by two lists: the normal velocity shaping functions and their free coefficients.
    """
    # Retrieve default methods (lowest-order in Gondelach and Noomen, 2015)
    normal_velocity_shaping_functions = shape_based_thrust.recommended_normal_hodograph_functions(time_of_flight)
    # Add degrees of freedom (highest-order in Gondelach and Noomen, 2015)
    if no_of_free_parameters > 0:
        normal_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_sine(
            exponent=1.0,
            frequency=0.5 * frequency,
            scale_factor=scale_factor))
    if no_of_free_parameters > 1:
        normal_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_cosine(
            exponent=1.0,
            frequency=0.5 * frequency,
            scale_factor=scale_factor))
    if no_of_free_parameters > 2:
        normal_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_cosine(
            exponent=1.0,
            frequency=0.5 * frequency,
            scale_factor=scale_factor))

    return normal_velocity_shaping_functions

def get_axial_velocity_shaping_functions(time_of_flight: float,
                                             no_of_free_parameters: int = 2,
                                             frequency: float = np.pi,
                                             scale_factor: float = 1,
                                             number_of_revolutions: int = 0): # -> \
                                            # list(shape_based_thrust.BaseFunctionHodographicShaping):

    """
    Retrieves the axial velocity shaping functions (lowest and highest order in Gondelach and Noomen, 2015) and returns
    them together with the free coefficients.

    Parameters
    ----------
    time_of_flight: float
        Time of flight of the trajectory.
    no_of_free_parameters : list
        The number of free parameters to determine what base functions to include
    frequency: float
        Frequency of the highest-order methods.
    scale_factor: float
        Scale factor of the highest-order methods.
    number_of_revolutions: int
        Number of revolutions around the Sun (currently unused).

    Returns
    -------
    tuple
        A tuple composed by two lists: the axial velocity shaping functions and their free coefficients.
    """

    # Add degrees of freedom (lowest-order in Gondelach and Noomen, 2015)
    axial_velocity_shaping_functions = shape_based_thrust.recommended_axial_hodograph_functions(
        time_of_flight,
        number_of_revolutions)

    exponent = 2.0
    # Add degrees of freedom (highest-order in Gondelach and Noomen, 2015)
    if no_of_free_parameters > 0:
        axial_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_cosine(
            exponent=exponent,
            frequency=(number_of_revolutions + 0.5) * frequency,
            scale_factor=scale_factor ** exponent))
    if no_of_free_parameters > 1:
        axial_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_sine(
            exponent=exponent,
            frequency=(number_of_revolutions + 0.5) * frequency,
            scale_factor=scale_factor ** exponent))
    if no_of_free_parameters > 2:
        axial_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_cosine(
            exponent=4.0, #randomly chosen
            frequency=0.5 * frequency,
            scale_factor=scale_factor))

    return axial_velocity_shaping_functions

