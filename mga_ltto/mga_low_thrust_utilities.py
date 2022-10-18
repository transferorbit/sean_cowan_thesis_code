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
import matplotlib.pyplot as plt

# Tudatpy imports
import tudatpy
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.trajectory_design import shape_based_thrust
from tudatpy.kernel.trajectory_design import transfer_trajectory
from tudatpy.kernel.interface import spice
spice.load_standard_kernels()

from trajectory3d import trajectory_3d

import io
import sys

# import warnings
# warnings.filterwarnings("error")



###########################################################################
# HODOGRAPH-SPECIFIC FUNCTIONS ############################################
###########################################################################

class transfer_body_order_conversion:
    
    def __init__(self):
        pass

    @staticmethod
    def get_transfer_body_integers(transfer_body_list: list, strip=True) -> np.ndarray:
        """
        Input : ["Venus", "Venus", "Earth", "Mercury", "Null", "Null", "Null", "Null"]
        Ouput : np.array([2, 2, 3, 5, 0, 0, 0, 0])
        """
        #transfer_body_list = transfer_body_list[1:-1]
        body_dict = {0: "Null",
                1: "Mercury",
                2: "Venus",
                3: "Earth",
                4: "Mars",
                5: "Jupiter",
                6: "Saturn",
                7: "Uranus",
                8: "Neptune"}

        body_values = list(body_dict.values())
        body_keys = list(body_dict.keys())
        body_list = []

        for body in transfer_body_list:
            for j in range(len(body_dict.values())):
                if body == body_values[j] and strip and body != "Null":
                    body_list.append(body_keys[j])
                elif body == body_values[j] and strip == False:
                    body_list.append(body_keys[j])

        body_array = np.array(body_list)

        return body_array

    @staticmethod
    def get_transfer_body_list(transfer_body_integers: np.ndarray, strip=True) -> list:
        """
        Input : np.array([2, 2, 3, 5, 0, 0, 0, 0])
        Output : ["Venus", "Venus", "Earth", "Jupiter"]
        """
        body_dict = {0: "Null",
                1: "Mercury",
                2: "Venus",
                3: "Earth",
                4: "Mars",
                5: "Jupiter",
                6: "Saturn",
                7: "Uranus",
                8: "Neptune"}

        body_values = list(body_dict.values())
        body_keys = list(body_dict.keys())
        body_list = []

        for j in transfer_body_integers:
            for k in range(len(body_dict.values())):
                if j == body_keys[k] and strip and j != 0:
                    body_list.append(body_values[k])
                elif j == body_keys[k] and strip == False:
                    body_list.append(body_values[k])

        return body_list
    
    @staticmethod
    def get_mga_characters_from_list(bodylist: list) -> str():

        character_dict = {'Y' : "Mercury",
                'V' : "Venus",
                'E' : "Earth",
                'M' : "Mars",
                'J' : "Jupiter",
                'S' : "Saturn",
                'U' : "Uranus",
                'N' : "Neptune"}

        mga_sequence = ""
        for i in bodylist:
            for j, k in character_dict.items():
                if i == k:
                    mga_sequence += j

        return mga_sequence

    
    @staticmethod
    def get_mga_list_from_characters(character_string: str) -> str():

        character_dict = {'Y' : "Mercury",
                'V' : "Venus",
                'E' : "Earth",
                'M' : "Mars",
                'J' : "Jupiter",
                'S' : "Saturn",
                'U' : "Uranus",
                'N' : "Neptune"}

        character_list = [i for i in character_string]

        return [character_dict[i] for i in character_list]

    @staticmethod
    def get_list_of_legs_from_characters(mga_sequence_characters):
        """
        This functions takes the string of characters and returns a list of the separate legs

        Parameters
        -----------
        mga_string : str

        Returns
        --------
        List[str]

        """

        chars = [i for i in mga_sequence_characters]
        number_of_legs = len(chars) - 1

        list_of_legs = []
        for i in range(number_of_legs):
            list_of_legs.append(chars[i] + chars[i+1])

        return list_of_legs

    def get_dict_of_legs_from_characters(mga_sequence_characters):
        chars = [i for i in mga_sequence_characters]
        number_of_legs = len(chars) - 1

        dict_of_legs = {}
        for i in range(number_of_legs):
            dict_of_legs[chars[i] + chars[i+1]] = i

        return dict_of_legs



def get_low_thrust_transfer_object(transfer_body_order : list,
                                        time_of_flight : np.ndarray,
                                        departure_elements : tuple,
                                        target_elements : tuple,
                                        bodies,
                                        central_body,
                                        no_of_free_parameters: int = 0,
                                        manual_base_functions=False,
                                        number_of_revolutions: np.ndarray = np.zeros(7, dtype=int))\
                                        -> transfer_trajectory.TransferTrajectory:
    """
    Provides the transfer settings required to construct a hodographic shaping mga trajectory.

    """

    # departure and target state from ephemerides
    body_dict = {"Mercury" : 0,
            "Venus" : 1,
            "Earth" : 2,
            "Mars" : 3,
            "Jupiter" : 4,
            "Saturn" : 5,
            "Uranus" : 6,
            "Neptune" : 7}

    # From approximate jpl model states
    planet_kep_states = [[0.38709927,      0.20563593 ,     7.00497902 ,     252.25032350,
            77.45779628,     48.33076593],
    [0.72333566  ,    0.00677672  ,    3.39467605   ,   181.97909950  ,  131.60246718   ,  76.67984255],
    [1.00000261  ,    0.01671123  ,   -0.00001531   ,   100.46457166  ,  102.93768193   ,   0.0],
    [1.52371034  ,    0.09339410  ,    1.84969142   ,    -4.55343205  ,  -23.94362959   ,  49.55953891],
    [5.20288700  ,    0.04838624  ,    1.30439695   ,    34.39644051  ,   14.72847983   , 100.47390909],
    [9.53667594  ,    0.05386179  ,    2.48599187   ,    49.95424423  ,   92.59887831   , 113.66242448],
    [19.18916464 ,     0.04725744 ,     0.77263783  ,    313.23810451 ,   170.95427630  ,   74.01692503],
    [30.06992276 ,     0.00859048 ,     1.77004347  ,    -55.12002969 ,    44.96476227  ,
        131.78422574]]

    astronomical_unit = 149597870.7e3 #m
    # departure_semi_major_axis = planet_kep_states[body_dict[transfer_body_order[0]]][0] * astronomical_unit
    # departure_eccentricity = planet_kep_states[body_dict[transfer_body_order[0]]][1]
    # target_semi_major_axis = planet_kep_states[body_dict[transfer_body_order[-1]]][0] * astronomical_unit
    # target_eccentricity = planet_kep_states[body_dict[transfer_body_order[-1]]][1]
    
    # departure and target state from ephemerides
    departure_semi_major_axis = np.inf
    departure_eccentricity = 0
    target_semi_major_axis = np.inf
    target_eccentricity = 0
    
    transfer_leg_settings = []
    for i in range(len(transfer_body_order)-1):

        # first definition of freq, scale
        tof = time_of_flight[i]
        frequency = 2.0 * np.pi / tof
        scale_factor = 1.0 / tof
        # scale_factor = 1.0

        radial_velocity_functions = get_radial_velocity_shaping_functions(time_of_flight[i],
                                            number_of_revolutions=number_of_revolutions[i],
                                            no_of_free_parameters=no_of_free_parameters,
                                            frequency=frequency,
                                            scale_factor=scale_factor,
                                            manual_base_functions=manual_base_functions)
        normal_velocity_functions = get_normal_velocity_shaping_functions(time_of_flight[i],
                                            number_of_revolutions=number_of_revolutions[i],
                                            no_of_free_parameters=no_of_free_parameters,
                                            frequency=frequency,
                                            scale_factor=scale_factor,
                                            manual_base_functions=manual_base_functions)
        axial_velocity_functions = get_axial_velocity_shaping_functions(time_of_flight[i],
                                            number_of_revolutions=number_of_revolutions[i],
                                            no_of_free_parameters=no_of_free_parameters,
                                            frequency=frequency,
                                            scale_factor=scale_factor,
                                            manual_base_functions=manual_base_functions)

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

def get_low_thrust_transfer_object_automated(transfer_body_order : list,
                                        time_of_flight : np.ndarray,
                                        departure_elements : tuple,
                                        target_elements : tuple,
                                        bodies,
                                        central_body,
                                        no_of_free_parameters: int = 0,
                                        manual_base_functions=False,
                                        number_of_revolutions = None)\
                                        -> transfer_trajectory.TransferTrajectory:

    departure_semi_major_axis = np.inf
    departure_eccentricity = 0
    target_semi_major_axis = np.inf
    target_eccentricity = 0
    time_of_flight_list = time_of_flight.tolist()

    transfer_leg_settings, transfer_node_settings = \
    transfer_trajectory.mga_settings_hodographic_shaping_legs_with_recommended_functions(transfer_body_order,
            time_of_flight_list, number_of_revolutions, (departure_semi_major_axis,
                departure_eccentricity), (target_semi_major_axis, target_eccentricity))

    transfer_trajectory_object = transfer_trajectory.create_transfer_trajectory(
        bodies,
        transfer_leg_settings,
        transfer_node_settings,
        transfer_body_order,
        central_body)

    return transfer_trajectory_object


def get_node_times(departure_date: float,
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

# NOT USED ANYMORE
# def get_leg_free_parameters(free_parameters: np.ndarray,
#                             transfer_body_order: list,
#                             number_of_revolutions: np.array = np.zeros(1)) -> list:
#     """
#     Forms list of leg free parameters based on free_parameter vector
# 
#     Returns
#     -------
#     list[list[float]]
# 
#     """
#     # free_parameters = np.ones(len(free_parameters))
#     
#     number_of_revolutions = np.zeros(len(transfer_body_order)-1)
#     leg_free_parameters = list()
#     for i in range(len(transfer_body_order)-1):
#         leg_parameters = list()
#         leg_parameters.append(number_of_revolutions[i])
#         # for j in free_parameters:
#         #     leg_parameters.append(j)
#         leg_free_parameters.append(leg_parameters)
# 
#     return leg_free_parameters

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

    assert len(incoming_velocities) == len(swingby_periapses)
    # Departure node
    node_free_parameters.append(np.array([departure_velocity, 0, 0]))#  departure_velocity

    assert len(incoming_velocities) == len(swingby_periapses)

    # Swingby nodes
    for i in range(len(transfer_body_order)-2): # no_of_gas
        node_parameters = list()
        node_parameters.append(incoming_velocities[i])
        # node_parameters.append(100)
        node_parameters.append(0)
        node_parameters.append(0)
        node_parameters.append(swingby_periapses[i])
        # node_parameters.append(10)
        node_parameters.append(0)
        node_parameters.append(0)

        node_free_parameters.append(node_parameters)

    # Arrival node
    node_free_parameters.append(np.array([arrival_velocity, 0, 0], dtype=float))
    
    return node_free_parameters

def get_radial_velocity_shaping_functions(time_of_flight: float,
                                             no_of_free_parameters: int = 2,
                                             frequency: float = np.pi,
                                             scale_factor: float = 1,
                                             number_of_revolutions: int = 0,
                                             manual_base_functions=False) -> list:

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
    if manual_base_functions:
        radial_velocity_shaping_functions = [shape_based_thrust.hodograph_constant()]
        radial_velocity_shaping_functions.append(shape_based_thrust.hodograph_power(exponent=1.0))
        radial_velocity_shaping_functions.append(shape_based_thrust.hodograph_power(exponent=2.0))
    else:
        radial_velocity_shaping_functions = \
        shape_based_thrust.recommended_radial_hodograph_functions(time_of_flight)

    # Add degrees of freedom (highest-order in Gondelach and Noomen, 2015)
    if no_of_free_parameters > 0:
        radial_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_sine(
            exponent=1.0,
            frequency=0.5 * frequency, # 0.25 if leon stubbig, 0.5 if gondelach
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
                                             number_of_revolutions: int = 0,
                                             manual_base_functions=False) -> list: # -> \
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
    if manual_base_functions:
        normal_velocity_shaping_functions = [shape_based_thrust.hodograph_constant()]
        normal_velocity_shaping_functions.append(shape_based_thrust.hodograph_power(exponent=1.0))
        normal_velocity_shaping_functions.append(shape_based_thrust.hodograph_power(exponent=2.0))
    else:
        normal_velocity_shaping_functions = \
        shape_based_thrust.recommended_normal_hodograph_functions(time_of_flight)

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
                                             number_of_revolutions: int = 0,
                                             manual_base_functions=False) -> list: # -> \
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
    if manual_base_functions:
        axial_velocity_shaping_functions = [shape_based_thrust.hodograph_cosine(
            frequency=(number_of_revolutions+0.5)*frequency)]
        axial_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_cosine( 
            exponent=3.0, 
            frequency=(number_of_revolutions+0.5)*frequency,
            scale_factor=scale_factor))
        axial_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_sine(
            exponent=3.0,
            frequency=(number_of_revolutions+0.5)*frequency,
            scale_factor=scale_factor))
    else:
        axial_velocity_shaping_functions = shape_based_thrust.recommended_axial_hodograph_functions(
            time_of_flight,
            number_of_revolutions)

    exponent = 4.0
    # Add degrees of freedom (highest-order in Gondelach and Noomen, 2015)
    if no_of_free_parameters > 0:
        axial_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_cosine(
            exponent=exponent,
            frequency=(number_of_revolutions + 0.5)*frequency,
            scale_factor=scale_factor ** exponent))
            # scale_factor=1))
    if no_of_free_parameters > 1:
        axial_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_sine(
            exponent=exponent,
            frequency=(number_of_revolutions + 0.5)*frequency,
            scale_factor=scale_factor ** exponent))
            # scale_factor=1))
    if no_of_free_parameters > 2:
        axial_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_cosine(
            exponent=4.0, #randomly chosen
            frequency=0.5 * frequency,
            scale_factor=scale_factor))

    return axial_velocity_shaping_functions

def create_modified_system_of_bodies(departure_date=None, central_body_mu=None, bodies=["Sun",
    "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"],
    ephemeris_type='JPL', planet_kep_states = None):
    frame_origin = 'Sun'
    frame_orientation = 'ECLIPJ2000'
    # central_body_mu = 1.3271244e20 # m^3 / s^2
    # departure_date = dpv()[0]
    # departure_date = 10000*constants.JULIAN_DAY
    #
    # planet_kep_states = [[0, 1, 0, 0, 0, 0], [0.38709927,      0.20563593 ,     7.00497902 ,     252.25032350,
    #         77.45779628,     48.33076593],
    # [0.72333566  ,    0.00677672  ,    3.39467605   ,   181.97909950  ,  131.60246718   ,  76.67984255],
    # [1.00000261  ,    0.01671123  ,   -0.00001531   ,   100.46457166  ,  102.93768193   ,   0.0],
    # [1.52371034  ,    0.09339410  ,    1.84969142   ,    -4.55343205  ,  -23.94362959   ,  49.55953891],
    # [5.20288700  ,    0.04838624  ,    1.30439695   ,    34.39644051  ,   14.72847983   , 100.47390909],
    # [9.53667594  ,    0.05386179  ,    2.48599187   ,    49.95424423  ,   92.59887831   , 113.66242448],
    # [19.18916464 ,     0.04725744 ,     0.77263783  ,    313.23810451 ,   170.95427630  ,   74.01692503],
    # [30.06992276 ,     0.00859048 ,     1.77004347  ,    -55.12002969 ,    44.96476227  ,
    #     131.78422574]]

    # this needs spice
    # print('Before spice data is needed')
    body_list_settings = lambda : \
        environment_setup.get_default_body_settings(bodies=bodies,
                base_frame_origin=frame_origin, base_frame_orientation=frame_orientation)
    # print('After spice data is needed')
    # print(bodies)
    for it, i in enumerate(bodies):
        # print(it)
        current_body_list_settings = body_list_settings() #cannot be moved out of scope yet.
        current_body_list_settings.add_empty_settings(i)            
        if ephemeris_type=='JPL':
            current_body_list_settings.get(i).ephemeris_settings = \
            environment_setup.ephemeris.approximate_jpl_model(i)        
        elif ephemeris_type=='KEPFROMSPICE':
            current_body_list_settings.get(i).ephemeris_settings = \
            environment_setup.ephemeris.keplerian_from_spice(i, 
                    departure_date,
                    central_body_mu,
                    frame_origin,
                    frame_orientation)
        elif ephemeris_type=='KEP':
            current_body_list_settings.get(i).ephemeris_settings = \
            environment_setup.ephemeris.keplerian(planet_kep_states[it], 
                    departure_date,
                    central_body_mu,
                    frame_origin,
                    frame_orientation)


            
        # print(current_body_list_settings.get(i).ephemeris_settings)

    return environment_setup.create_system_of_bodies(current_body_list_settings)
    # self.system_of_bodies = lambda : system_of_bodies


def get_delivery_mass(thrust_acceleration, Isp, m0, g0=9.81):
    # Isp = 4000
    # g0 = 9.81
    # m = 1300
    m = m0

    for i in range(len(thrust_acceleration)-1):
        dt = (thrust_acceleration[i+1, 0]-thrust_acceleration[i, 0])
        dvdt = np.linalg.norm(thrust_acceleration[i, 1:])
        dmdt = (1/(Isp*g0)) * m * dvdt
        m -= dmdt * dt

    return m


def hodographic_shaping_visualisation(dir=None , dir_of_dir=None , trajectory_function=trajectory_3d):
    """
    Function that plots the relevant data provided by the input parameters. Generally this function
    is designed for MGA trajectories, and is specifically adapted to use data created by multiple
    islands with PyGMO

    Parameters
    ----------
    dir : directory in which to look for state_history, thrust_acceleration, node_times, and auxiliary
    information
    dir_of_dir : directory in which to look for directories of various islands that all have
    to-be plotted data
    trajectory_function : function from tudatpy that visualizes mga trajectories based on specific
    inputs

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the 3D plot.
    ax : matplotlib.axes.Axes
        3D axis system used to plot the trajectory.
    """

    if dir == None and dir_of_dir == None:
        raise RuntimeError('No directory has been provided that contains to-be plotted data') 
    if dir != None and dir_of_dir != None:
        raise RuntimeError('Too many resources are provided. Please only pass a directory or \
                directory of directories to dir or dir_of_dir, respectively') 

    if dir != None:
        input_directory = dir
    elif dir_of_dir != None:
        input_directory = dir_of_dir
    else:
        raise RuntimeError('Something went wrong, check source')

    state_history = np.loadtxt(dir + 'state_history.dat')
    node_times = np.loadtxt(dir + 'node_times.dat')
    auxiliary_info = np.loadtxt(dir + 'auxiliary_info.dat', delimiter=',', dtype=str)
    # print(auxiliary_info[0][0])
    # print(auxiliary_info[1])
    # aux_info_list= [i.replace('\t') for i in auxiliary_info]
    # print(aux_info_list)

    state_history_dict = {}
    for i in range(len(state_history)):
        state_history_dict[state_history[i, 0]] = state_history[i,1:]
    # print(state_history_dict)
    auxiliary_info_dict = {}
    for i in range(len(auxiliary_info)):
        auxiliary_info_dict[auxiliary_info[i][0]] = auxiliary_info[i][1].replace('\t', '')

    # print(node_times[0, 1])
    # print(state_history_dict)
    # print(node_times)
    fly_by_states = np.array([state_history_dict[node_times[i, 1]] for i in range(len(node_times))])
    # print(fly_by_states)

    fig, ax = trajectory_function(
            state_history_dict,
            vehicles_names=["Spacecraft"],
            central_body_name="SSB",
            bodies=["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter"],
            frame_orientation= 'ECLIPJ2000'
            )
    # print(auxiliary_info_dict)
    ax.set_title(auxiliary_info_dict['MGA Sequence'], fontweight='semibold', fontsize=18)
    # ax.scatter(fly_by_states[0, 0] , fly_by_states[0, 1] , fly_by_states[0,
    #         2] , marker='+', color='yellow', label='Earth departure')
    # ax.scatter(fly_by_states[1, 0] , fly_by_states[1, 1] , fly_by_states[1,
    #         2] , marker='+', color='yellow', label='Mars arrival')
    # ax.scatter(fly_by_states[2, 0] , fly_by_states[2, 1] , fly_by_states[2,
    #         2] , marker='+', color='yellow', label='Mars fly-by')
    # ax.scatter(fly_by_states[3, 0] , fly_by_states[3, 1] , fly_by_states[3,
    #         2] , marker='+', color='yellow', label='Mars fly-by')

# Change the size of the figure
    fig.set_size_inches(8, 8)
# Show the plot
    plt.show()



def objective_per_generation_visualisation(dir=None):

    fitness_values = np.loadtxt(dir + 'champ_f_per_gen.dat')
    variable_values = np.loadtxt(dir + 'champs_per_gen.dat')

    generation_count = fitness_values[:50, 0]
    fitness_values = fitness_values[:50, 1:]
    variable_values = variable_values[:50, 1:]
    # print(fitness_values)
    # print(generation_count)


    no_of_parameters = len(variable_values[0, :])
    no_of_int_parameters = 5
    integer_bounds = (-10, -5)
    continuous_bounds = (-5.12, 5.12)

    minimum_fitness_value =  no_of_int_parameters**3
    color_dict = {0 : 'r', 1 : 'b'}
    color = [color_dict[0] if i < no_of_int_parameters else color_dict[1] for i in
            range(no_of_parameters)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    ax1.plot(generation_count, fitness_values)
    ax1.plot(generation_count, [minimum_fitness_value for i in range(len(generation_count))], c='k',
            linestyle='--', linewidth=0.8, label='Constraint value')
    ax1.set_xlabel('Generation count [-]')
    ax1.set_ylabel(' Function value [-]' )
    # ax1.set_yscale('log')
    ax1.legend()
    ax1.grid()

    for i in range(no_of_parameters):
        ax2.plot(generation_count, variable_values[:, i], c=color[i])
    ax2.plot(generation_count, [integer_bounds[0] for i in range(len(generation_count))], c='k',
            linewidth=0.8, linestyle='--', label='Constraint value')
    ax2.plot(generation_count, [integer_bounds[1] for i in range(len(generation_count))], c='k',
            linestyle='--', linewidth=0.8)
    ax2.plot(generation_count, [continuous_bounds[0] for i in range(len(generation_count))], c='k',
            linestyle='--', linewidth=0.8)
    ax2.plot(generation_count, [continuous_bounds[1] for i in range(len(generation_count))], c='k',
            linestyle='--', linewidth=0.8)
    ax2.set_xlabel('Generation count [-]')
    ax2.set_ylabel(' Variable value [-]' )
    ax2.legend()

    ax2.grid()
    plt.show()

