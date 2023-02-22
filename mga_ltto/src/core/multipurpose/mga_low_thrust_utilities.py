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

# General
import numpy as np

# Tudatpy 
from tudatpy.kernel import constants
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.trajectory_design import shape_based_thrust
from tudatpy.kernel.trajectory_design import transfer_trajectory
from tudatpy.kernel.interface import spice
spice.load_standard_kernels()

# import warnings
# warnings.filterwarnings("error")

class transfer_body_order_conversion:
    
    def __init__(self):
        pass

    @staticmethod
    def get_transfer_body_integers(transfer_body_list: list, strip=True) -> np.ndarray:
        """
        Input : ["Venus", "Venus", "Earth", "Mercury"]
        Ouput : np.array([1, 1, 2, 4])
        """
        #transfer_body_list = transfer_body_list[1:-1]
        body_dict = {0: "Mercury",
                1: "Venus",
                2: "Earth",
                3: "Mars",
                4: "Jupiter",
                5: "Saturn",
                6: "Uranus",
                7: "Neptune"}

        body_values = list(body_dict.values())
        body_keys = list(body_dict.keys())
        body_list = []

        for body in transfer_body_list:
            for j in range(len(body_values)):
                if body == body_values[j]:
                    body_list.append(body_keys[j])
                # elif body == body_values[j] and strip == False:
                #     body_list.append(body_keys[j])

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
    def get_mga_character_list_from_list(bodylist: list) -> str():
        """
        Input : ["Earth", "Venus", "Mercury"]
        Output : ["E", "V", "Y"]
        """

        character_dict = {'Y' : "Mercury",
                'V' : "Venus",
                'E' : "Earth",
                'M' : "Mars",
                'J' : "Jupiter",
                'S' : "Saturn",
                'U' : "Uranus",
                'N' : "Neptune"}

        mga_char_list = []
        for i in bodylist:
            for j, k in character_dict.items():
                if i == k:
                    mga_char_list.append(j)

        return mga_char_list

    @staticmethod
    def get_mga_characters_from_list(bodylist: list) -> str():
        """
        Input : ["Earth", "Venus", "Mercury"]
        Output : "EVY"
        """

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
    def get_mga_list_from_characters(character_string: str) -> list:
        """
        Input : "EVY"
        Output : ["Earth", "Venus", "Mercury"]
        """


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

        Example
        --------
        Input : ["EEEMJ"]
        Output : ["EE", "EE", "EM", "MJ"]

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
                                        # departure_elements : tuple,
                                        # target_elements : tuple,
                                        bodies,
                                        central_body,
                                        no_of_free_parameters: int = 0,
                                        manual_base_functions=False,
                                        number_of_revolutions: np.ndarray = np.zeros(7, dtype=int),
                                        dynamic_shaping_functions=False)\
                                        -> transfer_trajectory.TransferTrajectory:
    """
    Provides the transfer settings required to construct a hodographic shaping mga trajectory.

    """

    # departure and target state from ephemerides
#     body_dict = {"Mercury" : 0,
#             "Venus" : 1,
#             "Earth" : 2,
#             "Mars" : 3,
#             "Jupiter" : 4,
#             "Saturn" : 5,
#             "Uranus" : 6,
#             "Neptune" : 7}
# 
#     # From approximate jpl model states
#     planet_kep_states = [[0.38709927,      0.20563593 ,     7.00497902 ,     252.25032350, 77.45779628,     48.33076593],
                        # [0.72333566  ,    0.00677672  ,    3.39467605   ,   181.97909950  ,  131.60246718   ,  76.67984255],
                        # [1.00000261  ,    0.01671123  ,   -0.00001531   ,   100.46457166  ,  102.93768193   ,   0.0],
                        # [1.52371034  ,    0.09339410  ,    1.84969142   ,    -4.55343205  ,  -23.94362959   ,  49.55953891],
                        # [5.20288700  ,    0.04838624  ,    1.30439695   ,    34.39644051  ,   14.72847983   , 100.47390909],
                        # [9.53667594  ,    0.05386179  ,    2.48599187   ,    49.95424423  ,   92.59887831   , 113.66242448],
                        # [19.18916464 ,     0.04725744 ,     0.77263783  ,    313.23810451 ,   170.95427630  ,   74.01692503],
                        # [30.06992276 ,     0.00859048 ,     1.77004347  ,    -55.12002969 ,    44.96476227  , 131.78422574]]
# 
#     astronomical_unit = 149597870.7e3 #m

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
    # departure_planet = transfer_body_order[0]
    # departure_planet_integer = transfer_body_order_conversion.get_transfer_body_integers([departure_planet])

    for i in range(len(transfer_body_order)-1):

        if dynamic_shaping_functions:
            shaping_function_type = get_current_flyby_direction(i, transfer_body_order)  
            # print(shaping_function_type)
        else:
            shaping_function_type = 'default'

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
                                            manual_base_functions=manual_base_functions,
                                            shaping_function_type=shaping_function_type)
        normal_velocity_functions = get_normal_velocity_shaping_functions(time_of_flight[i],
                                            number_of_revolutions=number_of_revolutions[i],
                                            no_of_free_parameters=no_of_free_parameters,
                                            frequency=frequency,
                                            scale_factor=scale_factor,
                                            manual_base_functions=manual_base_functions,
                                            shaping_function_type=shaping_function_type)
        axial_velocity_functions = get_axial_velocity_shaping_functions(time_of_flight[i],
                                            number_of_revolutions=number_of_revolutions[i],
                                            no_of_free_parameters=no_of_free_parameters,
                                            frequency=frequency,
                                            scale_factor=scale_factor,
                                            manual_base_functions=manual_base_functions,
                                            shaping_function_type=shaping_function_type)

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

def get_node_free_parameters(transfer_body_order: list, 
                             swingby_periapses: np.ndarray=None,
                             incoming_velocities: np.ndarray=None, 
                             orbit_ori_angles : np.ndarray = None,
                             swingby_inplane_angles : np.ndarray = None,
                             swingby_outofplane_angles : np.ndarray = None,
                             dsm_deltav: np.ndarray = None, 
                             departure_velocity: float = 0,
                             departure_inplane_angle : float = 0,
                             departure_outofplane_angle: float = 0,
                             arrival_velocity: float = 0,
                             arrival_inplane_angle : float = 0,
                             arrival_outofplane_angle : float = 0) -> list:
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

    node_free_parameters = []

    if incoming_velocities is not None and swingby_periapses is not None:
        assert len(incoming_velocities) == len(swingby_periapses) 
    # Departure node
    node_free_parameters.append(np.array([departure_velocity, departure_inplane_angle,
                                          departure_outofplane_angle]))#  departure_velocity

    # Swingby nodes
    for i in range(len(transfer_body_order)-2): # no_of_gas
        node_parameters = list()
        node_parameters.append(incoming_velocities[i] if incoming_velocities[0]  is not None else 0)
        node_parameters.append(swingby_inplane_angles[i] if swingby_inplane_angles[0]  is not None else 0)
        node_parameters.append(swingby_outofplane_angles[i] if swingby_outofplane_angles[0]  is not None else 0)
        node_parameters.append(swingby_periapses[i] if swingby_periapses[0]  is not None else 0)
        node_parameters.append(orbit_ori_angles[i] if orbit_ori_angles[0]  is not None else 0)
        # node_parameters.append(dsm_deltav[i] if dsm_deltav[0] is not None else 0)
        node_parameters.append(0)

        node_free_parameters.append(node_parameters)

    # Arrival node
    node_free_parameters.append(np.array([arrival_velocity, arrival_inplane_angle,
                                          arrival_outofplane_angle]))
    
    return node_free_parameters

def get_radial_velocity_shaping_functions(time_of_flight: float,
                                             no_of_free_parameters: int = 2,
                                             frequency: float = np.pi,
                                             scale_factor: float = 1,
                                             number_of_revolutions: int = 0,
                                             manual_base_functions=False,
                                             shaping_function_type='outer') -> list:

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
                                             manual_base_functions=False,
                                             shaping_function_type='outer') -> list: # -> \
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
        normal_velocity_shape_function_type = [shape_based_thrust.hodograph_constant()]
        normal_velocity_shape_function_type.append(shape_based_thrust.hodograph_power(exponent=1.0))
        normal_velocity_shape_function_type.append(shape_based_thrust.hodograph_power(exponent=2.0))
    else:
        normal_velocity_shape_function_type = \
        shape_based_thrust.recommended_normal_hodograph_functions(time_of_flight)

    # Add degrees of freedom (highest-order in Gondelach and Noomen, 2015)
    if no_of_free_parameters > 0:
        normal_velocity_shape_function_type.append(shape_based_thrust.hodograph_scaled_power_sine(
            exponent=1.0,
            frequency=0.5 * frequency,
            scale_factor=scale_factor))
    if no_of_free_parameters > 1:
        normal_velocity_shape_function_type.append(shape_based_thrust.hodograph_scaled_power_cosine(
            exponent=1.0,
            frequency=0.5 * frequency,
            scale_factor=scale_factor))
    if no_of_free_parameters > 2:
        normal_velocity_shape_function_type.append(shape_based_thrust.hodograph_scaled_power_cosine(
            exponent=1.0,
            frequency=0.5 * frequency,
            scale_factor=scale_factor))

    return normal_velocity_shape_function_type

def get_axial_velocity_shaping_functions(time_of_flight: float,
                                             no_of_free_parameters: int = 2,
                                             frequency: float = np.pi,
                                             scale_factor: float = 1,
                                             number_of_revolutions: int = 0,
                                             manual_base_functions=False,
                                             shaping_function_type='outer') -> list: # -> \
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
    base_function_exponent = 3.0
    added_velocity_function_exponent = 4.0
    if shaping_function_type == 'inner':
        base_function_exponent = 5.0 # +2
        added_velocity_function_exponent = 6.0 # +2

    # Add degrees of freedom (lowest-order in Gondelach and Noomen, 2015)
    if manual_base_functions:
        axial_velocity_shape_function_type = [shape_based_thrust.hodograph_cosine(
            frequency=(number_of_revolutions+0.5)*frequency)]
        axial_velocity_shape_function_type.append(shape_based_thrust.hodograph_scaled_power_cosine( 
            exponent=base_function_exponent, 
            frequency=(number_of_revolutions+0.5)*frequency,
            scale_factor=scale_factor))
        axial_velocity_shape_function_type.append(shape_based_thrust.hodograph_scaled_power_sine(
            exponent=base_function_exponent,
            frequency=(number_of_revolutions+0.5)*frequency,
            scale_factor=scale_factor))
    else:
        axial_velocity_shape_function_type = shape_based_thrust.recommended_axial_hodograph_functions(
            time_of_flight,
            number_of_revolutions)

    # exponent = 4.0
    # Add degrees of freedom (highest-order in Gondelach and Noomen, 2015)
    if no_of_free_parameters > 0:
        axial_velocity_shape_function_type.append(shape_based_thrust.hodograph_scaled_power_cosine(
            exponent=added_velocity_function_exponent,
            frequency=(number_of_revolutions + 0.5)*frequency,
            scale_factor=scale_factor ** added_velocity_function_exponent))
            # scale_factor=1))
    if no_of_free_parameters > 1:
        axial_velocity_shape_function_type.append(shape_based_thrust.hodograph_scaled_power_sine(
            exponent=added_velocity_function_exponent,
            frequency=(number_of_revolutions + 0.5)*frequency,
            scale_factor=scale_factor ** added_velocity_function_exponent))
            # scale_factor=1))
    if no_of_free_parameters > 2:
        axial_velocity_shape_function_type.append(shape_based_thrust.hodograph_scaled_power_cosine(
            exponent=added_velocity_function_exponent, #randomly chosen
            frequency=0.5 * frequency,
            scale_factor=scale_factor))

    return axial_velocity_shape_function_type

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
    current_body_list_settings = body_list_settings() #cannot be moved out of scope yet.
    for it, i in enumerate(bodies):
        # print(it)
        # current_body_list_settings.add_empty_settings(i)            
        if ephemeris_type=='JPL':
            if i == "Sun":
                central_body_mu = 1.3271244e20 # m^3 / s^2
                current_body_list_settings.get(i).ephemeris_settings = \
                environment_setup.ephemeris.keplerian([0, 0, 0, 0, 0, 0], 1000*constants.JULIAN_DAY,
                        central_body_mu, 'SSB', frame_orientation)
            else:
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
        elif ephemeris_type=='NEW':
            current_body_list_settings = environment_setup.BodyListSettings(frame_origin, frame_orientation)
            for i in bodies:
                current_body_list_settings.add_empty_settings(i)            
                if i == "Sun":
                    central_body_mu = 1.3271244e20 # m^3 / s^2
                    current_body_list_settings.get(i).ephemeris_settings = \
                    environment_setup.ephemeris.keplerian([0, 0, 0, 0, 0, 0], 1000*constants.JULIAN_DAY,
                            central_body_mu, 'SSB', frame_orientation)
                else:
                    current_body_list_settings.get(i).ephemeris_settings = \
                    environment_setup.ephemeris.approximate_jpl_model(i)        


            
    # print(current_body_list_settings.get("Mars").ephemeris_settings)

    return environment_setup.create_system_of_bodies(current_body_list_settings)
    # self.system_of_bodies = lambda : system_of_bodies

def create_modified_system_of_bodies_validation(bodies=["Sun", "Mercury", "Venus", "Earth"]):


    departure_body = 'Depart'
    arrival_body = 'Arriva'
    # No thrust
    # departure_state = [1.32331228e+11, 6.66542553e+10, 8.54359784e+07, -1.70376556e+04, 2.53431018e+04, 2.73557351e+02]
    # arrival_state = [5.11725300e+10, -3.53584027e+08, -4.79993046e+09, -1.29567837e+04, 5.42787716e+04, 5.54404347e+03]

    #With thrust
    # 639550000.0
    departure_state = [-1.42103917e+11, -4.49049607e+10, 8.80866518e+07, 5.02094627e+03, -3.01872576e+04, -2.74769320e+02]
    # 640040000.0
    arrival_state = [-1.38878050e+11, -5.83263232e+10, -1.91905921e+08, 8.61923618e+03, -2.45349516e+04, -7.99796323e+02]

    frame_origin = 'Sun'
    frame_orientation = 'ECLIPJ2000'

    # this needs spice
    # print('Before spice data is needed')
    body_list_settings = lambda : \
        environment_setup.get_default_body_settings(bodies=bodies,
                base_frame_origin=frame_origin, base_frame_orientation=frame_orientation)
    # print('After spice data is needed')
    # print(bodies)
    current_body_list_settings = body_list_settings() #cannot be moved out of scope yet.
    for it, i in enumerate(bodies):
        # print(it)
        # current_body_list_settings.add_empty_settings(i)            
        if i == "Sun":
            central_body_mu = 1.3271244e20 # m^3 / s^2
            current_body_list_settings.get(i).ephemeris_settings = \
            environment_setup.ephemeris.keplerian([0, 0, 0, 0, 0, 0], 1000*constants.JULIAN_DAY,
                    central_body_mu, 'SSB', frame_orientation)
        else:
            current_body_list_settings.get(i).ephemeris_settings = \
            environment_setup.ephemeris.approximate_jpl_model(i)        

    # Add depart
    current_body_list_settings.add_empty_settings(departure_body)
    mu_depart = 1 # m^3 / s^2
    current_body_list_settings.get(departure_body).gravity_field_settings = \
    environment_setup.gravity_field.central(mu_depart)
    current_body_list_settings.get(departure_body).ephemeris_settings = \
    environment_setup.ephemeris.constant(departure_state, 'SSB', frame_orientation)

    # Add arriva
    current_body_list_settings.add_empty_settings(arrival_body)
    mu_arriva = 1 # m^3 / s^2
    current_body_list_settings.get(arrival_body).gravity_field_settings = \
    environment_setup.gravity_field.central(mu_arriva)
    current_body_list_settings.get(arrival_body).ephemeris_settings = \
    environment_setup.ephemeris.constant(arrival_state, 'SSB', frame_orientation)

    return environment_setup.create_system_of_bodies(current_body_list_settings)

def get_mass_propagation(thrust_acceleration : dict, Isp, m0, g0=9.81):

    m = m0
    time_history = np.array(list(thrust_acceleration.keys()))
    thrust_acceleration = np.array(list(thrust_acceleration.values()))
    mass_history = {}
    invalid_trajectory = False

    mass_history[time_history[0]] = m0
    for it, thrust in enumerate(thrust_acceleration):
        if it == len(thrust_acceleration)-1:
            break
        dt = time_history[it+1] - time_history[it]
        dvdt = np.linalg.norm(thrust)
        dmdt = (1/(Isp*g0)) * m * dvdt
        m -= dmdt * dt
        mass_history[time_history[it+1]] = m
        if np.abs(m) < 1e-7:
            m = 0 
            # print('Mass approaches 0')
            invalid_trajectory = True
            break

    return mass_history, mass_history[time_history[it]], invalid_trajectory


def get_flyby_directions(transfer_body_order : list):
    flyby_types = []
    for i in range(len(transfer_body_order)-1):
        previous_body = transfer_body_order[i]
        previous_body_integer = \
        transfer_body_order_conversion.get_transfer_body_integers([previous_body])
        flyby_target_body = transfer_body_order[i+1]
        flyby_target_body_integer = \
        transfer_body_order_conversion.get_transfer_body_integers([flyby_target_body])

        if flyby_target_body_integer < previous_body_integer: # this works
            flyby_types.append('inner')
        elif flyby_target_body_integer > previous_body_integer:
            flyby_types.append('outer')
        else:
            flyby_types.append('same')

    return flyby_types

def get_current_flyby_direction(i : int, transfer_body_order : list):
    previous_body = transfer_body_order[i]
    previous_body_integer = \
    transfer_body_order_conversion.get_transfer_body_integers([previous_body])
    flyby_target_body = transfer_body_order[i+1]
    flyby_target_body_integer = \
    transfer_body_order_conversion.get_transfer_body_integers([flyby_target_body])

    if flyby_target_body_integer < previous_body_integer: # this works
        flyby_type = 'inner'
    elif flyby_target_body_integer > previous_body_integer:
        flyby_type = 'outer'
    else:
        flyby_type = 'same'
    return flyby_type

