'''

Author: Sean Cowan
Purpose: MSc Thesis
Date Created: 26-07-2022

This module creates a PyGMO compatible problem class that represents the mga low-thrust trajectories.
'''
###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General 
import numpy as np
import sys
import os

# Tudatpy 
from tudatpy.kernel import constants
from tudatpy.util import result2array

current_dir = os.getcwd()
sys.path.append('/Users/sean/Desktop/tudelft/thesis/code/mga_ltto/src/') # this only works if you run ltto and mgso while in the directory that includes those files
# Local
import core.multipurpose.mga_low_thrust_utilities as util

#######################################################################
# PROBLEM CLASS #######################################################
#######################################################################

class MGALowThrustTrajectoryOptimizationProblem:
    """
    Class to initialize, simulate, and optimize the MGA low-thrust trajectory optimization problem.
    """

    def __init__(self,
                    transfer_body_order,
                    no_of_free_parameters = 0,
                    bounds = [[], []], 
                    depart_semi_major_axis=np.inf,
                    depart_eccentricity=0, 
                    target_semi_major_axis=np.inf, 
                    target_eccentricity=0,
                    swingby_altitude=200000000, #2e5 km
                    departure_velocity=2000, 
                    arrival_velocity=0,
                    Isp=3000,
                    m0=1000,
                    no_of_points=500,
                    manual_base_functions=False,
                    dynamic_bounds = {'time_of_flight' : False,
                                      'orbit_ori_angle' : False,
                                      'swingby_outofplane' : False,
                                      'swingby_inplane' : False,
                                      'shaping_function' : False},
                    manual_tof_bounds=None,
                    objectives=['dv'],
                    zero_revs=False):

        self.transfer_body_order = transfer_body_order
        self.mga_characters = \
        util.transfer_body_order_conversion.get_mga_characters_from_list(self.transfer_body_order)
        self.legstrings = \
        util.transfer_body_order_conversion.get_list_of_legs_from_characters(self.mga_characters)
        self.no_of_gas = len(transfer_body_order)-2
        self.target_body = transfer_body_order[0]
        self.depart_body = transfer_body_order[-1]
        self.no_of_legs = len(transfer_body_order) - 1

        self.no_of_free_parameters = no_of_free_parameters
        # self.no_of_free_parameters = 2 if len(transfer_body_order) <= 3 else 1
        self.total_no_of_free_coefficients = self.no_of_free_parameters*3*(self.no_of_legs)

        self.bounds = bounds

        self.depart_semi_major_axis = depart_semi_major_axis
        self.depart_eccentricity = depart_eccentricity
        self.target_semi_major_axis = target_semi_major_axis
        self.target_eccentricity = target_eccentricity

        self.swingby_altitude = swingby_altitude
        self.departure_velocity = departure_velocity
        self.arrival_velocity = arrival_velocity

        self.Isp = Isp
        self.m0 = m0
        self.no_of_points = no_of_points

        self.objectives = objectives
        self.zero_revs = zero_revs

        self.design_parameter_vector = None

        self.transfer_trajectory_object = None
        self.delivery_mass = None
        self.node_times = None

        self.planetary_radii = {'Sun': 696000000.0, 'Mercury': 2439699.9999999995, 'Venus':
                6051800.0, 'Earth': 6371008.366666666, 'Mars': 3389526.6666666665, 'Jupiter':
                69946000.0, 'Saturn': 58300000.0}

        self.manual_base_functions = manual_base_functions
        self.dynamic_bounds = dynamic_bounds
        self.manual_tof_bounds = manual_tof_bounds
        self.bound_names= ['Departure date [mjd2000]', 'Departure velocity [m/s]', 'Arrival velocity [m/s]',
                           'Time of Flight [s]', 'Incoming velocity [m/s]', 'Swingby periapsis [m]',
                           'Free coefficient [-]', 'Number of revolutions [-]']


    def get_tof_bound(self, leg_number : int, original_bounds : tuple):

        planet_kep_states = [[0.38709927,      0.20563593 ,     7.00497902 ,     252.25032350, 77.45779628,     48.33076593],
        [0.72333566  ,    0.00677672  ,    3.39467605   ,   181.97909950  ,  131.60246718   ,  76.67984255],
        [1.00000261  ,    0.01671123  ,   -0.00001531   ,   100.46457166  ,  102.93768193   ,   0.0],
        [1.52371034  ,    0.09339410  ,    1.84969142   ,    -4.55343205  ,  -23.94362959   ,  49.55953891],
        [5.20288700  ,    0.04838624  ,    1.30439695   ,    34.39644051  ,   14.72847983   , 100.47390909],
        [9.53667594  ,    0.05386179  ,    2.48599187   ,    49.95424423  ,   92.59887831   , 113.66242448],
        [19.18916464 ,     0.04725744 ,     0.77263783  ,    313.23810451 ,   170.95427630  ,   74.01692503],
        [30.06992276 ,     0.00859048 ,     1.77004347  ,    -55.12002969 ,    44.96476227  , 131.78422574]]

        astronomical_unit = 149597870.7e3 #m
        central_body_mu = 1.3271244e20 # m^3 / s^2

        previous_body = self.transfer_body_order[leg_number]
        next_body = self.transfer_body_order[leg_number+1]
        previous_body_integer = \
        util.transfer_body_order_conversion.get_transfer_body_integers([previous_body])
        next_body_integer = \
        util.transfer_body_order_conversion.get_transfer_body_integers([next_body])

        pseudoperiod = lambda a, e : 2 * np.pi * np.sqrt((a * astronomical_unit *(1+e))**3/  central_body_mu)

        sma_previous = planet_kep_states[previous_body_integer[0]][0]
        sma_next = planet_kep_states[next_body_integer[0]][0]
        ecc_previous = planet_kep_states[previous_body_integer[0]][1]
        ecc_next = planet_kep_states[next_body_integer[0]][1]

        pperiod_previous = pseudoperiod(sma_previous, ecc_previous)
        pperiod_next = pseudoperiod(sma_next, ecc_next)

        if previous_body == next_body:
            # print("Option 1")
            assert pperiod_previous == pperiod_next
            bounds = [pperiod_previous / 4, pperiod_previous * 4]
        elif max([sma_previous, sma_next]) < 2: # if smaller than 2 AU (Smaller than Jupiter)
            # print("Option 2")
            bounds = [0.1 * min([pperiod_previous, pperiod_next]), 4 *  max([pperiod_previous,
                                                                        pperiod_next])]
        elif max([sma_previous, sma_next]) >= 2:
            # print("Option 3")
            bounds = [0.1 * min([pperiod_previous, pperiod_next]), 4 * max([pperiod_previous,
                                                                        pperiod_next])]
        else:
            raise RuntimeError("The periods provided are formatted incorrectly")


        if bounds[0] < original_bounds[0]:
            print(f"Lower bound updated from {original_bounds[0] / 86400} to {bounds[0] / 86400} days")
            bounds[0] = original_bounds[0]
        if bounds[1] > original_bounds[1]:
            print(f"Upper bound updated from {original_bounds[1] / 86400} to {bounds[1] / 86400} days")
            bounds[1] = original_bounds[1]


        return bounds



    def mjd2000_to_seconds(self, mjd2000):
        # mjd2000 = 51544
        # mjd += mjd2000
        mjd2000 *= constants.JULIAN_DAY
        return mjd2000
        # return time_conversion.julian_day_to_seconds_since_epoch(time_conversion.modified_julian_day_to_julian_day(mjd2000))

    def swingby_altitude_to_bound(self, bound):
        return np.floor(np.log10(np.abs(bound))).astype(int)


    def get_bounds(self):
        departure_date_lb = self.mjd2000_to_seconds(self.bounds[0][0])
        departure_date_ub = self.mjd2000_to_seconds(self.bounds[1][0])

        departure_velocity_lb = self.bounds[0][1]
        departure_velocity_ub = self.bounds[1][1]

        arrival_velocity_lb = self.bounds[0][2]
        arrival_velocity_ub = self.bounds[1][2]

        time_of_flight_lb = self.mjd2000_to_seconds(self.bounds[0][3])
        time_of_flight_ub = self.mjd2000_to_seconds(self.bounds[1][3])

        incoming_velocity_lb = self.bounds[0][4]
        incoming_velocity_ub = self.bounds[1][4]

        swingby_altitude_lb = self.bounds[0][5]
        swingby_altitude_ub = self.bounds[1][5]

        free_coefficients_lb = self.bounds[0][6]
        free_coefficients_ub = self.bounds[1][6]

        number_of_revolutions_lb = self.bounds[0][7]
        number_of_revolutions_ub = self.bounds[1][7]

        lower_bounds = [departure_date_lb] # departure date
        upper_bounds = [departure_date_ub] # departure date
        lower_bounds.append(departure_velocity_lb) # departure velocity # FIXED
        upper_bounds.append(departure_velocity_ub) # departure velocity
        lower_bounds.append(arrival_velocity_lb) # departure velocity # FIXED
        upper_bounds.append(arrival_velocity_ub) # departure velocity

        for leg in range(self.no_of_legs): # time of flight
            if self.manual_tof_bounds != None:
                lower_bounds.append(self.manual_tof_bounds[0][leg] * constants.JULIAN_DAY)
                upper_bounds.append(self.manual_tof_bounds[1][leg] * constants.JULIAN_DAY)
            elif self.dynamic_bounds:
                current_time_of_flight_bounds = self.get_tof_bound(leg, (time_of_flight_lb,
                                                                         time_of_flight_ub))
                lower_bounds.append(current_time_of_flight_bounds[0])
                upper_bounds.append(current_time_of_flight_bounds[1])
            else:
                lower_bounds.append(time_of_flight_lb)
                upper_bounds.append(time_of_flight_ub)
        for _ in range(self.no_of_gas):
            lower_bounds.append(incoming_velocity_lb)
            upper_bounds.append(incoming_velocity_ub)
        for _ in range(self.no_of_gas):
            lower_bounds.append(swingby_altitude_lb)
            upper_bounds.append(swingby_altitude_ub)
        for _ in range(self.total_no_of_free_coefficients): # free coefficients
            lower_bounds.append(free_coefficients_lb)
            upper_bounds.append(free_coefficients_ub)
        for _ in range(self.no_of_legs): # number of revolutions
            lower_bounds.append(number_of_revolutions_lb)
            upper_bounds.append(number_of_revolutions_ub)

        return (lower_bounds, upper_bounds)

    def get_nobj(self):
        if len(self.objectives) == 2:
            return 2
        elif len(self.objectives) == 1:
            return 1
        else:
            raise RuntimeError('The amount of objectives has not bee implemented yet, please choose 1 or 2 objectives')

    def get_nic(self):
        return 0

    def get_nix(self):
        # free coefficients, number of revolutions
        return self.no_of_legs 
        # return self.no_of_legs

    def get_states_along_trajectory(self, no_of_points) -> dict:
        """
        Returns the full state history of the hodographic shaping state interpolated at
        'no_of_points' points
        Parameters
        ----------
        no_of_points : number of points interpolated along trajectory
        Returns
        -------
        dict
        """
        return self.transfer_trajectory_object.states_along_trajectory(no_of_points)


    def get_inertial_thrust_accelerations_along_trajectory(self, no_of_points) -> dict:
        """
        Returns the full thrusty acceleration history of the hodographic shaping state interpolated
        at 'no_of_points' points
        Parameters
        ----------
        no_of_points : number of points interpolated along trajectory
        Returns
        -------
        dict
        """
        return self.transfer_trajectory_object.inertial_thrust_accelerations_along_trajectory(no_of_points)

    def get_node_time_list(self):
        """
        Returns the node_times list variable so that it can be used for post-processing
        """
        return self.node_times

    def get_design_parameter_vector(self):
        return self.design_parameter_vector

    def get_objectives(self, transfer_trajectory_object):
        objective_values = []
        for it, j in enumerate(self.objectives):
            if j == 'dv':
                objective_values.append(transfer_trajectory_object.delta_v)
            elif j == 'tof':
                objective_values.append(transfer_trajectory_object.time_of_flight)
            elif j == 'dmf' or j == 'pmf' or j == 'dm':
                thrust_acceleration = \
                transfer_trajectory_object.inertial_thrust_accelerations_along_trajectory(self.no_of_points)

                mass_history, delivery_mass, invalid_trajectory = \
                util.get_mass_propagation(thrust_acceleration, self.Isp, self.m0)
                if delivery_mass < 0:
                    raise RuntimeError('Error with validity of trajectory: the delivery mass is negative.')
                # (self.m0 - delivery_mass) because of the propulsion mass
                if j == 'dmf':
                    objective_values.append(- delivery_mass / self.m0) # try to maximize
                elif j == 'pmf':
                    objective_values.append((self.m0 - delivery_mass) / self.m0) # try to minimize
                elif j == 'dm':
                    objective_values.append(- delivery_mass) # try to maximize
            else:
                raise RuntimeError('An objective was provided that is not permitted')
        return objective_values

    def get_delivery_mass(self):
        thrust_acceleration = \
        self.transfer_trajectory_object.inertial_thrust_accelerations_along_trajectory(self.no_of_points)
        mass_history, self.delivery_mass, invalid_trajectory = \
        util.get_mass_propagation(thrust_acceleration, self.Isp, self.m0)
        return self.delivery_mass

    
    def delivery_mass_constraint_check(self, transfer_object, Isp, m0, no_of_points=500):
        thrust_acceleration = \
        transfer_object.inertial_thrust_accelerations_along_trajectory(no_of_points)

        mass_history, delivery_mass, invalid_trajectory = util.get_mass_propagation(thrust_acceleration, Isp, m0)


        if invalid_trajectory:
            raise RuntimeError('Error with validity of trajectory: the delivery mass is approaching 0 1e-7.')

        return mass_history, delivery_mass
        

    def fitness(self, 
                design_parameter_vector : list, 
                bodies = util.create_modified_system_of_bodies(ephemeris_type='JPL'),
                # bodies = environment_setup.create_simplified_system_of_bodies(),
                post_processing=False):

        """
        Assuming no_of_gas == 2 & #fc == 2
        0 - departure_date
        1 - departure velocity
        2 - arrival velocity
        3..6 - time of flights
        6..8 - incoming velocities
        8..10 - swingby periapses
        10..29 - free_coefficients
        29..32 - number of revolutions
        """
        # print("Design Parameters:", design_parameter_vector, "\n")
        self.design_parameter_vector = design_parameter_vector

        # parameters
        central_body = 'Sun'

        #depart and target elements
        # departure_elements = (self.depart_semi_major_axis, self.depart_eccentricity) 
        # target_elements = (self.target_semi_major_axis, self.target_eccentricity) 

        # indexes
        time_of_flight_index = 3 + self.no_of_legs
        incoming_velocity_index = time_of_flight_index + self.no_of_gas
        swingby_altitude_index = incoming_velocity_index + self.no_of_gas
        free_coefficient_index = swingby_altitude_index + self.total_no_of_free_coefficients
        revolution_index = free_coefficient_index + self.no_of_legs

        ### CONTINUOUS PART ###
        departure_date = np.array(design_parameter_vector[0]) # departure date
        departure_velocity = np.array(design_parameter_vector[1]) # departure velocity
        arrival_velocity = np.array(design_parameter_vector[2]) # arrival velocity
        time_of_flights = np.array(design_parameter_vector[3:time_of_flight_index]) # time of flight
        incoming_velocities = \
        np.array(design_parameter_vector[time_of_flight_index:incoming_velocity_index]) # incoming velocities
        swingby_altitudes = np.array(design_parameter_vector[incoming_velocity_index:swingby_altitude_index]) # swingby_altitudes

        ### INTEGER PART ###
        # hodographic shaping free coefficients
        free_coefficients = \
        np.array(design_parameter_vector[swingby_altitude_index:free_coefficient_index])

        # number of revolutions
        if self.zero_revs:
            number_of_revolutions = \
            np.array([0 for _ in design_parameter_vector[free_coefficient_index:revolution_index]])
        else:
            number_of_revolutions = np.array([int(x) for x in
                      design_parameter_vector[free_coefficient_index:revolution_index]])

        transfer_trajectory_object = util.get_low_thrust_transfer_object(self.transfer_body_order,
                                                        time_of_flights,
                                                        bodies,
                                                        central_body,
                                                        no_of_free_parameters=self.no_of_free_parameters,
                                                        manual_base_functions=self.manual_base_functions,
                                                        number_of_revolutions=number_of_revolutions,
                                                        dynamic_shaping_functions=self.dynamic_bounds['shaping_function'])

        planetary_radii_sequence = np.zeros(self.no_of_gas)
        for i, body in enumerate(self.transfer_body_order[1:-1]):
            planetary_radii_sequence[i] = self.planetary_radii[body]

        # swingby_altitudes = np.array([planetary_radii_sequence[i] + self.swingby_altitude for i in
        #    range(self.no_of_gas)]) # defined depending on problem
        swingby_periapses = np.array([planetary_radii_sequence[i] + swingby_altitudes[i] for i in
            range(self.no_of_gas)]) # defined depending on problem
        # incoming_velocities = np.array([2000 for _ in range(self.no_of_gas)]) 
        # incoming_velocity= np.array([incoming_velocities[i] for i in range(self.no_of_gas)])

        # node times
        self.node_times = util.get_node_times(departure_date, time_of_flights)
        # print(node_times)

        # leg free parameters 
        leg_free_parameters = np.concatenate(np.array([np.append(number_of_revolutions[i],
            free_coefficients[i*(self.no_of_free_parameters*3):(i+1)*(self.no_of_free_parameters*3)]) for i
            in range(self.no_of_legs)])).reshape((self.no_of_legs, 1 + 3*
                self.no_of_free_parameters)) # added reshape

        # node free parameters
        node_free_parameters = util.get_node_free_parameters(self.transfer_body_order,
                                                             swingby_periapses, 
                                                             incoming_velocities, 
                                                             departure_velocity=departure_velocity,
                                                             arrival_velocity=arrival_velocity)

        try:
            transfer_trajectory_object.evaluate(self.node_times, leg_free_parameters, node_free_parameters)
            # self.delivery_mass_constraint_check(transfer_trajectory_object, self.Isp, self.m0, self.no_of_points)

            # self.transfer_trajectory_object = transfer_trajectory_object
            #
            # if post_processing == False:
            #     objective = self.get_objectives()
            # else:
            #     return

            if post_processing == False:
                objective = self.get_objectives(transfer_trajectory_object)
            elif post_processing == True:
                self.transfer_trajectory_object = transfer_trajectory_object
                return


        except RuntimeError as e:
            mass_penalty = 0
            negative_distance_penalty = 0
            if e == 'Error with validity of trajectory: the delivery mass is negative.':
                print(e)
                mass_penalty = 10**16
            elif e == 'Error when computing radial distance in hodographic shaping: computed distance is negative.':
                print(e)
                negative_distance_penalty = 10**16
            else:
                print('Unspecified error : ', e)
                other_penalty = 10**16

            if len(self.objectives) == 1:
                objective = [mass_penalty + negative_distance_penalty + other_penalty]
            else:
                objective = [mass_penalty + negative_distance_penalty + other_penalty for _ in range(2)]
                print(objective)

        return objective


class MGALowThrustTrajectoryOptimizationProblemDSM(MGALowThrustTrajectoryOptimizationProblem):
    def __init__(self,
                    transfer_body_order,
                    no_of_free_parameters = 0,
                    bounds = None, 
                    depart_semi_major_axis=np.inf,
                    depart_eccentricity=0, 
                    target_semi_major_axis=np.inf, 
                    target_eccentricity=0,
                    swingby_altitude=200000000, #2e5 km
                    departure_velocity=2000, 
                    arrival_velocity=0,
                    Isp=3000,
                    m0=1000,
                    no_of_points=500,
                    manual_base_functions=False,
                    dynamic_shaping_functions=False,
                    dynamic_bounds=False,
                    manual_tof_bounds=None,
                    objectives=['dv'],
                    zero_revs=False):

        super().__init__(transfer_body_order=transfer_body_order,
                         no_of_free_parameters=no_of_free_parameters, bounds=bounds,
                         depart_semi_major_axis=depart_semi_major_axis,
                         depart_eccentricity=depart_eccentricity,
                         target_semi_major_axis=target_semi_major_axis,
                         target_eccentricity=target_eccentricity, swingby_altitude=swingby_altitude,
                         departure_velocity=departure_velocity, arrival_velocity=arrival_velocity,
                         Isp=Isp, m0=m0, no_of_points=no_of_points,
                         manual_base_functions=manual_base_functions,
                         dynamic_shaping_functions=dynamic_shaping_functions,
                         dynamic_bounds=dynamic_bounds, manual_tof_bounds=manual_tof_bounds,
                         objectives=objectives, zero_revs=zero_revs)

    def get_bounds(self):
        departure_date_lb = self.mjd2000_to_seconds(self.bounds[0][0])
        departure_date_ub = self.mjd2000_to_seconds(self.bounds[1][0])

        departure_velocity_lb = self.bounds[0][1]
        departure_velocity_ub = self.bounds[1][1]

        arrival_velocity_lb = self.bounds[0][2]
        arrival_velocity_ub = self.bounds[1][2]

        time_of_flight_lb = self.mjd2000_to_seconds(self.bounds[0][3])
        time_of_flight_ub = self.mjd2000_to_seconds(self.bounds[1][3])

        incoming_velocity_lb = self.bounds[0][4]
        incoming_velocity_ub = self.bounds[1][4]

        # swingby_altitude_lb = self.swingby_altitude_to_bound(self.bounds[0][5])
        # swingby_altitude_ub = self.swingby_altitude_to_bound(self.bounds[1][5])
        swingby_altitude_lb = self.bounds[0][5]
        swingby_altitude_ub = self.bounds[1][5]

        dsm_dv_lb = self.bounds[0][6]
        dsm_dv_ub = self.bounds[1][6]

        free_coefficients_lb = self.bounds[0][7]
        free_coefficients_ub = self.bounds[1][7]

        number_of_revolutions_lb = self.bounds[0][8]
        number_of_revolutions_ub = self.bounds[1][8]

        lower_bounds = [departure_date_lb] # departure date
        upper_bounds = [departure_date_ub] # departure date
        lower_bounds.append(departure_velocity_lb) # departure velocity # FIXED
        upper_bounds.append(departure_velocity_ub) # departure velocity
        lower_bounds.append(arrival_velocity_lb) # departure velocity # FIXED
        upper_bounds.append(arrival_velocity_ub) # departure velocity

        for leg in range(self.no_of_legs): # time of flight
            if self.manual_tof_bounds != None:
                lower_bounds.append(self.manual_tof_bounds[0][leg] * constants.JULIAN_DAY)
                upper_bounds.append(self.manual_tof_bounds[1][leg] * constants.JULIAN_DAY)
            elif self.dynamic_bounds:
                current_time_of_flight_bounds = self.get_tof_bound(leg, (time_of_flight_lb,
                                                                         time_of_flight_ub))
                lower_bounds.append(current_time_of_flight_bounds[0])
                upper_bounds.append(current_time_of_flight_bounds[1])
            else:
                lower_bounds.append(time_of_flight_lb)
                upper_bounds.append(time_of_flight_ub)
        for _ in range(self.no_of_gas):
            lower_bounds.append(incoming_velocity_lb)
            upper_bounds.append(incoming_velocity_ub)
        for _ in range(self.no_of_gas):
            lower_bounds.append(swingby_altitude_lb)
            upper_bounds.append(swingby_altitude_ub)
        for _ in range(self.no_of_gas):
            lower_bounds.append(dsm_dv_lb)
            upper_bounds.append(dsm_dv_ub)
        for _ in range(self.total_no_of_free_coefficients): # free coefficients
            lower_bounds.append(free_coefficients_lb)
            upper_bounds.append(free_coefficients_ub)
        for _ in range(self.no_of_legs): # number of revolutions
            lower_bounds.append(number_of_revolutions_lb)
            upper_bounds.append(number_of_revolutions_ub)

        return (lower_bounds, upper_bounds)

    def fitness(self, 
                design_parameter_vector : list, 
                bodies = util.create_modified_system_of_bodies(ephemeris_type='JPL'),
                # bodies = environment_setup.create_simplified_system_of_bodies(),
                post_processing=False):

        """
        Assuming no_of_gas == 2 & #fc == 2
        0 - departure_date
        1 - departure velocity
        2 - arrival velocity
        3..6 - time of flights
        6..8 - incoming velocities
        8..10 - swingby periapses
        10..12 - dsm delta v 
        12..30 - free_coefficients
        30..33 - number of revolutions
        """
        # print("Design Parameters:", design_parameter_vector, "\n")
        self.design_parameter_vector = design_parameter_vector

        # parameters
        central_body = 'Sun'

        #depart and target elements
        # departure_elements = (self.depart_semi_major_axis, self.depart_eccentricity) 
        # target_elements = (self.target_semi_major_axis, self.target_eccentricity) 

        # indexes
        time_of_flight_index = 3 + self.no_of_legs
        incoming_velocity_index = time_of_flight_index + self.no_of_gas
        swingby_altitude_index = incoming_velocity_index + self.no_of_gas
        dsm_dv_index = swingby_altitude_index + self.no_of_gas
        free_coefficient_index = dsm_dv_index + self.total_no_of_free_coefficients
        revolution_index = free_coefficient_index + self.no_of_legs

        ### CONTINUOUS PART ###
        departure_date = np.array(design_parameter_vector[0]) # departure date
        departure_velocity = np.array(design_parameter_vector[1]) # departure velocity
        arrival_velocity = np.array(design_parameter_vector[2]) # arrival velocity
        time_of_flights = np.array(design_parameter_vector[3:time_of_flight_index]) # time of flight
        incoming_velocities = \
        np.array(design_parameter_vector[time_of_flight_index:incoming_velocity_index]) # incoming velocities
        swingby_altitudes = np.array([x for x in
                                      design_parameter_vector[incoming_velocity_index:swingby_altitude_index]]) # swingby_altitudes
        dsm_deltav = np.array(design_parameter_vector[swingby_altitude_index:dsm_dv_index]) # dsm_dv
        free_coefficients = np.array(design_parameter_vector[dsm_dv_index:free_coefficient_index]) # hodographic shaping free coefficients

        ### INTEGER PART ###
        if self.zero_revs:
            number_of_revolutions = \
            np.array([0 for _ in design_parameter_vector[free_coefficient_index:revolution_index]])
        else:
            number_of_revolutions = np.array([int(x) for x in
                      design_parameter_vector[free_coefficient_index:revolution_index]]) # number of revolutions

        transfer_trajectory_object = util.get_low_thrust_transfer_object(self.transfer_body_order,
                                                            time_of_flights,
                                                            bodies,
                                                            central_body,
                                                            no_of_free_parameters=self.no_of_free_parameters,
                                                            manual_base_functions=self.manual_base_functions,
                                                            number_of_revolutions=number_of_revolutions,
                                                            dynamic_shaping_functions=self.dynamic_shaping_functions)

        planetary_radii_sequence = np.zeros(self.no_of_gas)
        for i, body in enumerate(self.transfer_body_order[1:-1]):
            planetary_radii_sequence[i] = self.planetary_radii[body]

        swingby_periapses = np.array([planetary_radii_sequence[i] + swingby_altitudes[i] for i in
            range(self.no_of_gas)]) 
        # incoming_velocity= np.array([incoming_velocities[i] for i in range(self.no_of_gas)])
        # dsm_deltav= np.array([dsm_deltav[i] for i in range(self.no_of_gas)])

        # node times
        self.node_times = util.get_node_times(departure_date, time_of_flights)
        # print(node_times)

        # leg free parameters 
        leg_free_parameters = np.concatenate(np.array([np.append(number_of_revolutions[i],
            free_coefficients[i*(self.no_of_free_parameters*3):(i+1)*(self.no_of_free_parameters*3)]) for i
            in range(self.no_of_legs)])).reshape((self.no_of_legs, 1 + 3*
                self.no_of_free_parameters)) # added reshape

        # node free parameters
        node_free_parameters = util.get_node_free_parameters(self.transfer_body_order,
                                                             swingby_periapses, 
                                                             incoming_velocities, 
                                                             dsm_deltav=dsm_deltav, 
                                                             departure_velocity=departure_velocity,
                                                             arrival_velocity=arrival_velocity)

        try:
            transfer_trajectory_object.evaluate(self.node_times, leg_free_parameters, node_free_parameters)
            # self.delivery_mass_constraint_check(transfer_trajectory_object, self.Isp, self.m0, self.no_of_points)

            # self.transfer_trajectory_object = transfer_trajectory_object
            #
            # if post_processing == False:
            #     objective = self.get_objectives()
            # else:
            #     return
            #
            if post_processing == False:
                objective = self.get_objectives(transfer_trajectory_object)
            elif post_processing == True:
                self.transfer_trajectory_object = transfer_trajectory_object
                return


        except RuntimeError as e:
            mass_penalty = 0
            negative_distance_penalty = 0
            if e == 'Error with validity of trajectory: the delivery mass is negative.':
                print(e)
                mass_penalty = 10**16
            elif e == 'Error when computing radial distance in hodographic shaping: computed distance is negative.':
                print(e)
                negative_distance_penalty = 10**16
            else:
                print('Unspecified error : ', e)
                other_penalty = 10**16

            if len(self.objectives) == 1:
                objective = [mass_penalty + negative_distance_penalty + other_penalty]
            else:
                objective = [mass_penalty + negative_distance_penalty + other_penalty for _ in range(2)]
                print(objective)

        return objective

class MGALowThrustTrajectoryOptimizationProblemOOA(MGALowThrustTrajectoryOptimizationProblem):
    def __init__(self,
                    transfer_body_order,
                    no_of_free_parameters = 0,
                    bounds = None, 
                    depart_semi_major_axis=np.inf,
                    depart_eccentricity=0, 
                    target_semi_major_axis=np.inf, 
                    target_eccentricity=0,
                    swingby_altitude=200000000, #2e5 km
                    departure_velocity=2000, 
                    arrival_velocity=0,
                    Isp=3000,
                    m0=1000,
                    no_of_points=500,
                    manual_base_functions=False,
                    dynamic_shaping_functions=False,
                    dynamic_bounds=False,
                    manual_tof_bounds=None,
                    objectives=['dv'],
                    zero_revs=False):

        super().__init__(transfer_body_order=transfer_body_order,
                         no_of_free_parameters=no_of_free_parameters, bounds=bounds,
                         depart_semi_major_axis=depart_semi_major_axis,
                         depart_eccentricity=depart_eccentricity,
                         target_semi_major_axis=target_semi_major_axis,
                         target_eccentricity=target_eccentricity, swingby_altitude=swingby_altitude,
                         departure_velocity=departure_velocity, arrival_velocity=arrival_velocity,
                         Isp=Isp, m0=m0, no_of_points=no_of_points,
                         manual_base_functions=manual_base_functions,
                         dynamic_shaping_functions=dynamic_shaping_functions,
                         dynamic_bounds=dynamic_bounds, manual_tof_bounds=manual_tof_bounds,
                         objectives=objectives, zero_revs=zero_revs)

    def get_bounds(self):
        departure_date_lb = self.mjd2000_to_seconds(self.bounds[0][0])
        departure_date_ub = self.mjd2000_to_seconds(self.bounds[1][0])

        departure_velocity_lb = self.bounds[0][1]
        departure_velocity_ub = self.bounds[1][1]

        arrival_velocity_lb = self.bounds[0][2]
        arrival_velocity_ub = self.bounds[1][2]

        time_of_flight_lb = self.mjd2000_to_seconds(self.bounds[0][3])
        time_of_flight_ub = self.mjd2000_to_seconds(self.bounds[1][3])

        incoming_velocity_lb = self.bounds[0][4]
        incoming_velocity_ub = self.bounds[1][4]

        swingby_altitude_lb = self.bounds[0][5]
        swingby_altitude_ub = self.bounds[1][5]

        orbit_ori_angle_lb = self.bounds[0][6]
        orbit_ori_angle_ub = self.bounds[1][6]

        free_coefficients_lb = self.bounds[0][7]
        free_coefficients_ub = self.bounds[1][7]

        number_of_revolutions_lb = self.bounds[0][8]
        number_of_revolutions_ub = self.bounds[1][8]

        lower_bounds = [departure_date_lb] # departure date
        upper_bounds = [departure_date_ub] # departure date
        lower_bounds.append(departure_velocity_lb) # departure velocity # FIXED
        upper_bounds.append(departure_velocity_ub) # departure velocity
        lower_bounds.append(arrival_velocity_lb) # departure velocity # FIXED
        upper_bounds.append(arrival_velocity_ub) # departure velocity

        for leg in range(self.no_of_legs): # time of flight
            if self.manual_tof_bounds != None:
                lower_bounds.append(self.manual_tof_bounds[0][leg] * constants.JULIAN_DAY)
                upper_bounds.append(self.manual_tof_bounds[1][leg] * constants.JULIAN_DAY)
            elif self.dynamic_bounds:
                current_time_of_flight_bounds = self.get_tof_bound(leg, (time_of_flight_lb,
                                                                         time_of_flight_ub))
                lower_bounds.append(current_time_of_flight_bounds[0])
                upper_bounds.append(current_time_of_flight_bounds[1])
            else:
                lower_bounds.append(time_of_flight_lb)
                upper_bounds.append(time_of_flight_ub)
        for _ in range(self.no_of_gas):
            lower_bounds.append(incoming_velocity_lb)
            upper_bounds.append(incoming_velocity_ub)
        for _ in range(self.no_of_gas):
            lower_bounds.append(swingby_altitude_lb)
            upper_bounds.append(swingby_altitude_ub)
        for _ in range(self.no_of_gas):
            lower_bounds.append(orbit_ori_angle_lb)
            upper_bounds.append(orbit_ori_angle_ub)
        for _ in range(self.total_no_of_free_coefficients): # free coefficients
            lower_bounds.append(free_coefficients_lb)
            upper_bounds.append(free_coefficients_ub)
        for _ in range(self.no_of_legs): # number of revolutions
            lower_bounds.append(number_of_revolutions_lb)
            upper_bounds.append(number_of_revolutions_ub)

        return (lower_bounds, upper_bounds)

    def fitness(self, 
                design_parameter_vector : list, 
                bodies = util.create_modified_system_of_bodies(ephemeris_type='JPL'),
                # bodies = environment_setup.create_simplified_system_of_bodies(),
                post_processing=False):

        """
        Assuming no_of_gas == 2 & #fc == 2
        0 - departure_date
        1 - departure velocity
        2 - arrival velocity
        3..6 - time of flights
        6..8 - incoming velocities
        8..10 - swingby periapses
        10..12 - orbit orientation angle
        12..30 - free_coefficients
        30..33 - number of revolutions
        """
        # print("Design Parameters:", design_parameter_vector, "\n")
        self.design_parameter_vector = design_parameter_vector

        # parameters
        central_body = 'Sun'

        #depart and target elements
        # departure_elements = (self.depart_semi_major_axis, self.depart_eccentricity) 
        # target_elements = (self.target_semi_major_axis, self.target_eccentricity) 

        # indexes
        time_of_flight_index = 3 + self.no_of_legs
        incoming_velocity_index = time_of_flight_index + self.no_of_gas
        swingby_altitude_index = incoming_velocity_index + self.no_of_gas
        orbit_ori_angle_index = swingby_altitude_index + self.no_of_gas
        free_coefficient_index = orbit_ori_angle_index + self.total_no_of_free_coefficients
        revolution_index = free_coefficient_index + self.no_of_legs

        ### CONTINUOUS PART ###
        departure_date = np.array(design_parameter_vector[0]) # departure date
        departure_velocity = np.array(design_parameter_vector[1]) # departure velocity
        arrival_velocity = np.array(design_parameter_vector[2]) # arrival velocity
        time_of_flights = np.array(design_parameter_vector[3:time_of_flight_index]) # time of flight
        incoming_velocities = \
        np.array(design_parameter_vector[time_of_flight_index:incoming_velocity_index]) # incoming velocities
        swingby_altitudes = np.array([x for x in
                  design_parameter_vector[incoming_velocity_index:swingby_altitude_index]]) # swingby_altitudes
        orbit_ori_angles = \
        np.array(design_parameter_vector[swingby_altitude_index:orbit_ori_angle_index]) # orbit_ori_angle
        free_coefficients = \
        np.array(design_parameter_vector[orbit_ori_angle_index:free_coefficient_index]) # hodographic shaping free coefficients

        ### INTEGER PART ###
        if self.zero_revs:
            number_of_revolutions = \
            np.array([0 for _ in design_parameter_vector[free_coefficient_index:revolution_index]])
        else:
            number_of_revolutions = np.array([int(x) for x in
                      design_parameter_vector[free_coefficient_index:revolution_index]]) # number of revolutions

        transfer_trajectory_object = util.get_low_thrust_transfer_object(self.transfer_body_order,
                                                            time_of_flights,
                                                            bodies,
                                                            central_body,
                                                            no_of_free_parameters=self.no_of_free_parameters,
                                                            manual_base_functions=self.manual_base_functions,
                                                            number_of_revolutions=number_of_revolutions,
                                                            dynamic_shaping_functions=self.dynamic_shaping_functions)

        planetary_radii_sequence = np.zeros(self.no_of_gas)
        for i, body in enumerate(self.transfer_body_order[1:-1]):
            planetary_radii_sequence[i] = self.planetary_radii[body]

        swingby_periapses = np.array([planetary_radii_sequence[i] + swingby_altitudes[i] for i in
            range(self.no_of_gas)]) 
        # incoming_velocity= np.array([incoming_velocities[i] for i in range(self.no_of_gas)])
        # orbit_ori_angle= np.array([orbit_ori_angles[i] for i in range(self.no_of_gas)])

        # node times
        self.node_times = util.get_node_times(departure_date, time_of_flights)
        # print(node_times)

        # leg free parameters 
        leg_free_parameters = np.concatenate(np.array([np.append(number_of_revolutions[i],
            free_coefficients[i*(self.no_of_free_parameters*3):(i+1)*(self.no_of_free_parameters*3)]) for i
            in range(self.no_of_legs)])).reshape((self.no_of_legs, 1 + 3*
                self.no_of_free_parameters)) # added reshape

        # node free parameters
        node_free_parameters = util.get_node_free_parameters(self.transfer_body_order,
                                                             swingby_periapses,
                                                             incoming_velocities,
                                                             orbit_ori_angle=orbit_ori_angles,
                                                             departure_velocity=departure_velocity,
                                                             arrival_velocity=arrival_velocity)

        try:
            transfer_trajectory_object.evaluate(self.node_times, leg_free_parameters, node_free_parameters)
            # self.delivery_mass_constraint_check(transfer_trajectory_object, self.Isp, self.m0, self.no_of_points)

            # self.transfer_trajectory_object = transfer_trajectory_object
            #
            # if post_processing == False:
            #     objective = self.get_objectives()
            # else:
            #     return
            #
            if post_processing == False:
                objective = self.get_objectives(transfer_trajectory_object)
            elif post_processing == True:
                self.transfer_trajectory_object = transfer_trajectory_object
                return


        except RuntimeError as e:
            mass_penalty = 0
            negative_distance_penalty = 0
            if e == 'Error with validity of trajectory: the delivery mass is negative.':
                print(e)
                mass_penalty = 10**16
            elif e == 'Error when computing radial distance in hodographic shaping: computed distance is negative.':
                print(e)
                negative_distance_penalty = 10**16
            else:
                print('Unspecified error : ', e)
                other_penalty = 10**16

            if len(self.objectives) == 1:
                objective = [mass_penalty + negative_distance_penalty + other_penalty]
            else:
                objective = [mass_penalty + negative_distance_penalty + other_penalty for _ in range(2)]
                print(objective)

        return objective


class MGALowThrustTrajectoryOptimizationProblemOOADAAA(MGALowThrustTrajectoryOptimizationProblem):
    def __init__(self,
                    transfer_body_order,
                    no_of_free_parameters = 0,
                    bounds = None, 
                    depart_semi_major_axis=np.inf,
                    depart_eccentricity=0, 
                    target_semi_major_axis=np.inf, 
                    target_eccentricity=0,
                    swingby_altitude=200000000, #2e5 km
                    departure_velocity=2000, 
                    arrival_velocity=0,
                    Isp=3000,
                    m0=1000,
                    no_of_points=500,
                    manual_base_functions=False,
                    dynamic_shaping_functions=False,
                    dynamic_bounds=False,
                    manual_tof_bounds=None,
                    objectives=['dv'],
                    zero_revs=False):

        super().__init__(transfer_body_order=transfer_body_order,
                         no_of_free_parameters=no_of_free_parameters, bounds=bounds,
                         depart_semi_major_axis=depart_semi_major_axis,
                         depart_eccentricity=depart_eccentricity,
                         target_semi_major_axis=target_semi_major_axis,
                         target_eccentricity=target_eccentricity, swingby_altitude=swingby_altitude,
                         departure_velocity=departure_velocity, arrival_velocity=arrival_velocity,
                         Isp=Isp, m0=m0, no_of_points=no_of_points,
                         manual_base_functions=manual_base_functions,
                         dynamic_shaping_functions=dynamic_shaping_functions,
                         dynamic_bounds=dynamic_bounds, manual_tof_bounds=manual_tof_bounds,
                         objectives=objectives, zero_revs=zero_revs)

    def get_bounds(self):
        departure_date_lb = self.mjd2000_to_seconds(self.bounds[0][0])
        departure_date_ub = self.mjd2000_to_seconds(self.bounds[1][0])

        departure_velocity_lb = self.bounds[0][1]
        departure_velocity_ub = self.bounds[1][1]

        departure_inplane_angle_lb = self.bounds[0][2]
        departure_inplane_angle_ub = self.bounds[1][2]

        departure_outofplane_angle_lb = self.bounds[0][3]
        departure_outofplane_angle_ub = self.bounds[1][3]

        arrival_velocity_lb = self.bounds[0][4]
        arrival_velocity_ub = self.bounds[1][4]

        arrival_inplane_angle_lb = self.bounds[0][5]
        arrival_inplane_angle_ub = self.bounds[1][5]

        arrival_outofplane_angle_lb = self.bounds[0][6]
        arrival_outofplane_angle_ub = self.bounds[1][6]

        time_of_flight_lb = self.mjd2000_to_seconds(self.bounds[0][7])
        time_of_flight_ub = self.mjd2000_to_seconds(self.bounds[1][7])

        incoming_velocity_lb = self.bounds[0][8]
        incoming_velocity_ub = self.bounds[1][8]

        swingby_altitude_lb = self.bounds[0][9]
        swingby_altitude_ub = self.bounds[1][9]

        orbit_ori_angle_lb = self.bounds[0][10]
        orbit_ori_angle_ub = self.bounds[1][10]

        free_coefficients_lb = self.bounds[0][11]
        free_coefficients_ub = self.bounds[1][11]

        number_of_revolutions_lb = self.bounds[0][12]
        number_of_revolutions_ub = self.bounds[1][12]

        lower_bounds = [departure_date_lb] # departure date
        upper_bounds = [departure_date_ub] # departure date

        lower_bounds.append(departure_velocity_lb) 
        upper_bounds.append(departure_velocity_ub)
        lower_bounds.append(departure_inplane_angle_lb) 
        upper_bounds.append(departure_inplane_angle_ub)
        lower_bounds.append(departure_outofplane_angle_lb) 
        upper_bounds.append(departure_outofplane_angle_ub) 

        lower_bounds.append(arrival_velocity_lb) 
        upper_bounds.append(arrival_velocity_ub)
        lower_bounds.append(arrival_inplane_angle_lb) 
        upper_bounds.append(arrival_inplane_angle_ub)
        lower_bounds.append(arrival_outofplane_angle_lb) 
        upper_bounds.append(arrival_outofplane_angle_ub) 

        for leg in range(self.no_of_legs): # time of flight
            if self.manual_tof_bounds != None:
                lower_bounds.append(self.manual_tof_bounds[0][leg] * constants.JULIAN_DAY)
                upper_bounds.append(self.manual_tof_bounds[1][leg] * constants.JULIAN_DAY)
            elif self.dynamic_bounds:
                current_time_of_flight_bounds = self.get_tof_bound(leg, (time_of_flight_lb,
                                                                         time_of_flight_ub))
                lower_bounds.append(current_time_of_flight_bounds[0])
                upper_bounds.append(current_time_of_flight_bounds[1])
            else:
                lower_bounds.append(time_of_flight_lb)
                upper_bounds.append(time_of_flight_ub)
        for _ in range(self.no_of_gas):
            lower_bounds.append(incoming_velocity_lb)
            upper_bounds.append(incoming_velocity_ub)
        for _ in range(self.no_of_gas):
            lower_bounds.append(swingby_altitude_lb)
            upper_bounds.append(swingby_altitude_ub)
        for _ in range(self.no_of_gas):
            lower_bounds.append(orbit_ori_angle_lb)
            upper_bounds.append(orbit_ori_angle_ub)
        for _ in range(self.total_no_of_free_coefficients): # free coefficients
            lower_bounds.append(free_coefficients_lb)
            upper_bounds.append(free_coefficients_ub)
        for _ in range(self.no_of_legs): # number of revolutions
            lower_bounds.append(number_of_revolutions_lb)
            upper_bounds.append(number_of_revolutions_ub)

        return (lower_bounds, upper_bounds)

    def fitness(self, 
                design_parameter_vector : list, 
                bodies = util.create_modified_system_of_bodies(ephemeris_type='JPL'),
                # bodies = environment_setup.create_simplified_system_of_bodies(),
                post_processing=False):

        """
        Assuming no_of_gas == 2 & #fc == 2
        0 - departure_date
        1 - departure velocity
        2 - dep inplane angle 
        3 - dep outofplane angle 
        4 - arrival velocity
        5 - arr inplane angle
        6 - arr outofplane angle
        7..10 - time of flights
        10..12 - incoming velocities
        12..14 - swingby periapses
        14..16 - orbit orientation angle
        16..34 - free_coefficients
        34..37 - number of revolutions
        """
        # print("Design Parameters:", design_parameter_vector, "\n")
        self.design_parameter_vector = design_parameter_vector

        # parameters
        central_body = 'Sun'

        #depart and target elements
        # departure_elements = (self.depart_semi_major_axis, self.depart_eccentricity) 
        # target_elements = (self.target_semi_major_axis, self.target_eccentricity) 

        # indexes
        time_of_flight_index = 7 + self.no_of_legs
        incoming_velocity_index = time_of_flight_index + self.no_of_gas
        swingby_altitude_index = incoming_velocity_index + self.no_of_gas
        orbit_ori_angle_index = swingby_altitude_index + self.no_of_gas
        free_coefficient_index = orbit_ori_angle_index + self.total_no_of_free_coefficients
        revolution_index = free_coefficient_index + self.no_of_legs

        ### CONTINUOUS PART ###
        departure_date = np.array(design_parameter_vector[0]) # departure date
        departure_velocity = np.array(design_parameter_vector[1]) # departure velocity
        departure_inplane_angle = np.array(design_parameter_vector[2]) 
        departure_outofplane_angle = np.array(design_parameter_vector[3]) 
        arrival_velocity = np.array(design_parameter_vector[4]) # arrival velocity
        arrival_inplane_angle = np.array(design_parameter_vector[5]) 
        arrival_outofplane_angle = np.array(design_parameter_vector[6]) 

        time_of_flights = np.array(design_parameter_vector[7:time_of_flight_index]) # time of flight
        incoming_velocities = \
        np.array(design_parameter_vector[time_of_flight_index:incoming_velocity_index]) # incoming velocities
        swingby_altitudes = np.array([x for x in
                  design_parameter_vector[incoming_velocity_index:swingby_altitude_index]]) # swingby_altitudes
        orbit_ori_angles = \
        np.array(design_parameter_vector[swingby_altitude_index:orbit_ori_angle_index]) # orbit_ori_angle
        free_coefficients = \
        np.array(design_parameter_vector[orbit_ori_angle_index:free_coefficient_index]) # hodographic shaping free coefficients

        ### INTEGER PART ###
        if self.zero_revs:
            number_of_revolutions = \
            np.array([0 for _ in design_parameter_vector[free_coefficient_index:revolution_index]])
        else:
            number_of_revolutions = np.array([int(x) for x in
                      design_parameter_vector[free_coefficient_index:revolution_index]]) # number of revolutions

        transfer_trajectory_object = util.get_low_thrust_transfer_object(self.transfer_body_order,
                                                            time_of_flights,
                                                            bodies,
                                                            central_body,
                                                            no_of_free_parameters=self.no_of_free_parameters,
                                                            manual_base_functions=self.manual_base_functions,
                                                            number_of_revolutions=number_of_revolutions,
                                                            dynamic_shaping_functions=self.dynamic_shaping_functions)

        planetary_radii_sequence = np.zeros(self.no_of_gas)
        for i, body in enumerate(self.transfer_body_order[1:-1]):
            planetary_radii_sequence[i] = self.planetary_radii[body]

        swingby_periapses = np.array([planetary_radii_sequence[i] + swingby_altitudes[i] for i in
            range(self.no_of_gas)]) 

        # node times
        self.node_times = util.get_node_times(departure_date, time_of_flights)
        # print(node_times)

        # leg free parameters 
        leg_free_parameters = np.concatenate(np.array([np.append(number_of_revolutions[i],
            free_coefficients[i*(self.no_of_free_parameters*3):(i+1)*(self.no_of_free_parameters*3)]) for i
            in range(self.no_of_legs)])).reshape((self.no_of_legs, 1 + 3*
                self.no_of_free_parameters)) # added reshape

        # node free parameters
        node_free_parameters = util.get_node_free_parameters(self.transfer_body_order,
                                                     swingby_periapses,
                                                     incoming_velocities,
                                                     orbit_ori_angles=orbit_ori_angles,
                                                     departure_velocity=departure_velocity,
                                                     departure_inplane_angle=departure_inplane_angle,
                                                     departure_outofplane_angle=departure_outofplane_angle,
                                                     arrival_velocity=arrival_velocity,
                                                     arrival_inplane_angle=arrival_inplane_angle,
                                                     arrival_outofplane_angle=arrival_outofplane_angle)

        try:
            transfer_trajectory_object.evaluate(self.node_times, leg_free_parameters, node_free_parameters)
            # self.delivery_mass_constraint_check(transfer_trajectory_object, self.Isp, self.m0, self.no_of_points)

            # self.transfer_trajectory_object = transfer_trajectory_object
            #
            # if post_processing == False:
            #     objective = self.get_objectives()
            # else:
            #     return
            #
            if post_processing == False:
                objective = self.get_objectives(transfer_trajectory_object)
            elif post_processing == True:
                self.transfer_trajectory_object = transfer_trajectory_object
                return


        except RuntimeError as e:
            mass_penalty = 0
            negative_distance_penalty = 0
            if e == 'Error with validity of trajectory: the delivery mass is negative.':
                print(e)
                mass_penalty = 10**16
            elif e == 'Error when computing radial distance in hodographic shaping: computed distance is negative.':
                print(e)
                negative_distance_penalty = 10**16
            else:
                print('Unspecified error : ', e)
                other_penalty = 10**16

            if len(self.objectives) == 1:
                objective = [mass_penalty + negative_distance_penalty + other_penalty]
            else:
                objective = [mass_penalty + negative_distance_penalty + other_penalty for _ in range(2)]
                print(objective)

        return objective

class MGALowThrustTrajectoryOptimizationProblemAllAngles(MGALowThrustTrajectoryOptimizationProblem):
    def __init__(self,
                    transfer_body_order,
                    no_of_free_parameters = 0,
                    bounds = None, 
                    depart_semi_major_axis=np.inf,
                    depart_eccentricity=0, 
                    target_semi_major_axis=np.inf, 
                    target_eccentricity=0,
                    swingby_altitude=200000000, #2e5 km
                    departure_velocity=2000, 
                    arrival_velocity=0,
                    Isp=3000,
                    m0=1000,
                    no_of_points=500,
                    manual_base_functions=False,
                    dynamic_bounds = {'time_of_flight' : False,
                                      'orbit_ori_angle' : False,
                                      'swingby_outofplane' : False,
                                      'swingby_inplane' : False,
                                      'shaping_function' : False},
                    manual_tof_bounds=None,
                    objectives=['dv'],
                    zero_revs=False):

        super().__init__(transfer_body_order=transfer_body_order,
                         no_of_free_parameters=no_of_free_parameters, bounds=bounds,
                         depart_semi_major_axis=depart_semi_major_axis,
                         depart_eccentricity=depart_eccentricity,
                         target_semi_major_axis=target_semi_major_axis,
                         target_eccentricity=target_eccentricity, swingby_altitude=swingby_altitude,
                         departure_velocity=departure_velocity, arrival_velocity=arrival_velocity,
                         Isp=Isp, m0=m0, no_of_points=no_of_points,
                         manual_base_functions=manual_base_functions,
                         dynamic_bounds=dynamic_bounds, manual_tof_bounds=manual_tof_bounds,
                         objectives=objectives, zero_revs=zero_revs)

        self.flyby_directions = util.get_flyby_directions(self.transfer_body_order)
        self.bound_names= ['Departure date [mjd2000]', 'Departure velocity [m/s]', 'Departure in-plane angle [rad]', 
                      'Departure out-of-plane angle [rad]', 'Arrival velocity [m/s]', 'Arrival in-plane angle [rad]', 
                      'Arrival out-of-plane angle [rad]', 'Time of Flight [s]', 'Incoming velocity [m/s]', 
                      'Swingby periapsis [m]', 'Orbit orientation angle [rad]', 'Swingby in-plane Angle [rad]', 
                      'Swingby out-of-plane angle [rad]', 'Free coefficient [-]', 'Number of revolutions [-]']


    def get_ooa_angle_from_quantity(self, ooa_quantity):
        ooa_angle = ooa_quantity.copy()
        if ooa_quantity > np.pi:
            ooa_angle = ooa_angle * (1 / 3) + (np.pi / 2)
            try:
                assert 5 * np.pi / 6 < ooa_angle < 7 * np.pi / 6
            except AssertionError:
                print(ooa_quantity, ooa_angle)
        else:
            ooa_angle = ooa_angle * (1 / 3) - (np.pi / 6)
            try:
                assert -np.pi / 6 < ooa_angle < np.pi / 6
            except AssertionError:
                print(ooa_quantity, ooa_angle)

        return ooa_angle

    def get_ooa_quantity_from_angle(self):
        pass

    def get_inplane_bounds(self, i):

        if self.flyby_directions[i] == 'inner':
            lb =  0; ub = 2 * np.pi
        elif self.flyby_directions[i] == 'outer':
            lb = np.pi; ub = 2 * np.pi
        elif self.flyby_directions[i] == 'same':
            lb =  0; ub = 2 * np.pi
        else:
            raise RuntimeError(f"The type of flyby direction, {self.flyby_directions[i]}, is not defined properly")

        return lb, ub

    def get_outofplane_bounds(self):
        # Assuming Earth-Neptune, Mercury is 7.005 degrees, with 5% margin of safety.
        # Venus is 3.395 degrees, max difference is 10.400 degrees if exactly opposite

        return -10.400 * 1.05 * np.pi / 180, 10.400 * 1.05 * np.pi / 180 # rad


    def get_bounds(self):
        departure_date_lb = self.mjd2000_to_seconds(self.bounds[0][0])
        departure_date_ub = self.mjd2000_to_seconds(self.bounds[1][0])

        departure_velocity_lb = self.bounds[0][1]
        departure_velocity_ub = self.bounds[1][1]

        departure_inplane_angle_lb = self.bounds[0][2]
        departure_inplane_angle_ub = self.bounds[1][2]

        departure_outofplane_angle_lb = self.bounds[0][3]
        departure_outofplane_angle_ub = self.bounds[1][3]

        arrival_velocity_lb = self.bounds[0][4]
        arrival_velocity_ub = self.bounds[1][4]

        arrival_inplane_angle_lb = self.bounds[0][5]
        arrival_inplane_angle_ub = self.bounds[1][5]

        arrival_outofplane_angle_lb = self.bounds[0][6]
        arrival_outofplane_angle_ub = self.bounds[1][6]

        time_of_flight_lb = self.mjd2000_to_seconds(self.bounds[0][7])
        time_of_flight_ub = self.mjd2000_to_seconds(self.bounds[1][7])

        incoming_velocity_lb = self.bounds[0][8]
        incoming_velocity_ub = self.bounds[1][8]

        swingby_altitude_lb = self.bounds[0][9]
        swingby_altitude_ub = self.bounds[1][9]

        orbit_ori_angle_lb = self.bounds[0][10]
        orbit_ori_angle_ub = self.bounds[1][10]

        swingby_inplane_angle_lb = self.bounds[0][11]
        swingby_inplane_angle_ub = self.bounds[1][11]

        if self.dynamic_bounds['swingby_outofplane']:
            self.bounds[0][12], self.bounds[1][12] = self.get_outofplane_bounds()
            # print('swingby_outofplane', swingby_outofplane_angle_lb, swingby_outofplane_angle_ub)
        swingby_outofplane_angle_lb = self.bounds[0][12]
        swingby_outofplane_angle_ub = self.bounds[1][12]


        free_coefficients_lb = self.bounds[0][13]
        free_coefficients_ub = self.bounds[1][13]

        number_of_revolutions_lb = self.bounds[0][14]
        number_of_revolutions_ub = self.bounds[1][14]

        lower_bounds = [departure_date_lb] # departure date
        upper_bounds = [departure_date_ub] # departure date

        lower_bounds.append(departure_velocity_lb) 
        upper_bounds.append(departure_velocity_ub)
        lower_bounds.append(departure_inplane_angle_lb) 
        upper_bounds.append(departure_inplane_angle_ub)
        lower_bounds.append(departure_outofplane_angle_lb) 
        upper_bounds.append(departure_outofplane_angle_ub) 

        lower_bounds.append(arrival_velocity_lb) 
        upper_bounds.append(arrival_velocity_ub)
        lower_bounds.append(arrival_inplane_angle_lb) 
        upper_bounds.append(arrival_inplane_angle_ub)
        lower_bounds.append(arrival_outofplane_angle_lb) 
        upper_bounds.append(arrival_outofplane_angle_ub) 

        for leg in range(self.no_of_legs): # time of flight
            if self.manual_tof_bounds != None:
                lower_bounds.append(self.manual_tof_bounds[0][leg] * constants.JULIAN_DAY)
                upper_bounds.append(self.manual_tof_bounds[1][leg] * constants.JULIAN_DAY)
            elif self.dynamic_bounds['time_of_flight']:
                current_time_of_flight_bounds = self.get_tof_bound(leg, (time_of_flight_lb,
                                                                         time_of_flight_ub))
                lower_bounds.append(current_time_of_flight_bounds[0])
                upper_bounds.append(current_time_of_flight_bounds[1])
                # print('tof', current_time_of_flight_bounds[0], current_time_of_flight_bounds[1])
            else:
                lower_bounds.append(time_of_flight_lb)
                upper_bounds.append(time_of_flight_ub)
        for _ in range(self.no_of_gas):
            lower_bounds.append(incoming_velocity_lb)
            upper_bounds.append(incoming_velocity_ub)
        for _ in range(self.no_of_gas):
            lower_bounds.append(swingby_altitude_lb)
            upper_bounds.append(swingby_altitude_ub)
        for _ in range(self.no_of_gas):
            lower_bounds.append(orbit_ori_angle_lb)
            upper_bounds.append(orbit_ori_angle_ub)
        for ga in range(self.no_of_gas):
            if self.dynamic_bounds['swingby_inplane']:
                lb, ub = self.get_inplane_bounds(ga)
                lower_bounds.append(lb)
                upper_bounds.append(ub)
                # print('swingby_inplane', lb, ub)
            else:
                lower_bounds.append(swingby_inplane_angle_lb)
                upper_bounds.append(swingby_inplane_angle_ub)
        for _ in range(self.no_of_gas):
            lower_bounds.append(swingby_outofplane_angle_lb)
            upper_bounds.append(swingby_outofplane_angle_ub)
        for _ in range(self.total_no_of_free_coefficients): # free coefficients
            lower_bounds.append(free_coefficients_lb)
            upper_bounds.append(free_coefficients_ub)
        for _ in range(self.no_of_legs): # number of revolutions
            lower_bounds.append(number_of_revolutions_lb)
            upper_bounds.append(number_of_revolutions_ub)

        return (lower_bounds, upper_bounds)

    def fitness(self, 
                design_parameter_vector : list, 
                bodies = util.create_modified_system_of_bodies(ephemeris_type='JPL'),
                # bodies = environment_setup.create_simplified_system_of_bodies(),
                post_processing=False):

        """
        Assuming no_of_gas == 2 & #fc == 2
        0 - departure_date
        1 - departure velocity
        2 - dep inplane angle 
        3 - dep outofplane angle 
        4 - arrival velocity
        5 - arr inplane angle
        6 - arr outofplane angle
        7..10 - time of flights
        10..12 - incoming velocities
        12..14 - swingby periapses
        14..16 - orbit orientation angle
        16..18 - swingby inplane angle
        18..20 - swingby outofplane angle
        20..38 - free_coefficients
        38..41 - number of revolutions
        """
        # print("Design Parameters:", design_parameter_vector, "\n")
        self.design_parameter_vector = design_parameter_vector

        # parameters
        central_body = 'Sun'

        #depart and target elements
        # departure_elements = (self.depart_semi_major_axis, self.depart_eccentricity) 
        # target_elements = (self.target_semi_major_axis, self.target_eccentricity) 

        # indexes
        time_of_flight_index = 7 + self.no_of_legs
        incoming_velocity_index = time_of_flight_index + self.no_of_gas
        swingby_altitude_index = incoming_velocity_index + self.no_of_gas
        orbit_ori_angle_index = swingby_altitude_index + self.no_of_gas
        swingby_inplane_angle_index = orbit_ori_angle_index + self.no_of_gas
        swingby_outofplane_angle_index = swingby_inplane_angle_index + self.no_of_gas
        free_coefficient_index = swingby_outofplane_angle_index + self.total_no_of_free_coefficients
        revolution_index = free_coefficient_index + self.no_of_legs

        ### CONTINUOUS PART ###
        departure_date = np.array(design_parameter_vector[0]) # departure date
        departure_velocity = np.array(design_parameter_vector[1]) # departure velocity
        departure_inplane_angle = np.array(design_parameter_vector[2]) 
        departure_outofplane_angle = np.array(design_parameter_vector[3]) 
        arrival_velocity = np.array(design_parameter_vector[4]) # arrival velocity
        arrival_inplane_angle = np.array(design_parameter_vector[5]) 
        arrival_outofplane_angle = np.array(design_parameter_vector[6]) 

        time_of_flights = np.array(design_parameter_vector[7:time_of_flight_index]) # time of flight
        incoming_velocities = \
        np.array(design_parameter_vector[time_of_flight_index:incoming_velocity_index]) # incoming velocities
        swingby_altitudes = np.array([x for x in
                  design_parameter_vector[incoming_velocity_index:swingby_altitude_index]]) # swingby_altitudes

        orbit_ori_angles = \
        np.array(design_parameter_vector[swingby_altitude_index:orbit_ori_angle_index]) # orbit_ori_angle
        # print(f'pre{orbit_ori_angles}')
        if self.dynamic_bounds['orbit_ori_angle']:
            orbit_ori_angles = np.array([self.get_ooa_angle_from_quantity(i) for i in orbit_ori_angles])
            # print('ooa', orbit_ori_angles)


        swingby_inplane_angles = \
        np.array(design_parameter_vector[orbit_ori_angle_index:swingby_inplane_angle_index])
        swingby_outofplane_angles = \
        np.array(design_parameter_vector[swingby_inplane_angle_index:swingby_outofplane_angle_index])
        free_coefficients = \
        np.array(design_parameter_vector[orbit_ori_angle_index:free_coefficient_index]) # hodographic shaping free coefficients

        ### INTEGER PART ###
        if self.zero_revs:
            number_of_revolutions = \
            np.array([0 for _ in design_parameter_vector[free_coefficient_index:revolution_index]])
        else:
            number_of_revolutions = np.array([int(x) for x in
                      design_parameter_vector[free_coefficient_index:revolution_index]]) # number of revolutions

        transfer_trajectory_object = util.get_low_thrust_transfer_object(self.transfer_body_order,
                                                        time_of_flights,
                                                        bodies,
                                                        central_body,
                                                        no_of_free_parameters=self.no_of_free_parameters,
                                                        manual_base_functions=self.manual_base_functions,
                                                        number_of_revolutions=number_of_revolutions,
                                                        dynamic_shaping_functions=self.dynamic_bounds['shaping_function'])

        planetary_radii_sequence = np.zeros(self.no_of_gas)
        for i, body in enumerate(self.transfer_body_order[1:-1]):
            planetary_radii_sequence[i] = self.planetary_radii[body]

        swingby_periapses = np.array([planetary_radii_sequence[i] + swingby_altitudes[i] for i in
            range(self.no_of_gas)]) 
        # incoming_velocity= np.array([incoming_velocities[i] for i in range(self.no_of_gas)])
        # orbit_ori_angle= np.array([orbit_ori_angles[i] for i in range(self.no_of_gas)])
        # swingby_inplane_angles= np.array([swingby_inplane_angles[i] for i in range(self.no_of_gas)])
        # swingby_outofplane_angles= np.array([swingby_outofplane_angles[i] for i in range(self.no_of_gas)])

        # node times
        self.node_times = util.get_node_times(departure_date, time_of_flights)
        # print(node_times)

        # leg free parameters 
        leg_free_parameters = np.concatenate(np.array([np.append(number_of_revolutions[i],
            free_coefficients[i*(self.no_of_free_parameters*3):(i+1)*(self.no_of_free_parameters*3)]) for i
            in range(self.no_of_legs)])).reshape((self.no_of_legs, 1 + 3*
                self.no_of_free_parameters)) # added reshape

        # node free parameters
        node_free_parameters = util.get_node_free_parameters(self.transfer_body_order,
                                                             swingby_periapses,
                                                             incoming_velocities,
                                                             orbit_ori_angles=orbit_ori_angles,
                                                             swingby_inplane_angles=swingby_inplane_angles,
                                                             swingby_outofplane_angles=swingby_outofplane_angles,
                                                             departure_velocity=departure_velocity,
                                                             departure_inplane_angle=departure_inplane_angle,
                                                             departure_outofplane_angle=departure_outofplane_angle,
                                                             arrival_velocity=arrival_velocity,
                                                             arrival_inplane_angle=arrival_inplane_angle,
                                                             arrival_outofplane_angle=arrival_outofplane_angle)

        try:
            transfer_trajectory_object.evaluate(self.node_times, leg_free_parameters, node_free_parameters)
            # self.delivery_mass_constraint_check(transfer_trajectory_object, self.Isp, self.m0, self.no_of_points)

            # self.transfer_trajectory_object = transfer_trajectory_object
            #
            # if post_processing == False:
            #     objective = self.get_objectives()
            # else:
            #     return
            #
            if post_processing == False:
                objective = self.get_objectives(transfer_trajectory_object)
            elif post_processing == True:
                self.transfer_trajectory_object = transfer_trajectory_object
                return


        except RuntimeError as e:
            mass_penalty = 0
            negative_distance_penalty = 0
            if e == 'Error with validity of trajectory: the delivery mass is negative.':
                print(e)
                mass_penalty = 10**16
            elif e == 'Error when computing radial distance in hodographic shaping: computed distance is negative.':
                print(e)
                negative_distance_penalty = 10**16
            else:
                print('Unspecified error : ', e)
                other_penalty = 10**16

            if len(self.objectives) == 1:
                objective = [mass_penalty + negative_distance_penalty + other_penalty]
            else:
                objective = [mass_penalty + negative_distance_penalty + other_penalty for _ in range(2)]
                print(objective)

        return objective


class MGALowThrustTrajectoryOptimizationProblemOptimAngles(MGALowThrustTrajectoryOptimizationProblem):
    def __init__(self,
                    transfer_body_order,
                    no_of_free_parameters = 0,
                    bounds = None, 
                    depart_semi_major_axis=np.inf,
                    depart_eccentricity=0, 
                    target_semi_major_axis=np.inf, 
                    target_eccentricity=0,
                    swingby_altitude=200000000, #2e5 km
                    departure_velocity=2000, 
                    arrival_velocity=0,
                    Isp=3000,
                    m0=1000,
                    no_of_points=500,
                    manual_base_functions=False,
                    dynamic_bounds = {'time_of_flight' : False,
                                      'orbit_ori_angle' : False,
                                      'swingby_outofplane' : False,
                                      'swingby_inplane' : False,
                                      'shaping_function' : False},
                    manual_tof_bounds=None,
                    objectives=['dv'],
                    zero_revs=False):

        super().__init__(transfer_body_order=transfer_body_order,
                         no_of_free_parameters=no_of_free_parameters, bounds=bounds,
                         depart_semi_major_axis=depart_semi_major_axis,
                         depart_eccentricity=depart_eccentricity,
                         target_semi_major_axis=target_semi_major_axis,
                         target_eccentricity=target_eccentricity, swingby_altitude=swingby_altitude,
                         departure_velocity=departure_velocity, arrival_velocity=arrival_velocity,
                         Isp=Isp, m0=m0, no_of_points=no_of_points,
                         manual_base_functions=manual_base_functions,
                         dynamic_bounds=dynamic_bounds, manual_tof_bounds=manual_tof_bounds,
                         objectives=objectives, zero_revs=zero_revs)

        self.flyby_directions = util.get_flyby_directions(self.transfer_body_order)
        self.bound_names = ['Departure date [mjd2000]', 'Time of Flight [s]', 'Incoming velocity [m/s]', 
                  'Swingby periapsis [m]', 'Orbit orientation angle [rad]', 'Swingby in-plane Angle [rad]', 
                  'Swingby out-of-plane angle [rad]', 'Free coefficient [-]', 'Number of revolutions [-]']


    def get_ooa_angle_from_quantity(self, ooa_quantity):
        ooa_angle = ooa_quantity.copy()
        if ooa_quantity > np.pi:
            ooa_angle = ooa_angle * (1 / 3) + (np.pi / 2)
            try:
                assert 5 * np.pi / 6 < ooa_angle < 7 * np.pi / 6
            except AssertionError:
                print(ooa_quantity, ooa_angle)
        else:
            ooa_angle = ooa_angle * (1 / 3) - (np.pi / 6)
            try:
                assert -np.pi / 6 < ooa_angle < np.pi / 6
            except AssertionError:
                print(ooa_quantity, ooa_angle)

        return ooa_angle

    def get_ooa_quantity_from_angle(self):
        pass

    def get_inplane_bounds(self, i):

        if self.flyby_directions[i] == 'inner':
            lb =  0; ub = 2 * np.pi
        elif self.flyby_directions[i] == 'outer':
            lb = np.pi; ub = 2 * np.pi
        elif self.flyby_directions[i] == 'same':
            lb =  0; ub = 2 * np.pi
        else:
            raise RuntimeError(f"The type of flyby direction, {self.flyby_directions[i]}, is not defined properly")

        return lb, ub

    def get_outofplane_bounds(self):
        # Assuming Earth-Neptune, Mercury is 7.005 degrees, with 5% margin of safety.
        # Venus is 3.395 degrees, max difference is 10.400 degrees if exactly opposite

        return -10.400 * 1.05 * np.pi / 180, 10.400 * 1.05 * np.pi / 180 # rad


    def get_bounds(self):
        departure_date_lb = self.mjd2000_to_seconds(self.bounds[0][0])
        departure_date_ub = self.mjd2000_to_seconds(self.bounds[1][0])

        time_of_flight_lb = self.mjd2000_to_seconds(self.bounds[0][1])
        time_of_flight_ub = self.mjd2000_to_seconds(self.bounds[1][1])

        incoming_velocity_lb = self.bounds[0][2]
        incoming_velocity_ub = self.bounds[1][2]

        swingby_altitude_lb = self.bounds[0][3]
        swingby_altitude_ub = self.bounds[1][3]

        orbit_ori_angle_lb = self.bounds[0][4]
        orbit_ori_angle_ub = self.bounds[1][4]

        swingby_inplane_angle_lb = self.bounds[0][5]
        swingby_inplane_angle_ub = self.bounds[1][5]

        if self.dynamic_bounds['swingby_outofplane']:
            self.bounds[0][6], self.bounds[1][6] = self.get_outofplane_bounds()
            # print('swingby_outofplane', swingby_outofplane_angle_lb, swingby_outofplane_angle_ub)
        swingby_outofplane_angle_lb = self.bounds[0][6]
        swingby_outofplane_angle_ub = self.bounds[1][6]


        free_coefficients_lb = self.bounds[0][7]
        free_coefficients_ub = self.bounds[1][7]

        number_of_revolutions_lb = self.bounds[0][8]
        number_of_revolutions_ub = self.bounds[1][8]

        lower_bounds = [departure_date_lb] # departure date
        upper_bounds = [departure_date_ub] # departure date

        for leg in range(self.no_of_legs): # time of flight
            if self.manual_tof_bounds != None:
                lower_bounds.append(self.manual_tof_bounds[0][leg] * constants.JULIAN_DAY)
                upper_bounds.append(self.manual_tof_bounds[1][leg] * constants.JULIAN_DAY)
            elif self.dynamic_bounds['time_of_flight']:
                current_time_of_flight_bounds = self.get_tof_bound(leg, (time_of_flight_lb,
                                                                         time_of_flight_ub))
                lower_bounds.append(current_time_of_flight_bounds[0])
                upper_bounds.append(current_time_of_flight_bounds[1])
                # print('tof', current_time_of_flight_bounds[0], current_time_of_flight_bounds[1])
            else:
                lower_bounds.append(time_of_flight_lb)
                upper_bounds.append(time_of_flight_ub)
        for _ in range(self.no_of_gas):
            lower_bounds.append(incoming_velocity_lb)
            upper_bounds.append(incoming_velocity_ub)
        for _ in range(self.no_of_gas):
            lower_bounds.append(swingby_altitude_lb)
            upper_bounds.append(swingby_altitude_ub)
        for _ in range(self.no_of_gas):
            lower_bounds.append(orbit_ori_angle_lb)
            upper_bounds.append(orbit_ori_angle_ub)
        for ga in range(self.no_of_gas):
            if self.dynamic_bounds['swingby_inplane']:
                lb, ub = self.get_inplane_bounds(ga)
                lower_bounds.append(lb)
                upper_bounds.append(ub)
                # print('swingby_inplane', lb, ub)
            else:
                lower_bounds.append(swingby_inplane_angle_lb)
                upper_bounds.append(swingby_inplane_angle_ub)
        for _ in range(self.no_of_gas):
            lower_bounds.append(swingby_outofplane_angle_lb)
            upper_bounds.append(swingby_outofplane_angle_ub)
        for _ in range(self.total_no_of_free_coefficients): # free coefficients
            lower_bounds.append(free_coefficients_lb)
            upper_bounds.append(free_coefficients_ub)
        for _ in range(self.no_of_legs): # number of revolutions
            lower_bounds.append(number_of_revolutions_lb)
            upper_bounds.append(number_of_revolutions_ub)

        return (lower_bounds, upper_bounds)

    def fitness(self, 
                design_parameter_vector : list, 
                bodies = util.create_modified_system_of_bodies(ephemeris_type='JPL'),
                # bodies = environment_setup.create_simplified_system_of_bodies(),
                post_processing=False):

        """
        Assuming no_of_gas == 2 & #fc == 2
        0 - departure_date
        1..4 - time of flights
        4..6 - incoming velocities
        6..8 - swingby periapses
        8..4 - orbit orientation angle
        10..12 - swingby inplane angle
        12..14 - swingby outofplane angle
        14..32 - free_coefficients
        32..35 - number of revolutions
        """
        # print("Design Parameters:", design_parameter_vector, "\n")
        self.design_parameter_vector = design_parameter_vector

        # parameters
        central_body = 'Sun'

        #depart and target elements
        # departure_elements = (self.depart_semi_major_axis, self.depart_eccentricity) 
        # target_elements = (self.target_semi_major_axis, self.target_eccentricity) 

        # indexes
        time_of_flight_index = 1 + self.no_of_legs
        incoming_velocity_index = time_of_flight_index + self.no_of_gas
        swingby_altitude_index = incoming_velocity_index + self.no_of_gas
        orbit_ori_angle_index = swingby_altitude_index + self.no_of_gas
        swingby_inplane_angle_index = orbit_ori_angle_index + self.no_of_gas
        swingby_outofplane_angle_index = swingby_inplane_angle_index + self.no_of_gas
        free_coefficient_index = swingby_outofplane_angle_index + self.total_no_of_free_coefficients
        revolution_index = free_coefficient_index + self.no_of_legs

        ### CONTINUOUS PART ###
        departure_date = np.array(design_parameter_vector[0]) # departure date

        time_of_flights = np.array(design_parameter_vector[1:time_of_flight_index]) # time of flight
        incoming_velocities = \
        np.array(design_parameter_vector[time_of_flight_index:incoming_velocity_index]) # incoming velocities
        swingby_altitudes = np.array([x for x in
                  design_parameter_vector[incoming_velocity_index:swingby_altitude_index]]) # swingby_altitudes

        orbit_ori_angles = \
        np.array(design_parameter_vector[swingby_altitude_index:orbit_ori_angle_index]) # orbit_ori_angle
        # print(f'pre{orbit_ori_angles}')
        if self.dynamic_bounds['orbit_ori_angle']:
            orbit_ori_angles = np.array([self.get_ooa_angle_from_quantity(i) for i in orbit_ori_angles])
            # print('ooa', orbit_ori_angles)


        swingby_inplane_angles = \
        np.array(design_parameter_vector[orbit_ori_angle_index:swingby_inplane_angle_index])
        swingby_outofplane_angles = \
        np.array(design_parameter_vector[swingby_inplane_angle_index:swingby_outofplane_angle_index])
        free_coefficients = \
        np.array(design_parameter_vector[orbit_ori_angle_index:free_coefficient_index]) # hodographic shaping free coefficients

        ### INTEGER PART ###
        if self.zero_revs:
            number_of_revolutions = \
            np.array([0 for _ in design_parameter_vector[free_coefficient_index:revolution_index]])
        else:
            number_of_revolutions = np.array([int(x) for x in
                      design_parameter_vector[free_coefficient_index:revolution_index]]) # number of revolutions

        transfer_trajectory_object = util.get_low_thrust_transfer_object(self.transfer_body_order,
                                                        time_of_flights,
                                                        bodies,
                                                        central_body,
                                                        no_of_free_parameters=self.no_of_free_parameters,
                                                        manual_base_functions=self.manual_base_functions,
                                                        number_of_revolutions=number_of_revolutions,
                                                        dynamic_shaping_functions=self.dynamic_bounds['shaping_function'])

        planetary_radii_sequence = np.zeros(self.no_of_gas)
        for i, body in enumerate(self.transfer_body_order[1:-1]):
            planetary_radii_sequence[i] = self.planetary_radii[body]

        swingby_periapses = np.array([planetary_radii_sequence[i] + swingby_altitudes[i] for i in
            range(self.no_of_gas)]) 
        # incoming_velocity= np.array([incoming_velocities[i] for i in range(self.no_of_gas)])
        # orbit_ori_angle= np.array([orbit_ori_angles[i] for i in range(self.no_of_gas)])
        # swingby_inplane_angles= np.array([swingby_inplane_angles[i] for i in range(self.no_of_gas)])
        # swingby_outofplane_angles= np.array([swingby_outofplane_angles[i] for i in range(self.no_of_gas)])

        # node times
        self.node_times = util.get_node_times(departure_date, time_of_flights)
        # print(node_times)

        # leg free parameters 
        leg_free_parameters = np.concatenate(np.array([np.append(number_of_revolutions[i],
            free_coefficients[i*(self.no_of_free_parameters*3):(i+1)*(self.no_of_free_parameters*3)]) for i
            in range(self.no_of_legs)])).reshape((self.no_of_legs, 1 + 3*
                self.no_of_free_parameters)) # added reshape

        # node free parameters
        node_free_parameters = util.get_node_free_parameters(self.transfer_body_order,
                                                             swingby_periapses,
                                                             incoming_velocities,
                                                             orbit_ori_angles=orbit_ori_angles,
                                                             swingby_inplane_angles=swingby_inplane_angles,
                                                             swingby_outofplane_angles=swingby_outofplane_angles)

        try:
            transfer_trajectory_object.evaluate(self.node_times, leg_free_parameters, node_free_parameters)
            # self.delivery_mass_constraint_check(transfer_trajectory_object, self.Isp, self.m0, self.no_of_points)

            # self.transfer_trajectory_object = transfer_trajectory_object
            #
            # if post_processing == False:
            #     objective = self.get_objectives()
            # else:
            #     return
            #
            if post_processing == False:
                objective = self.get_objectives(transfer_trajectory_object)
            elif post_processing == True:
                self.transfer_trajectory_object = transfer_trajectory_object
                return


        except RuntimeError as e:
            mass_penalty = 0
            negative_distance_penalty = 0
            if e == 'Error with validity of trajectory: the delivery mass is negative.':
                # print(e)
                mass_penalty = 10**16
            elif e == 'Error when computing radial distance in hodographic shaping: computed distance is negative.':
                # print(e)
                negative_distance_penalty = 10**16
            else:
                # print('Unspecified error : ', e)
                other_penalty = 10**16

            if len(self.objectives) == 1:
                objective = [mass_penalty + negative_distance_penalty + other_penalty]
            else:
                objective = [mass_penalty + negative_distance_penalty + other_penalty for _ in range(2)]
                print(objective)

        return objective

class MGALowThrustTrajectoryOptimizationProblemSingleLeg:
    """
    Class to initialize, simulate, and optimize the MGA low-thrust trajectory optimization problem.
    """

    def __init__(self,
                 transfer_body_order,
                 departure_date,
                 # departure_state,
                 # arrival_state,
                 time_of_flight,
                 number_of_revs,
                 departure_velocity=0,
                 departure_ip_angle=0,
                 departure_oop_angle=0,
                 arrival_velocity=0,
                 arrival_ip_angle=0,
                 arrival_oop_angle=0,
                 no_of_free_parameters = 0,
                 bounds = [[], []], 
                 depart_semi_major_axis=np.inf,
                 depart_eccentricity=0, 
                 target_semi_major_axis=np.inf, 
                 target_eccentricity=0,
                 swingby_altitude=200000000, #2e5 km
                 Isp=3000,
                 m0=1000,
                 no_of_points=500,
                 manual_base_functions=False,
                 dynamic_bounds = {'time_of_flight' : False,
                                   'orbit_ori_angle' : False,
                                   'swingby_outofplane' : False,
                                   'swingby_inplane' : False,
                                   'shaping_function' : False},
                 manual_tof_bounds=None,
                 objectives=['dv'],
                 zero_revs=False,
                 bepi_state=None):

        self.transfer_body_order = transfer_body_order
        self.mga_characters = \
        util.transfer_body_order_conversion.get_mga_characters_from_list(self.transfer_body_order)
        self.no_of_gas = len(transfer_body_order)-2
        self.target_body = transfer_body_order[0]
        self.depart_body = transfer_body_order[-1]
        self.no_of_legs = len(transfer_body_order) - 1

        self.departure_date = departure_date
        # self.departure_state = departure_state
        # self.arrival_state = arrival_state
        self.time_of_flight = time_of_flight
        self.number_of_revs = number_of_revs

        self.no_of_free_parameters = no_of_free_parameters
        # self.no_of_free_parameters = 2 if len(transfer_body_order) <= 3 else 1
        self.total_no_of_free_coefficients = self.no_of_free_parameters*3*(self.no_of_legs)

        self.bounds = bounds

        self.depart_semi_major_axis = depart_semi_major_axis
        self.depart_eccentricity = depart_eccentricity
        self.target_semi_major_axis = target_semi_major_axis
        self.target_eccentricity = target_eccentricity

        self.departure_velocity = departure_velocity
        self.departure_ip_angle = departure_ip_angle
        self.departure_oop_angle = departure_oop_angle

        self.arrival_velocity = arrival_velocity
        self.arrival_ip_angle = arrival_ip_angle
        self.arrival_oop_angle = arrival_oop_angle

        self.Isp = Isp
        self.m0 = m0
        self.no_of_points = no_of_points

        self.objectives = objectives
        self.zero_revs = zero_revs
        self.bepi_state = bepi_state

        self.design_parameter_vector = None

        self.transfer_trajectory_object = None
        self.delivery_mass = None
        self.node_times = None

        self.planetary_radii = {'Sun': 696000000.0, 'Mercury': 2439699.9999999995, 'Venus':
                6051800.0, 'Earth': 6371008.366666666, 'Mars': 3389526.6666666665, 'Jupiter':
                69946000.0, 'Saturn': 58300000.0}

        self.manual_base_functions = manual_base_functions
        self.dynamic_bounds = dynamic_bounds
        self.manual_tof_bounds = manual_tof_bounds
        self.bound_names= ['Free coefficient [-]']



    def mjd2000_to_seconds(self, mjd2000):
        # mjd2000 = 51544
        # mjd += mjd2000
        mjd2000 *= constants.JULIAN_DAY
        return mjd2000
        # return time_conversion.julian_day_to_seconds_since_epoch(time_conversion.modified_julian_day_to_julian_day(mjd2000))

    def get_bounds(self):
        # departure_date_lb = self.mjd2000_to_seconds(self.bounds[0][0])
        # departure_date_ub = self.mjd2000_to_seconds(self.bounds[1][0])
        # Departure epoch is given

        # departure_velocity_lb = self.bounds[0][1]
        # departure_velocity_ub = self.bounds[1][1]
        #
        # arrival_velocity_lb = self.bounds[0][2]
        # arrival_velocity_ub = self.bounds[1][2]
        # Departure and arrival conditions are given

        # time_of_flight_lb = self.mjd2000_to_seconds(self.bounds[0][3])
        # time_of_flight_ub = self.mjd2000_to_seconds(self.bounds[1][3])
        # Manually defined

        # incoming_velocity_lb = self.bounds[0][4]
        # incoming_velocity_ub = self.bounds[1][4]
        #
        # swingby_altitude_lb = self.bounds[0][5]
        # swingby_altitude_ub = self.bounds[1][5]
        # Single leg only

        free_coefficients_lb = self.bounds[0][0]
        free_coefficients_ub = self.bounds[1][0]

        # number_of_revolutions_lb = self.bounds[0][7]
        # number_of_revolutions_ub = self.bounds[1][7]
        # Manually defined

        # lower_bounds = [departure_date_lb] # departure date
        # upper_bounds = [departure_date_ub] # departure date
        # lower_bounds.append(departure_velocity_lb) # departure velocity # FIXED
        # upper_bounds.append(departure_velocity_ub) # departure velocity
        # lower_bounds.append(arrival_velocity_lb) # departure velocity # FIXED
        # upper_bounds.append(arrival_velocity_ub) # departure velocity
        lower_bounds = []
        upper_bounds = []

        # for leg in range(self.no_of_legs): # time of flight
        #     if self.manual_tof_bounds != None:
        #         lower_bounds.append(self.manual_tof_bounds[0][leg] * constants.JULIAN_DAY)
        #         upper_bounds.append(self.manual_tof_bounds[1][leg] * constants.JULIAN_DAY)
        #     elif self.dynamic_bounds:
        #         current_time_of_flight_bounds = self.get_tof_bound(leg, (time_of_flight_lb,
        #                                                                  time_of_flight_ub))
        #         lower_bounds.append(current_time_of_flight_bounds[0])
        #         upper_bounds.append(current_time_of_flight_bounds[1])
        #     else:
        #         lower_bounds.append(time_of_flight_lb)
        #         upper_bounds.append(time_of_flight_ub)
        # for _ in range(self.no_of_gas):
        #     lower_bounds.append(incoming_velocity_lb)
        #     upper_bounds.append(incoming_velocity_ub)
        # for _ in range(self.no_of_gas):
        #     lower_bounds.append(swingby_altitude_lb)
        #     upper_bounds.append(swingby_altitude_ub)
        for _ in range(self.total_no_of_free_coefficients): # free coefficients
            lower_bounds.append(free_coefficients_lb)
            upper_bounds.append(free_coefficients_ub)
        # for _ in range(self.no_of_legs): # number of revolutions
        #     lower_bounds.append(number_of_revolutions_lb)
        #     upper_bounds.append(number_of_revolutions_ub)

        return (lower_bounds, upper_bounds)

    def get_nobj(self):
        if len(self.objectives) == 2:
            return 2
        elif len(self.objectives) == 1:
            return 1
        else:
            raise RuntimeError('The amount of objectives has not bee implemented yet, please choose 1 or 2 objectives')

    def get_nic(self):
        return 0

    def get_nix(self):
        # free coefficients, number of revolutions
        return 0 
        # return self.no_of_legs

    def get_states_along_trajectory(self, no_of_points) -> dict:
        """
        Returns the full state history of the hodographic shaping state interpolated at
        'no_of_points' points
        Parameters
        ----------
        no_of_points : number of points interpolated along trajectory
        Returns
        -------
        dict
        """
        return self.transfer_trajectory_object.states_along_trajectory(no_of_points)


    def get_inertial_thrust_accelerations_along_trajectory(self, no_of_points) -> dict:
        """
        Returns the full thrusty acceleration history of the hodographic shaping state interpolated
        at 'no_of_points' points
        Parameters
        ----------
        no_of_points : number of points interpolated along trajectory
        Returns
        -------
        dict
        """
        return self.transfer_trajectory_object.inertial_thrust_accelerations_along_trajectory(no_of_points)

    def get_node_time_list(self):
        """
        Returns the node_times list variable so that it can be used for post-processing
        """
        return self.node_times

    def get_design_parameter_vector(self):
        return self.design_parameter_vector

    def get_objectives(self, transfer_trajectory):
        objective_values = []
        for it, j in enumerate(self.objectives):
            if j == 'dv':
                objective_values.append(transfer_trajectory.delta_v)
            elif j == 'tof':
                objective_values.append(transfer_trajectory.time_of_flight)
            elif j == 'dmf' or j == 'pmf' or j == 'dm':
                thrust_acceleration = \
                transfer_trajectory.inertial_thrust_accelerations_along_trajectory(self.no_of_points)

                mass_history, delivery_mass, invalid_trajectory = \
                util.get_mass_propagation(thrust_acceleration, self.Isp, self.m0)
                if delivery_mass < 0:
                    raise RuntimeError('Error with validity of trajectory: the delivery mass is negative.')
                # (self.m0 - delivery_mass) because of the propulsion mass
                if j == 'dmf':
                    objective_values.append(- delivery_mass / self.m0) # try to maximize
                elif j == 'pmf':
                    objective_values.append((self.m0 - delivery_mass) / self.m0) # try to minimize
                elif j == 'dm':
                    objective_values.append(- delivery_mass) # try to maximize
            elif j == 'int':

                diff = {}
                hs_states = transfer_trajectory.states_along_trajectory(self.no_of_points)
                assert len(self.bepi_state) == len(hs_states)
                # assert all([True for i in range(len(self.bepi_state)) if self.bepi_state[i, 0] == hs_times[i]])

                for it, (epoch, hs_row) in enumerate(enumerate(hs_states.values())):
                    # print(bepi_row, hs_states[it, :])
                    diff[epoch] = np.linalg.norm(np.subtract(self.bepi_state[it, 1:4], hs_row[0:3]))
                    sum = np.sum(result2array(diff))
                objective_values.append(sum)

            else:
                raise RuntimeError('An objective was provided that is not permitted')
        return objective_values

    def get_delivery_mass(self):
        thrust_acceleration = \
        self.transfer_trajectory_object.inertial_thrust_accelerations_along_trajectory(self.no_of_points)
        mass_history, self.delivery_mass, invalid_trajectory = \
        util.get_mass_propagation(thrust_acceleration, self.Isp, self.m0)
        return self.delivery_mass

    
    def delivery_mass_constraint_check(self, transfer_object, Isp, m0, no_of_points=500):
        thrust_acceleration = \
        transfer_object.inertial_thrust_accelerations_along_trajectory(no_of_points)

        mass_history, delivery_mass, invalid_trajectory = util.get_mass_propagation(thrust_acceleration, Isp, m0)


        if invalid_trajectory:
            raise RuntimeError('Error with validity of trajectory: the delivery mass is approaching 0 1e-7.')

        return mass_history, delivery_mass
        
    # def get_data(self):
    #     return self.departure_body, self.departure_state, self.arrival_body, self.arrival_state

    def fitness(self, 
                design_parameter_vector : list,
                bodies = util.create_modified_system_of_bodies_validation(),
                post_processing=False):

        """
        Assuming #fc == 2
        0..6 - free_coefficients
        """
        # bodies = self.create_modified_system_of_bodies_validation()
        # print("Design Parameters:", design_parameter_vector, "\n")
        self.design_parameter_vector = design_parameter_vector

        # parameters
        central_body = 'Sun'

        ### CONTINUOUS PART ###
        departure_date = self.departure_date
        departure_velocity = self.departure_velocity # departure velocity
        departure_oop_angle = self.departure_oop_angle
        departure_ip_angle = self.departure_ip_angle
        arrival_velocity = self.arrival_velocity# arrival velocity
        arrival_oop_angle = self.arrival_oop_angle
        arrival_ip_angle = self.arrival_ip_angle
        time_of_flights = [self.time_of_flight] # time of flight

        ### INTEGER PART ###
        # hodographic shaping free coefficients
        free_coefficients = \
        np.array(design_parameter_vector[0:self.total_no_of_free_coefficients])

        number_of_revolutions = np.array([self.number_of_revs])

        transfer_trajectory_object = util.get_low_thrust_transfer_object(self.transfer_body_order,
                                                        time_of_flights,
                                                        bodies,
                                                        central_body,
                                                        no_of_free_parameters=self.no_of_free_parameters,
                                                        manual_base_functions=self.manual_base_functions,
                                                        number_of_revolutions=number_of_revolutions,
                                                        dynamic_shaping_functions=self.dynamic_bounds['shaping_function'])

        # node times
        self.node_times = util.get_node_times(departure_date, time_of_flights)
        # print(node_times)

        # leg free parameters 
        leg_free_parameters = np.concatenate(np.array([np.append(number_of_revolutions[i],
            free_coefficients[i*(self.no_of_free_parameters*3):(i+1)*(self.no_of_free_parameters*3)]) for i
            in range(self.no_of_legs)])).reshape((self.no_of_legs, 1 + 3*
                self.no_of_free_parameters)) # added reshape

        # node free parameters
        node_free_parameters = util.get_node_free_parameters(self.transfer_body_order,
                                                             departure_velocity=departure_velocity,
                                                             departure_inplane_angle=departure_ip_angle,
                                                             departure_outofplane_angle=departure_oop_angle,
                                                             arrival_velocity=arrival_velocity,
                                                             arrival_inplane_angle=arrival_ip_angle,
                                                             arrival_outofplane_angle=arrival_oop_angle)

        try:
            transfer_trajectory_object.evaluate(self.node_times, leg_free_parameters, node_free_parameters)
            # self.delivery_mass_constraint_check(transfer_trajectory_object, self.Isp, self.m0, self.no_of_points)

            # self.transfer_trajectory_object = transfer_trajectory_object
            #
            # if post_processing == False:
            #     objective = self.get_objectives()
            # else:
            #     return

            if post_processing == False:
                objective = self.get_objectives(transfer_trajectory_object)
            elif post_processing == True:
                self.transfer_trajectory_object = transfer_trajectory_object
                return


        except RuntimeError as e:
            mass_penalty = 0
            negative_distance_penalty = 0
            if e == 'Error with validity of trajectory: the delivery mass is negative.':
                print(e)
                mass_penalty = 10**16
            elif e == 'Error when computing radial distance in hodographic shaping: computed distance is negative.':
                print(e)
                negative_distance_penalty = 10**16
            else:
                print('Unspecified error : ', e)
                other_penalty = 10**16

            if len(self.objectives) == 1:
                objective = [mass_penalty + negative_distance_penalty + other_penalty]
            else:
                objective = [mass_penalty + negative_distance_penalty + other_penalty for _ in range(2)]
                print(objective)

        return objective

#######################################################################
# Penalty functions ###################################################
#######################################################################





