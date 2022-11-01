'''
Author: Sean Cowan
Purpose: MSc Thesis
Date Created: 26-07-2022

This module creates a PyGMO compatible problem class that represents the mga low-thrust trajectories.
'''
###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import pickle
import numpy as np
import os

# Tudatpy imports
import tudatpy
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.trajectory_design import shape_based_thrust
from tudatpy.kernel.trajectory_design import transfer_trajectory
from tudatpy.kernel.astro import time_conversion

import mga_low_thrust_utilities as util

#######################################################################
# PROBLEM CLASS #######################################################
#######################################################################

class MGALowThrustTrajectoryOptimizationProblem:
    """
    Class to initialize, simulate, and optimize the MGA low-thrust trajectory optimization problem.
    """

    def __init__(self,
                    transfer_body_order,
                    no_of_free_parameters =0 ,
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
                    planet_kep_states = None,
                    manual_base_functions=False,
                    mo_optimisation=False):

        self.transfer_body_order = transfer_body_order
        self.no_of_gas = len(transfer_body_order)-2
        self.target_body = transfer_body_order[0]
        self.depart_body = transfer_body_order[-1]
        self.no_of_legs = len(transfer_body_order) - 1

        self.no_of_free_parameters = no_of_free_parameters
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

        self.mo_optimisation = mo_optimisation

        # self.bodies_to_create = ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn",
        #         "Uranus", "Neptune"] 
        # self.bodies_to_create = ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn"]
        # self.central_body = 'Sun'
        # self.central_body_mu = 1.3271244e20 # m^3 / s^2

        # self.system_of_bodies = lambda :  util.create_modified_system_of_bodies(self.bounds[0][0], # departure date
        #     self.central_body_mu, bodies=self.bodies_to_create, ephemeris_type='JPL',
        #     planet_kep_states=planet_kep_states)

        self.design_parameter_vector = None

        self.transfer_trajectory_object = None
        self.node_times = None

        # planetary_radii = {}
        # for i in self.bodies_to_create:
        #     planetary_radii[i] = 1e8#spice.get_average_radius(i)
        # planetary radius from spice from get)
        self.planetary_radii = {'Sun': 696000000.0, 'Mercury': 2439699.9999999995, 'Venus':
                6051800.0, 'Earth': 6371008.366666666, 'Mars': 3389526.6666666665, 'Jupiter':
                69946000.0, 'Saturn': 58300000.0}

        self.manual_base_functions = manual_base_functions

    def mjd2000_to_seconds(self, mjd2000):
        # mjd2000 = 51544
        # mjd += mjd2000
        mjd2000 *= constants.JULIAN_DAY
        return mjd2000
        # return time_conversion.julian_day_to_seconds_since_epoch(time_conversion.modified_julian_day_to_julian_day(mjd2000))

    def swingby_periapsis_to_bound(self, bound):
        return np.floor(np.log10(np.abs(bound))).astype(int)

    def get_bounds(self):
        departure_date_lb = self.mjd2000_to_seconds(self.bounds[0][0])
        departure_date_ub = self.mjd2000_to_seconds(self.bounds[1][0])
        # departure_date_lb = self.bounds[0][0]
        # departure_date_ub = self.bounds[1][0]

        departure_velocity_lb = self.bounds[0][1]
        departure_velocity_ub = self.bounds[1][1]

        arrival_velocity_lb = self.bounds[0][2]
        arrival_velocity_ub = self.bounds[1][2]

        # time_of_flight_lb = self.bounds[0][2]
        # time_of_flight_ub = self.bounds[1][2]
        time_of_flight_lb = self.mjd2000_to_seconds(self.bounds[0][3])
        time_of_flight_ub = self.mjd2000_to_seconds(self.bounds[1][3])

        incoming_velocity_lb = self.bounds[0][4]
        incoming_velocity_ub = self.bounds[1][4]

        swingby_periapsis_lb = self.swingby_periapsis_to_bound(self.bounds[0][5])
        swingby_periapsis_ub = self.swingby_periapsis_to_bound(self.bounds[1][5])

        free_coefficients_lb = self.bounds[0][6]
        free_coefficients_ub = self.bounds[1][6]

        number_of_revolutions_lb = self.bounds[0][7]
        number_of_revolutions_ub = self.bounds[1][7]

        lower_bounds = [departure_date_lb] # departure date
        lower_bounds.append(departure_velocity_lb) # departure velocity # FIXED
        lower_bounds.append(arrival_velocity_lb) # departure velocity # FIXED

        for _ in range(self.no_of_legs): # time of flight
            lower_bounds.append(time_of_flight_lb) 
        for _ in range(self.no_of_gas):
            lower_bounds.append(incoming_velocity_lb)
        for _ in range(self.no_of_gas):
            lower_bounds.append(swingby_periapsis_lb)
        for _ in range(self.total_no_of_free_coefficients): # free coefficients
            lower_bounds.append(free_coefficients_lb)
        for _ in range(self.no_of_legs): # number of revolutions
            lower_bounds.append(number_of_revolutions_lb)


        upper_bounds = [departure_date_ub] # departure date
        upper_bounds.append(departure_velocity_ub) # departure velocity
        upper_bounds.append(arrival_velocity_ub) # departure velocity

        for _ in range(self.no_of_legs): # time of flight
            upper_bounds.append(time_of_flight_ub)
        for _ in range(self.no_of_gas):
            upper_bounds.append(incoming_velocity_ub)
        for _ in range(self.no_of_gas):
            upper_bounds.append(swingby_periapsis_ub)
        for _ in range(self.total_no_of_free_coefficients): # free coefficients
            upper_bounds.append(free_coefficients_ub)
        for _ in range(self.no_of_legs): # number of revolutions
            upper_bounds.append(number_of_revolutions_ub)

        return (lower_bounds, upper_bounds)

    def get_nobj(self):
        if self.mo_optimisation:
            return 2
        else:
            return 1

    def get_nic(self):
        return 0

    def get_nix(self):
        # free coefficients, number of revolutions
        return self.total_no_of_free_coefficients + self.no_of_legs 

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
        departure_elements = (self.depart_semi_major_axis, self.depart_eccentricity) 
        target_elements = (self.target_semi_major_axis, self.target_eccentricity) 

        # indexes
        time_of_flight_index = 3 + self.no_of_legs
        incoming_velocity_index = time_of_flight_index + self.no_of_gas
        swingby_periapsis_index = incoming_velocity_index + self.no_of_gas
        free_coefficient_index = swingby_periapsis_index + self.total_no_of_free_coefficients
        revolution_index = free_coefficient_index + self.no_of_legs

        ### CONTINUOUS PART ###
        # departure date
        departure_date = design_parameter_vector[0]

        # departure velocity
        departure_velocity = design_parameter_vector[1] 

        # arrival velocity
        arrival_velocity = design_parameter_vector[2] 

        # time of flight
        time_of_flights = design_parameter_vector[3:time_of_flight_index]
        # print(time_of_flights)

        # incoming velocities
        incoming_velocities = design_parameter_vector[time_of_flight_index:incoming_velocity_index]

        # swingby_periapses
        swingby_periapses = \
        [10**x for x in design_parameter_vector[incoming_velocity_index:swingby_periapsis_index]]

        ### INTEGER PART ###
        # hodographic shaping free coefficients
        free_coefficients = design_parameter_vector[swingby_periapsis_index:free_coefficient_index]

        # number of revolutions
        number_of_revolutions = \
        [int(x) for x in design_parameter_vector[free_coefficient_index:revolution_index]]
        if self.mo_optimisation:
            number_of_revolutions = \
            [0 for _ in design_parameter_vector[free_coefficient_index:revolution_index]]

        transfer_trajectory_object = util.get_low_thrust_transfer_object(self.transfer_body_order,
                                                            time_of_flights,
                                                            departure_elements,
                                                            target_elements,
                                                            # self.system_of_bodies(),
                                                            bodies,
                                                            central_body,
                                                            no_of_free_parameters=self.no_of_free_parameters,
                                                            manual_base_functions=self.manual_base_functions,
                                                            number_of_revolutions=number_of_revolutions)

        planetary_radii_sequence = np.zeros(self.no_of_gas)
        for i, body in enumerate(self.transfer_body_order[1:-1]):
            planetary_radii_sequence[i] = self.planetary_radii[body]

        # swingby_periapses = np.array([planetary_radii_sequence[i] + self.swingby_altitude for i in
        #    range(self.no_of_gas)]) # defined depending on problem
        swingby_periapses_array = np.array([planetary_radii_sequence[i] + swingby_periapses[i] for i in
            range(self.no_of_gas)]) # defined depending on problem
        # incoming_velocities = np.array([2000 for _ in range(self.no_of_gas)]) 
        incoming_velocity_array = np.array([incoming_velocities[i] for i in range(self.no_of_gas)])

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
                swingby_periapses_array, incoming_velocity_array, departure_velocity=departure_velocity,
                arrival_velocity=arrival_velocity)

        try:
            transfer_trajectory_object.evaluate(self.node_times, leg_free_parameters, node_free_parameters)
            # delivery_mass_constraint_check(transfer_trajectory_object, self.Isp, self.m0, self.no_of_points)

            if post_processing == False and self.mo_optimisation == False:
                objective = [transfer_trajectory_object.delta_v]
            elif post_processing == False and self.mo_optimisation == True:
                objective = [transfer_trajectory_object.delta_v, transfer_trajectory_object.time_of_flight]
            elif post_processing == True:
                self.transfer_trajectory_object = transfer_trajectory_object

        except RuntimeError as e:
            mass_penalty = 0
            negative_distance_penalty = 0
            if e == 'Error with validity of trajectory: the delivery mass is negative.':
                # mass_penalty = 5 * 10**15
                print(e)
                mass_penalty = 10**16
            elif e == 'Error when computing radial distance in hodographic shaping: computed distance is negative.':
                # negative_distance_penalty = 5 * 10**15
                print(e)
                negative_distance_penalty = 10**16
            else:
                print('Unspecified error : ', e)
                other_penalty = 10**16

            if self.mo_optimisation == False:
                # objective = [10**16]
                objective = [mass_penalty + negative_distance_penalty + other_penalty]
            else:
                # objective = [10**16, 10**16]
                objective = [mass_penalty + negative_distance_penalty + other_penalty for _ in range(2)]
                print(objective)


            # objective[0] += [10**16] + [mass_penalty]
            if post_processing == True:
                raise
        if post_processing == False:
            return objective
            # print('Fitness evaluated')

#######################################################################
# Penalty functions ###################################################
#######################################################################

def delivery_mass_constraint_check(transfer_object, Isp, m0, no_of_points=500):
    thrust_acceleration = \
    transfer_object.inertial_thrust_accelerations_along_trajectory(no_of_points)

    mass_history, delivery_mass, invalid_trajectory = util.get_mass_propagation(thrust_acceleration, Isp, m0)


    if invalid_trajectory:
        raise RuntimeError('Error with validity of trajectory: the delivery mass is approaching 0 1e-7.')

    return mass_history, delivery_mass



