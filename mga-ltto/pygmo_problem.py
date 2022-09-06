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
import numpy as np
import os

# Tudatpy imports
import tudatpy
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.math import interpolators
from tudatpy.kernel.trajectory_design import shape_based_thrust
from tudatpy.kernel.trajectory_design import transfer_trajectory
from tudatpy.kernel.interface import spice
spice.load_standard_kernels()

import mga_low_thrust_utilities as mga_util


#######################################################################
# PROBLEM FUNCTIONS ###################################################
#######################################################################

# def constraint_check(design_parameter_vector: np.array):
#     """
#     Checks if all the parameters are within certain constraints, if not provide information on what is going wrong
#     """
# 
#     # Swingby periapsis check
#     transfer_body_order = design_parameter_vector[]
#     swingby_periapses = design_parameter_vector[]
# 
#     for i in range(len(transfer_body_order)-2):
#         planetary_radii = np.zeros(6)
#         if swingby_periapses[i+1] < spice.get_average_radius(transfer_body_order[i+1]) + 20000: #20 km buffer
#             print("The swingby of {} is occurring within the planet".format{transfer_body_order[i+1]})
#             swingby_periapses[i+1] = spice.get_average_radius(transfer_body_order[i+1] + 20000
# 
#     # Transfer length check
#     if len(transfer_body_order) > 10:
#         print("The swingby length is {}, which is longer than the maximum allowed".format(len(transfer_body_order)))
# 







#######################################################################
# PROBLEM CLASS #######################################################
#######################################################################

class MGALowThrustTrajectoryOptimizationProblem:
    """
    Class to initialize, simulate, and optimize the MGA low-thrust trajectory optimization problem.
    """

    # def __init__(self, design_parameter_vector):
    #     self.design_parameter_vector = design_parameter_vector

    def __init__(self, 
                    target_body='Jupiter', 
                    depart_body='Earth', 
                    bodies_to_create = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus',
                        'Neptune', 'Sun'], 
                    depart_semi_major_axis=np.inf,
                    depart_eccentricity=0, 
                    target_semi_major_axis=np.inf, 
                    target_eccentricity=0,
                    swingby_altitude=200000, 
                    departure_velocity=2000, 
                    arrival_velocity=0, 
                    pre_dep_vel=False,
                    pre_arr_vel=True,
                    no_of_free_parameters=2,
                    max_no_of_gas=6):

        self.target_body = target_body
        self.depart_body = depart_body
        self.bodies_to_create = bodies_to_create
        self.depart_semi_major_axis = depart_semi_major_axis
        self.depart_eccentricity = depart_eccentricity
        self.target_body = target_body
        self.target_semi_major_axis = target_semi_major_axis
        self.target_eccentricity = target_eccentricity
        self.swingby_altitude = swingby_altitude
        self.departure_velocity = departure_velocity
        self.arrival_velocity = arrival_velocity
        self.pre_dep_vel = pre_dep_vel
        self.pre_arr_vel = pre_arr_vel
        self.no_of_free_parameters = no_of_free_parameters
        self.max_no_of_gas = max_no_of_gas

        self.transfer_trajectory_object = None
        self.node_times = None

        planetary_radii = {}
        for i in self.bodies_to_create:
            planetary_radii[i] = spice.get_average_radius(i)
        self.planetary_radii = planetary_radii

        self.number_of_revolution_parameters = self.max_no_of_gas + 1
        self.total_no_of_free_coefficients = self.no_of_free_parameters*3*(self.max_no_of_gas+1)

        # global_frame_origin = 'SSB'
        # global_frame_orientation = 'ECLIPJ2000'
        # self.bodies = \
        # environment_setup.create_system_of_bodies(environment_setup.get_default_body_settings(self.bodies_to_create,
        #     global_frame_origin, global_frame_orientation))

    def get_bounds(self):
        julian_day = constants.JULIAN_DAY
        lower_bounds = [-1000*julian_day] # departure date
        lower_bounds.append(self.departure_velocity) if self.pre_dep_vel == True else \
        lower_bounds.append(0) # departure velocity
        # lower_bounds.append([50 for i in range(7)])
        # lower_bounds.append([0 for i in range(8)])
        for _ in range(self.max_no_of_gas + 1): # time of flight
            lower_bounds.append(50) 
        for _ in range(self.total_no_of_free_coefficients): # free coefficients
            lower_bounds.append(-10**6)
        for _ in range(self.max_no_of_gas): # planet identifier
            lower_bounds.append(0)
        for _ in range(self.number_of_revolution_parameters): # number of revolutions
            lower_bounds.append(0)

        upper_bounds = [1000*julian_day] # departure date
        upper_bounds.append(self.departure_velocity) if self.pre_dep_vel == True else \
        upper_bounds.append(2000) # departure velocity

        for _ in range(self.max_no_of_gas + 1): # time of flight
            upper_bounds.append(2000)
        for _ in range(self.total_no_of_free_coefficients): # free coefficients
            upper_bounds.append(10**6)
        for _ in range(self.max_no_of_gas): # planet identifier
            upper_bounds.append(15)
        for _ in range(self.number_of_revolution_parameters): # number of revolutions
            upper_bounds.append(4)

        #print(lower_bounds, upper_bounds)

        return (lower_bounds, upper_bounds)
    # design_parameter_vector[0] = departure_date
    # design_parameter_vector[1] = departure_velocity_magnitude
    # design_parameter_vector[2:9] = time_of_flight
    # design_parameter_vector[9:17] = transfer_body_order

    def get_central_body(self, central_body):
        self.central_body = central_body

    # def get_bodies(self, bodies):
    #     self.bodies = bodies

    # def get_depart_body(self, depart_body, depart_semi_major_axis=np.inf, depart_eccentricity=0):

    # def get_target_body(self, target_body, target_semi_major_axis=np.inf, target_eccentricity=0):

    def get_nic(self):
        return 0

    def get_nix(self):
        return self.max_no_of_gas + self.number_of_revolution_parameters # maximum of 6 GAs

    # def get_planet_radii(transfer_body_order: list) -> np.ndarray:
    #     self.planetary_radii = planetary_radii

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
        return self.transfer_trajectory_object().states_along_trajectory(no_of_points)


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
        return self.transfer_trajectory_object().inertial_thrust_accelerations_along_trajectory(no_of_points)

    def get_node_times(self):
        """
        Returns the node_times list variable so that it can be used for post-processing
        """
        return self.node_times
 
    def fitness(self, 
                design_parameter_vector : list, 
                bodies = environment_setup.create_simplified_system_of_bodies()):

        """
        Assuming max_no_of_gas == 6
        0 - departure_date
        1 - departure velocity
        2..9 - time of flights
        9..51 - free_coefficients
        51..57 - integer ga identifier 
        """
        print("Design Parameters:", design_parameter_vector, "\n")

        # parameters
        freq = 1e-6
        scale = 1e-6
        # bodies and central body
        # cannot create system of bodies to self, because it cannot be pickled -> QUESTION
        # bodies = environment_setup.create_simplified_system_of_bodies()
        central_body = 'Sun'

        #depart and target elements
        departure_elements = (self.depart_semi_major_axis, self.depart_eccentricity) 
        target_elements = (self.target_semi_major_axis, self.target_eccentricity) 

        # indexes
        time_of_flight_index = 2 + self.max_no_of_gas + 1
        free_coefficient_index = time_of_flight_index + self.total_no_of_free_coefficients
        planet_identifier_index = free_coefficient_index + self.max_no_of_gas
        revolution_index = planet_identifier_index + self.number_of_revolution_parameters

        # departure date
        departure_date = design_parameter_vector[0]

        
        # time of flight
        time_of_flights = design_parameter_vector[2:time_of_flight_index]

        # hodographic shaping free coefficients
        free_coefficients = design_parameter_vector[time_of_flight_index:free_coefficient_index]
        # print('design parameters vector', design_parameter_vector)
        # print('free coefficients', free_coefficients.shape)
        # free_coefficients = free_coefficients.astype(np.float64)

        # transfer_body_order
        transfer_body_converter = mga_util.transfer_body_order_conversion()
        transfer_body_order = \
            transfer_body_converter.get_transfer_body_list(design_parameter_vector[
                free_coefficient_index:planet_identifier_index], strip=True)
        transfer_body_order.insert(0, self.depart_body)
        transfer_body_order.append(self.target_body)


        # number of revolutions
        # number_of_revolutions = \
        # design_parameter_vector[planet_identifier_index:revolution_index].astype(int)
        number_of_revolutions = \
        [int(x) for x in design_parameter_vector[planet_identifier_index:revolution_index]]
        # print('number of revolution', type(number_of_revolutions[0]))

        # approach 2
        # leg_parameters = np.zeros((self.max_no_of_gas + 1, 1 + self.no_of_free_parameters * 3))
        #
        # for i in range(self.max_no_of_gas + 1):
        #     leg_parameters[i+1, :] = \
        #     free_coefficients[i*(self.no_of_free_parameters*3):(i+1)*(self.no_of_free_parameters*3)]
        #
        # leg_parameters[:, 0] = number_of_revolutions
        # leg_parameters.flatten()
        # print('leg parameters: ', leg_parameters)
        # print('leg parameters: ', type(leg_parameters[0]))
        # print('leg parameters: ', leg_parameters.shape)


        #approach 3
        # just insert the number of revolutions elements in the free_coefficients array

        # approach 1
        leg_parameters = np.concatenate(np.array([np.append(number_of_revolutions[i],
            free_coefficients[i*(self.no_of_free_parameters*3):(i+1)*(self.no_of_free_parameters*3)]) for i
            in range(self.max_no_of_gas+1)])) # add free coefficients later
        # print('leg parameters: ', leg_parameters)
        # print('leg parameters: ', type(leg_parameters[0]))
        # print('leg parameters: ', leg_parameters.shape)

        transfer_trajectory_object = mga_util.get_low_thrust_transfer_object(transfer_body_order,
                                                                    time_of_flights,
                                                                    departure_elements,
                                                                    target_elements,
                                                                    bodies,
                                                                    central_body,
                                                                    no_of_free_parameters=self.no_of_free_parameters,
                                                                    number_of_revolutions=number_of_revolutions,
                                                                    frequency=freq,
                                                                    scale_factor=scale)

        

        planetary_radii_sequence = np.zeros(self.max_no_of_gas)
        for i, body in enumerate(transfer_body_order[1:-1]):
            planetary_radii_sequence[i] = self.planetary_radii[body]

        swingby_periapses = np.array([planetary_radii_sequence[i] + self.swingby_altitude for i in
            range(self.max_no_of_gas)]) # defined depending on problem
        incoming_velocities = np.array([2000 for _ in range(self.max_no_of_gas)]) #defined depending on
        # problem


        node_times = mga_util.get_node_times(transfer_body_order, departure_date, time_of_flights)
        self.node_times = node_times
        leg_free_parameters = mga_util.get_leg_free_parameters(leg_parameters, transfer_body_order)
        node_free_parameters=  mga_util.get_node_free_parameters(transfer_body_order,
                swingby_periapses, incoming_velocities)

        


        # print("test")
        try:
            transfer_trajectory_object.evaluate(node_times, leg_free_parameters, node_free_parameters)
            objective = transfer_trajectory_object.delta_v 
        except RuntimeError:
            print("excepted")
            objective = 10**12
            # print(transfer_trajectory_object)
            # objective = transfer_trajectory_object.delta_v + objective_penalty

        #self.transfer_trajectory_object = transfer_trajectory_object # resulted in error last time
        # try:
        #     objective = transfer_trajectory_object.delta_v
        # except RuntimeError:
        #     print("excepted")
        #     fitness +=10**9
        #constraint_check(design_parameter_vector)
        # objective = np.random.normal(0, 1000)

        # if cylindrical_penalty:
        #     objective += 10**12
        #     print('Penalty applied')
        
        print('Fitness evaluated')
        return [objective]

