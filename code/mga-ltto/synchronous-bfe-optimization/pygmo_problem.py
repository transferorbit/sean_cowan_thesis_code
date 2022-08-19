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

    def __init__(self, target_body, bodies, depart_body='Earth', depart_semi_major_axis=np.inf, depart_eccentricity=0,
        target_semi_major_axis=np.inf, target_eccentricity=0):
        self.depart_body = depart_body
        self.depart_semi_major_axis = depart_semi_major_axis
        self.depart_eccentricity = depart_eccentricity
        self.target_body = target_body
        self.target_semi_major_axis = target_semi_major_axis
        self.target_eccentricity = target_eccentricity
        self.transfer_trajectory_object = None
        self.node_times = None

        planetary_radii = {}
        for i in bodies:
            planetary_radii[i] = spice.get_average_radius(i)
        self.planetary_radii = planetary_radii

    def get_bounds(self):
        julian_day = constants.JULIAN_DAY
        lower_bounds = [-1000*julian_day, 0]
        # lower_bounds.append([50 for i in range(7)])
        # lower_bounds.append([0 for i in range(8)])
        for _ in range(7):
            lower_bounds.append(50)
        for _ in range(8):
            lower_bounds.append(0)

        upper_bounds = [1000*julian_day, 5000]
        # upper_bounds.append([2000 for i in range(7)])
        # upper_bounds.append([8 for i in range(8)])
        # upper_bounds.append(2000*7)
        # upper_bounds.append(8*8)
        for _ in range(7):
            upper_bounds.append(2000)
        for _ in range(8):
            upper_bounds.append(15)

        #print(lower_bounds, upper_bounds)

        return (lower_bounds, upper_bounds)
    # design_parameter_vector[0] = departure_date
    # design_parameter_vector[1] = departure_velocity_magnitude
    # design_parameter_vector[2:9] = time_of_flight
    # design_parameter_vector[9:17] = transfer_body_order

    def get_central_body(self, central_body):
        self.central_body = central_body

    def get_bodies(self, bodies):
        self.bodies = bodies

    # def get_depart_body(self, depart_body, depart_semi_major_axis=np.inf, depart_eccentricity=0):

    # def get_target_body(self, target_body, target_semi_major_axis=np.inf, target_eccentricity=0):

    def get_nic(self):
        return 0

    def get_nix(self):
        return 6

    # def get_planet_radii(transfer_body_order: list) -> np.ndarray:
    #     self.planetary_radii = planetary_radii

    def get_states_along_trajectory(self, no_of_points) -> dict:
        """
        Returns the full state history of the hodographic shaping state interpolated at 'no_of_points' points
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
        Returns the full thrusty acceleration history of the hodographic shaping state interpolated at 'no_of_points' points
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
 
    def batch_fitness(self, design_parameter_vector: list):

        
    def fitness(self, design_parameter_vector : list):

        # parameters
        freq = 1e-6
        scale = 1e-6
        # bodies and central body
        bodies = environment_setup.create_simplified_system_of_bodies()
        central_body = 'Sun'

        #depart and target elements
        depart_body=  "Earth"
        departure_elements = (self.depart_semi_major_axis, self.depart_eccentricity)
        target_body = "Jupiter"
        target_elements = (self.target_semi_major_axis, self.target_eccentricity)

        departure_date = design_parameter_vector[0]
        
        # transfer_body_order
        transfer_body_converter = mga_util.transfer_body_order_conversion()
        transfer_body_order = \
        transfer_body_converter.get_transfer_body_list(design_parameter_vector[10:18], strip=True)
        transfer_body_order.insert(0, depart_body)
        transfer_body_order.append(target_body)
        no_of_gas = len(transfer_body_order)-2

        # time of flight
        time_of_flight = design_parameter_vector[2:(2+no_of_gas+1)]
         
        transfer_trajectory_object = mga_util.get_low_thrust_transfer_object(transfer_body_order,
                                                                                time_of_flight,
                                                                                departure_elements,
                                                                                target_elements,
                                                                                bodies,
                                                                                central_body,
                                                                                frequency=freq,
                                                                                scale_factor=scale)

        

        planetary_radii_sequence = np.zeros(no_of_gas)
        for i, body in enumerate(transfer_body_order[1:-1]):
            planetary_radii_sequence[i] = self.planetary_radii[body]

        swingby_periapses = np.array([planetary_radii_sequence[i] + 200000 for i in range(no_of_gas)])
        incoming_velocities = np.array([2000 for i in range(no_of_gas)])


        node_times = mga_util.get_node_times(transfer_body_order, departure_date, time_of_flight)
        self.node_times = node_times
        leg_free_parameters = mga_util.get_leg_free_parameters(np.zeros(1), transfer_body_order)
        node_free_parameters=  mga_util.get_node_free_parameters(transfer_body_order, swingby_periapses, incoming_velocities)

        


        transfer_trajectory_object.evaluate(node_times, leg_free_parameters, node_free_parameters)

        self.transfer_trajectory_object = lambda: transfer_trajectory_object
        objective = transfer_trajectory_object.delta_v
        #constraint_check(design_parameter_vector)
        
        return [objective]

