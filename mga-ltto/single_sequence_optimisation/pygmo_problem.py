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
from tudatpy.kernel.interface import spice
spice.load_standard_kernels()

import mga_low_thrust_utilities as mga_util



#######################################################################
# PROBLEM CLASS #######################################################
#######################################################################

class MGALowThrustTrajectoryOptimizationProblem:
    """
    Class to initialize, simulate, and optimize the MGA low-thrust trajectory optimization problem.
    """

    def __init__(self,
                    transfer_body_order,
                    no_of_free_parameters,
                    bounds, 
                    depart_semi_major_axis=np.inf,
                    depart_eccentricity=0, 
                    target_semi_major_axis=np.inf, 
                    target_eccentricity=0,
                    swingby_altitude=200000000, #2e5 km
                    departure_velocity=2000, 
                    arrival_velocity=0):

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

        self.bodies_to_create = ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn",
                "Uranus", "Neptune"] 

# Create bodies in simulation
        # body_list_settings = lambda : \
        #     environment_setup.get_default_body_settings(bodies=self.bodies_to_create,
        #             base_frame_origin='SSB', base_frame_orientation="ECLIPJ2000")
        # for i in self.bodies_to_create:
        #     current_body_list_settings = body_list_settings()
        #     current_body_list_settings.add_empty_settings(i)            
        #     current_body_list_settings.get(i).ephemeris_settings = \
        #     environment_setup.ephemeris.approximate_jpl_model(i)        
        #
        # system_of_bodies = environment_setup.create_system_of_bodies(current_body_list_settings)
        # self.system_of_bodies = lambda : system_of_bodies
        #
            # INTERPOLATED SPICE
            # environment_setup.ephemeris.interpolated_spice(9000*constants.JULIAN_DAY,
            #         10500*constants.JULIAN_DAY,
            #         86400)
            # APPROXIMATE JPL MODEL
            # environment_setup.ephemeris.approximate_jpl_model(i)
            # KEPLERIAN FROM SPICE
            # environment_setup.ephemeris.keplerian_from_spice(i,
                    # 6000*constants.JULIAN_DAY,
                    # 1.327*10**20,
                    # frame_origin='SSB',
                    # frame_orientation='ECLIPJ2000')
            # TIME_LIMITED
            # environment_setup.get_default_body_settings_time_limited(bodies=self.bodies_to_create,
            #         initial_time=9000*constants.JULIAN_DAY, final_time=950*constants.JULIAN_DAY,
            #         base_frame_origin='SSB', base_frame_orientation="ECLIPJ2000")



        self.transfer_trajectory_object = None
        self.node_times = None

        planetary_radii = {}
        for i in self.bodies_to_create:
            planetary_radii[i] = spice.get_average_radius(i)
        self.planetary_radii = planetary_radii

    def get_bounds(self):
        departure_date_lb = self.bounds[0][0]
        departure_date_ub = self.bounds[1][0]

        departure_velocity_lb = self.bounds[0][1]
        departure_velocity_ub = self.bounds[1][1]

        time_of_flight_lb = self.bounds[0][2]
        time_of_flight_ub = self.bounds[1][2]

        free_coefficients_lb = self.bounds[0][3]
        free_coefficients_ub = self.bounds[1][3]

        number_of_revolutions_lb = self.bounds[0][4]
        number_of_revolutions_ub = self.bounds[1][4]


        lower_bounds = [departure_date_lb] # departure date
        lower_bounds.append(departure_velocity_lb) # departure velocity # FIXED
        for _ in range(self.no_of_legs): # time of flight
            lower_bounds.append(time_of_flight_lb) 
        for _ in range(self.total_no_of_free_coefficients): # free coefficients
            lower_bounds.append(free_coefficients_lb)
        for _ in range(self.no_of_legs): # number of revolutions
            lower_bounds.append(number_of_revolutions_lb)

        upper_bounds = [departure_date_ub] # departure date
        upper_bounds.append(departure_velocity_ub) # departure velocity

        for _ in range(self.no_of_legs): # time of flight
            upper_bounds.append(time_of_flight_ub)
        for _ in range(self.total_no_of_free_coefficients): # free coefficients
            upper_bounds.append(free_coefficients_ub)
        for _ in range(self.no_of_legs): # number of revolutions
            upper_bounds.append(number_of_revolutions_ub)

        return (lower_bounds, upper_bounds)

    def get_central_body(self, central_body):
        self.central_body = central_body

    # def get_bodies(self, bodies):
    #     self.bodies = bodies

    # def get_depart_body(self, depart_body, depart_semi_major_axis=np.inf, depart_eccentricity=0):

    # def get_target_body(self, target_body, target_semi_major_axis=np.inf, target_eccentricity=0):

    def get_nic(self):
        return 0

    def get_nix(self):
        return self.no_of_legs + self.total_no_of_free_coefficients # number of revolution parameters

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

    def post_processing_states(self, 
                design_parameter_vector : list, 
                bodies = mga_util.create_modified_system_of_bodies()):
                # bodies = environment_setup.create_simplified_system_of_bodies()):

        """
        Assuming no_of_gas == 6
        0 - departure_date
        1 - departure velocity
        2..9 - time of flights
        9..51 - free_coefficients
        51..57 - integer ga identifier 
        """
        # print("Design Parameters:", design_parameter_vector, "\n")

        central_body = 'Sun'

        #depart and target elements
        departure_elements = (self.depart_semi_major_axis, self.depart_eccentricity) 
        target_elements = (self.target_semi_major_axis, self.target_eccentricity) 

        # indexes
        time_of_flight_index = 2 + self.no_of_legs
        free_coefficient_index = time_of_flight_index + self.total_no_of_free_coefficients
        revolution_index = free_coefficient_index + self.no_of_legs

        ### INTEGER PART ###
        # number of revolutions
        number_of_revolutions = \
        [int(x) for x in design_parameter_vector[free_coefficient_index:revolution_index]]

        ### CONTINUOUS PART ###
        # departure date
        departure_date = design_parameter_vector[0]

        # departure velocity
        departure_velocity = design_parameter_vector[1] 

        # time of flight
        time_of_flights = design_parameter_vector[2:time_of_flight_index]

        # hodographic shaping free coefficients
        free_coefficients = design_parameter_vector[time_of_flight_index:free_coefficient_index]


        transfer_trajectory_object = mga_util.get_low_thrust_transfer_object(self.transfer_body_order,
                                                            time_of_flights,
                                                            departure_elements,
                                                            target_elements,
                                                            # self.system_of_bodies(),
                                                            bodies,
                                                            central_body,
                                                            no_of_free_parameters=self.no_of_free_parameters,
                                                            number_of_revolutions=number_of_revolutions)

        planetary_radii_sequence = np.zeros(self.no_of_gas)
        for i, body in enumerate(self.transfer_body_order[1:-1]):
            planetary_radii_sequence[i] = self.planetary_radii[body]

        swingby_periapses = np.array([planetary_radii_sequence[i] + self.swingby_altitude for i in
            range(self.no_of_gas)]) # defined depending on problem
        incoming_velocities = np.array([2000 for _ in range(self.no_of_gas)]) 

        # node times
        self.node_times = mga_util.get_node_times(self.transfer_body_order, departure_date, time_of_flights)

        # leg free parameters 
        leg_free_parameters = np.concatenate(np.array([np.append(number_of_revolutions[i],
            free_coefficients[i*(self.no_of_free_parameters*3):(i+1)*(self.no_of_free_parameters*3)]) for i
            in range(self.no_of_legs)])).reshape((self.no_of_legs, 1 + 3*
                self.no_of_free_parameters)) # added reshape

        # node free parameters
        node_free_parameters=  mga_util.get_node_free_parameters(self.transfer_body_order,
                swingby_periapses, incoming_velocities, departure_velocity=departure_velocity,
                arrival_velocity=self.arrival_velocity)

        transfer_trajectory_object.evaluate(self.node_times, leg_free_parameters, node_free_parameters)
        self.transfer_trajectory_object = transfer_trajectory_object
 
    def fitness(self, 
                design_parameter_vector : list, 
                bodies = mga_util.create_modified_system_of_bodies()):
                # bodies = environment_setup.create_simplified_system_of_bodies()):

        """
        Assuming no_of_gas == 6
        1 - departure_date
            1 - departure velocity
        2..9 - time of flights
        9..51 - free_coefficients
        51..57 - integer ga identifier 
        """
        # print("Design Parameters:", design_parameter_vector, "\n")

        # parameters
        central_body = 'Sun'

        #depart and target elements
        departure_elements = (self.depart_semi_major_axis, self.depart_eccentricity) 
        target_elements = (self.target_semi_major_axis, self.target_eccentricity) 

        # indexes
        time_of_flight_index = 2 + self.no_of_legs
        free_coefficient_index = time_of_flight_index + self.total_no_of_free_coefficients
        revolution_index = free_coefficient_index + self.no_of_legs

        ### INTEGER PART ###
        # number of revolutions
        number_of_revolutions = \
        [int(x) for x in design_parameter_vector[free_coefficient_index:revolution_index]]

        ### CONTINUOUS PART ###
        # departure date
        departure_date = design_parameter_vector[0]

        # departure velocity
        departure_velocity = design_parameter_vector[1] 

        # time of flight
        time_of_flights = design_parameter_vector[2:time_of_flight_index]

        # hodographic shaping free coefficients
        free_coefficients = design_parameter_vector[time_of_flight_index:free_coefficient_index]

        transfer_trajectory_object = mga_util.get_low_thrust_transfer_object(self.transfer_body_order,
                                                            time_of_flights,
                                                            departure_elements,
                                                            target_elements,
                                                            # self.system_of_bodies(),
                                                            bodies,
                                                            central_body,
                                                            no_of_free_parameters=self.no_of_free_parameters,
                                                            number_of_revolutions=number_of_revolutions)

        planetary_radii_sequence = np.zeros(self.no_of_gas)
        for i, body in enumerate(self.transfer_body_order[1:-1]):
            planetary_radii_sequence[i] = self.planetary_radii[body]

        swingby_periapses = np.array([planetary_radii_sequence[i] + self.swingby_altitude for i in
            range(self.no_of_gas)]) # defined depending on problem
        incoming_velocities = np.array([2000 for _ in range(self.no_of_gas)]) 

        # node times
        node_times = mga_util.get_node_times(self.transfer_body_order, departure_date, time_of_flights)
        # print(node_times)

        # leg free parameters 
        leg_free_parameters = np.concatenate(np.array([np.append(number_of_revolutions[i],
            free_coefficients[i*(self.no_of_free_parameters*3):(i+1)*(self.no_of_free_parameters*3)]) for i
            in range(self.no_of_legs)])).reshape((self.no_of_legs, 1 + 3*
                self.no_of_free_parameters)) # added reshape
        # print(leg_free_parameters)

        # node free parameters
        node_free_parameters=  mga_util.get_node_free_parameters(self.transfer_body_order,
                swingby_periapses, incoming_velocities, departure_velocity=departure_velocity,
                arrival_velocity=self.arrival_velocity)
        # print(node_free_parameters)

        try:
            transfer_trajectory_object.evaluate(node_times, leg_free_parameters, node_free_parameters)
            objective = transfer_trajectory_object.delta_v 
        except RuntimeError as e:
            print(str(e), "\n")#, "Objective increased by 10**16")
            objective = 10**16

        print('Fitness evaluated')
        return [objective]
