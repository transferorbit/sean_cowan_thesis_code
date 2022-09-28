'''
Author: Sean Cowan
Purpose: MSc Thesis
Date Created: 14-09-2022

This module aims at implementing the algorithm that exchanges information and determines what
islands to pass to the archipelago
'''
###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import numpy as np
import os
import pygmo as pg
import multiprocessing as mp
import random

# Tudatpy imports
# Still necessary to implement most recent version of the code

import sys
sys.path.insert(0, "/Users/sean/Desktop/tudelft/tudat/tudat-bundle/build/tudatpy")

import tudatpy
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.math import interpolators
from tudatpy.kernel.trajectory_design import shape_based_thrust
from tudatpy.kernel.trajectory_design import transfer_trajectory


# import mga_low_thrust_utilities as mga_util
from pygmo_problem import MGALowThrustTrajectoryOptimizationProblem
from mga_low_thrust_utilities import transfer_body_order_conversion

current_dir = os.getcwd()
write_results_to_file = True

class manualTopology:

    def __init__(self) -> None:
        pass

    @staticmethod
    def create_island(transfer_body_order, free_param_count, bounds, num_gen, pop_size):
        mga_low_thrust_object = \
        MGALowThrustTrajectoryOptimizationProblem(transfer_body_order=transfer_body_order,
                no_of_free_parameters=free_param_count, bounds=bounds)
        problem = pg.problem(mga_low_thrust_object)
        algorithm = pg.algorithm(pg.gaco(gen=num_gen))
        return pg.island(algo=algorithm, prob=problem, size=pop_size, udi=pg.mp_island()), mga_low_thrust_object

    @staticmethod
    def create_random_transfer_body_order(arrival_planet, max_no_of_gas=6) -> list:

        # transfer_body_order.append(predefined_sequence) if predefined_sequence != [] else None #?
        # create random sequence of numbers with max_no_of_gas length
        sequence_length = random.randrange(0, max_no_of_gas)
        sequence_digits = [random.randrange(1, 6) for _ in range(sequence_length)]

        # transform that into transfer_body_order
        transfer_body_strings = transfer_body_order_conversion.get_transfer_body_list(sequence_digits)

        transfer_body_strings.append(arrival_planet)

        return transfer_body_strings

    def add_sequence_to_database(self):
        pass

    @staticmethod
    def remove_excess_planets(planet_list : list, departure_planet : str, arrival_planet : str) -> list:
        for it, i in enumerate(planet_list):
            if i == departure_planet:
                dep_index = it
            if i == arrival_planet:
                arr_index = it
        if arr_index < dep_index:
            number_of_truncations = len(planet_list) - dep_index - 1
            # print(number_of_truncations)
        else:
            number_of_truncations = len(planet_list) - arr_index - 1
            # print(number_of_truncations)
        for i in range(number_of_truncations):
            planet_list.pop()
        return planet_list

    # def get_leg_specific_dict
