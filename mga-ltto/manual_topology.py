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
        problem = \
            pg.problem(MGALowThrustTrajectoryOptimizationProblem(transfer_body_order=transfer_body_order,
                no_of_free_parameters=free_param_count, bounds=bounds))
        algorithm = pg.algorithm(pg.gaco(gen=num_gen))
        return pg.island(algo=algorithm, prob=problem, size=pop_size, udi=pg.mp_island())

    @staticmethod
    def create_random_transfer_body_order(arrival_planet, max_no_of_gas=6):

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

