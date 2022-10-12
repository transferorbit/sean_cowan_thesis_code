'''
Author: Sean Cowan
Purpose: MSc Thesis
Date Created: 11-10-2022

This module performs unit tests for the mgso optimisation process.
'''

if __name__ == '__main__': #to prevent this code from running if this file is not the source file.
# https://stackoverflow.com/questions/419163/what-does-if-name-main-do

###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################
    
    # General imports
    import numpy as np
    import os
    import pygmo as pg
    import multiprocessing as mp
    import shutil
    
    # If conda environment does not work
    # import sys
    # sys.path.insert(0, "/Users/sean/Desktop/tudelft/tudat/tudat-bundle/build/tudatpy")
    
    import tudatpy
    from tudatpy.io import save2txt
    from tudatpy.kernel import constants
    from tudatpy.kernel.astro import time_conversion
    from tudatpy.kernel.numerical_simulation import propagation_setup
    from tudatpy.kernel.numerical_simulation import environment_setup
    from tudatpy.kernel.math import interpolators
    from tudatpy.kernel.astro import element_conversion
    from tudatpy.kernel.trajectory_design import shape_based_thrust
    from tudatpy.kernel.trajectory_design import transfer_trajectory
    from tudatpy.kernel.interface import spice
    spice.load_standard_kernels()
    
    
    from pygmo_problem import MGALowThrustTrajectoryOptimizationProblem
    import mga_low_thrust_utilities as util
    import manual_topology as topo
    
    current_dir = os.getcwd()
    output_directory = current_dir + '/pp_mgso'
    write_results_to_file = False
    
###########################################################################
# General parameters ######################################################
###########################################################################
    
    """
    Number of generations increases the amount of time spent in parallel computing the different islands
    Number of evolutions requires redefining the islands which requires more time on single thread.
    Population size; unknown
    """
    
    julian_day = constants.JULIAN_DAY
    seed = 421
    
####################################################################
# MGSO Problem Setup ###############################################
####################################################################

    max_no_of_gas = 0
    no_of_sequence_recursions = max_no_of_gas
    max_number_of_exchange_generations = 1 # amount of times it evolves
    number_of_sequences_per_planet = [1 for _ in range(max_no_of_gas)]
    
    ## Specific parameters
    departure_planet = "Earth"
    arrival_planet = "Jupiter"
    free_param_count = 2
    num_gen = 1
    pop_size = 100
    assert pop_size > 62
    no_of_points = 1000
    bounds = [[9000, 0, 200, 0, 2e2, -10**4, 0],
            [9200, 0, 1200, 7000, 2e11, 10**4, 4]]
    
    topo.run_mgso_optimisation(departure_planet=departure_planet,
                                arrival_planet=arrival_planet,
                                free_param_count=free_param_count,
                                num_gen=num_gen,
                                pop_size=pop_size,
                                no_of_points=no_of_points,
                                bounds=bounds,
                                output_directory=output_directory,
                                subdirectory=subdirectory,
                                max_no_of_gas=max_no_of_gas,
                                no_of_sequence_recursions=no_of_sequence_recursions,
                                max_number_of_exchange_generations=max_number_of_exchange_generations,
                                number_of_sequences_per_planet=number_of_sequences_per_planet,
                                seed=seed)


