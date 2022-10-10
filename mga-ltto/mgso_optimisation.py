'''
Author: Sean Cowan
Purpose: MSc Thesis
Date Created: 26-07-2022

This module performs the optimization calculations using the help modules from mga-low-thrust-utilities.py and
pygmo-utilities.py
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
    write_results_to_file = True
    
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
    
    # bodies_to_create = ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn"]
    # central_body_mu = 1.3271244e20 # m^3 / s^2
    #
    # planet_kep_states = []
    # for i in bodies_to_create:
    #     current_cartesian_elements = spice.get_body_cartesian_state_at_epoch(i,
    #             "Sun",
    #             "ECLIPJ2000",
    #             'None',
    #             bounds[0][0])
    #     planet_kep_states.append(element_conversion.cartesian_to_keplerian(current_cartesian_elements,
    #         central_body_mu))
    #
    # print(planet_kep_states)

####################################################################
# MGSO Problem Setup ###############################################
####################################################################
    
    """
    ITBS - Initial Target Body Sequence
    PTBS - Predefined Target Body Sequence
    MGSA - Multiple Gravity Assist Sequence
    """
    
    # mgso General parameters
    max_no_of_gas = 2
    no_of_sequence_recursions = max_no_of_gas # amount of different predefiend islands
    max_number_of_exchange_generations = 1 # amount of times it evolves
    # number_of_sequences_per_planet = 4
    number_of_sequences_per_planet = [3 for _ in range(max_no_of_gas)]
    # number_of_sequences_per_planet = [1 for i in range(max_no_of_gas)]
    
    ## Specific parameters
    departure_planet = "Earth"
    arrival_planet = "Jupiter"
    free_param_count = 2
    num_gen = 20
    pop_size = 500
    assert pop_size > 62
    no_of_points = 1000
    bounds = [[1000, 0, 50, -10**4, 0],
            [1200, 0, 2000, 10**4, 6]]
    # print('Departure date bounds : [%d, %d]' %
    #         (time_conversion.julian_day_to_calendar_date(1000),
    #     time_conversion.julian_day_to_calendar_date(1200)))
    subdirectory = '/mgso_full_test'
    shutil.rmtree(output_directory + subdirectory)
    # subdirectory = ''

    # num_gen = 1
    # pop_size = 100
    # no_of_points = 1000
    # bounds = [[10000*julian_day, 100, 50*julian_day, -10**6, 0],
    #         [10000*julian_day, 100, 2000*julian_day, 10**6, 6]]
    # subdirectory = '/island_testing/'
    
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

