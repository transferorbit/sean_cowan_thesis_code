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
    max_no_of_gas = 0
    # no_of_sequence_recursions = 1 if max_no_of_gas == 0 else max_no_of_gas # amount of different predefiend islands
    no_of_sequence_recursions = max_no_of_gas
    max_number_of_exchange_generations = 1 # amount of times it evolves
    # number_of_sequences_per_planet = [1]
    number_of_sequences_per_planet = [1 for _ in range(max_no_of_gas)]
    
    ## Specific parameters
    departure_planet = "Earth"
    arrival_planet = "Mars"
    free_param_count = 2
    num_gen = 2
    pop_size = 100
    assert pop_size > 62
    no_of_points = 1000
    bounds = [[8000, 0, 200, 0, 2e2, -10**4, 0],
            [8200, 0, 1200, 7000, 2e11, 10**4, 0]]
    print('Departure date bounds : [%s, %s]' %
            (time_conversion.julian_day_to_calendar_date(time_conversion.modified_julian_day_to_julian_day(bounds[0][0] + 51544.5)),
        time_conversion.julian_day_to_calendar_date(time_conversion.modified_julian_day_to_julian_day(bounds[1][0]) + 51544.5)))
    subdirectory = '/mgso_full_test'
    if os.path.exists(output_directory + subdirectory):
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

