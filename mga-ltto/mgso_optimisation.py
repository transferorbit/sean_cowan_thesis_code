'''
Author: Sean Cowan
Purpose: MSc Thesis
Date Created: 26-07-2022

This module performs the optimization calculations using the help modules from mga-low-thrust-utilities.py and
pygmo-utilities.py
This module is the child of optimization_runs.py which performed the optimization until mid
September 2022
'''

if __name__ == '__main__': 

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
    
    from pygmo_problem import MGALowThrustTrajectoryOptimizationProblem
    import mga_low_thrust_utilities as util
    import manual_topology as topo
    
###########################################################################
# General parameters ######################################################
###########################################################################
    
    """
    Number of generations increases the amount of time spent in parallel computing the different islands
    Number of recursions requires redefining the islands which requires more time on single thread.
    Population size; unknown
    """

    current_dir = os.getcwd()
    output_directory = current_dir + '/pp_mgso'
    julian_day = constants.JULIAN_DAY

####################################################################
# MGSO Problem Setup ###############################################
####################################################################
    
    """
    ITBS - Initial Target Body Sequence
    PTBS - Predefined Target Body Sequence
    MGSA - Multiple Gravity Assist Sequence
    """
    write_results_to_file = True
    max_no_of_gas = 2
    no_of_sequence_recursions = 1
    number_of_sequences_per_planet = [2 for _ in range(max_no_of_gas)]
    manual_base_functions = False
    leg_exchange = False
    seed = 421
    
    ## Specific parameters
    departure_planet = "Earth"
    arrival_planet = "Jupiter"
    free_param_count = 2
    num_gen = 1
    pop_size = 100
    # assert pop_size > 62 #only for gaco
    no_of_points = 1000
    bounds = [[9000, 0, 200, 0, 2e2, -10**4, 0],
            [9200, 0, 1200, 7000, 2e11, 10**4, 4]]
    print('Departure date bounds : [%s, %s]' %
            (time_conversion.julian_day_to_calendar_date(time_conversion.modified_julian_day_to_julian_day(bounds[0][0] + 51544.5)),
        time_conversion.julian_day_to_calendar_date(time_conversion.modified_julian_day_to_julian_day(bounds[1][0]) + 51544.5)))
    subdirectory = '/2ga_test'
    if os.path.exists(output_directory + subdirectory):
        shutil.rmtree(output_directory + subdirectory)
    
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
                                # max_number_of_exchange_generations=max_number_of_exchange_generations,
                                number_of_sequences_per_planet=number_of_sequences_per_planet,
                                seed=seed,
                                write_results_to_file=write_results_to_file,
                                manual_base_functions=manual_base_functions,
                                leg_exchange=leg_exchange)

