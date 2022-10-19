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
    import sys
    
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
    
    sys.path.append('../mga_ltto/src/')
    from pygmo_problem import MGALowThrustTrajectoryOptimizationProblem
    import mga_low_thrust_utilities as util
    import manual_topology as topo
    from date_conversion import dateConversion
    
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
    subdirectory = '/new_island_test'
    max_no_of_gas = 4
    no_of_sequence_recursions = 4
    number_of_sequences_per_planet = [2 for _ in range(max_no_of_gas)]
    elitist_fraction = 0.1
    manual_base_functions = False
    leg_exchange = True
    seed = 421
    
    ## Specific parameters
    Isp = 3200
    m0 = 1300
    departure_planet = "Earth"
    arrival_planet = "Jupiter"
    free_param_count = 2
    num_gen = 2
    pop_size = 100
    # assert pop_size > 62 #only for gaco
    no_of_points = 1000
    bounds = [[6000, 0, 200, 0, 2e2, -10**4, 0],
            [6200, 0, 1200, 7000, 2e11, 10**4, 4]]

    caldatelb = dateConversion(bounds[0][0]).mjd_to_date()
    caldateub = dateConversion(bounds[1][0]).mjd_to_date()
    print(f'Departure date bounds : [{caldatelb}, {caldateub}]')
    
    topo.run_mgso_optimisation(departure_planet=departure_planet,
                                arrival_planet=arrival_planet,
                                free_param_count=free_param_count,
                                Isp=Isp,
                                m0=m0,
                                num_gen=num_gen,
                                pop_size=pop_size,
                                no_of_points=no_of_points,
                                bounds=bounds,
                                output_directory=output_directory,
                                subdirectory=subdirectory,
                                max_no_of_gas=max_no_of_gas,
                                no_of_sequence_recursions=no_of_sequence_recursions,
                                elitist_fraction=elitist_fraction,
                                number_of_sequences_per_planet=number_of_sequences_per_planet,
                                seed=seed,
                                write_results_to_file=write_results_to_file,
                                manual_base_functions=manual_base_functions,
                                leg_exchange=leg_exchange)

