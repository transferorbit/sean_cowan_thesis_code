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
    import sys
    sys.path.insert(0, "/Users/sean/Desktop/tudelft/tudat/tudat-bundle/build/tudatpy")
    
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
    subdirectory = '/morante_testing_MO'
    max_no_of_gas = 3
    no_of_sequence_recursions = 2
    number_of_sequences_per_planet = [5 for _ in range(max_no_of_gas)]
    elitist_fraction = (1/3)
    manual_base_functions = False
    leg_exchange = True
    seed = 421
    possible_ga_planets = ["Venus", "Earth", "Mars"] # optional
    mo_optimisation = True
    # possible_ga_planets = None
    
    ## Specific parameters
    Isp = 3000
    m0 = 360
    departure_planet = "Earth"
    arrival_planet = "Jupiter"
    free_param_count = 2
    # num_gen = 2
    # pop_size = 20
    num_gen = 50
    pop_size = 200 # multiple of 12 makes the division also divisible by 4 if elitist fraction is 1/3
    # assert pop_size > 62 #only for gaco
    no_of_points = 1000
    bound_names= ['Departure date [mjd2000]', 'Departure velocity [m/s]', 'Arrival velocity [m/s]',
        'Time of Flight [s]', 'Incoming velocity [m/s]', 'Swingby periapsis [m]', 
        'Free coefficient [-]', 'Number of revolutions [-]']
    bounds = [[10592.5, 1999.999999, 0, 100, 0, 2e5, -10**4, 0],
            [11321.5, 2000, 7000, 1500, 7000, 2e11, 10**4, 1]]

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
                                possible_ga_planets=possible_ga_planets,
                                max_no_of_gas=max_no_of_gas,
                                no_of_sequence_recursions=no_of_sequence_recursions,
                                elitist_fraction=elitist_fraction,
                                number_of_sequences_per_planet=number_of_sequences_per_planet,
                                seed=seed,
                                write_results_to_file=write_results_to_file,
                                manual_base_functions=manual_base_functions,
                                leg_exchange=leg_exchange,
                                top_x_sequences =20,
                                mo_optimisation=mo_optimisation)

