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
    # import multiprocessing as mp
    import sys
    import argparse
    
    # If conda environment does not work
    # import sys
    # sys.path.insert(0, "/Users/sean/Desktop/tudelft/tudat/tudat-bundle/build/tudatpy")
    # sys.path.insert(0, "/Users/sean/Desktop/tudelft/tudat/tudat-bundle/tudatpy/tudatpy")
    
    from tudatpy.kernel import constants

    parser = argparse.ArgumentParser(description='This file runs an LTTO process')
    parser.add_argument('--id', default='0', dest='id', action='store', required=False) # id test number
    parser.add_argument('--spp', default='1', dest='spp', action='store', required=False) # sequences per planet
    parser.add_argument('--frac', default='0', dest='frac', action='store', required=False) #fraction sequences evaluated
    parser.add_argument('--ips', default='1', dest='ips', action='store', required=False) # islands per sequence
    # args = parser.parse_args(['--id'])
    args = parser.parse_args()
    # print(args)
    # print(args.id)
    id = args.id
    spp = int(args.spp)
    frac = float(args.frac)
    ips = int(args.ips)
    
    current_dir = os.getcwd()
    sys.path.append(current_dir) # this only works if you run ltto and mgaso while in the directory that includes those files
    # from pygmo_problem import MGALowThrustTrajectoryOptimizationProblem
    # import mga_low_thrust_utilities as util
    import src.manual_topology as topo
    from src.date_conversion import dateConversion
    
###########################################################################
# General parameters ######################################################
###########################################################################
    
    """
    Number of generations increases the amount of time spent in parallel computing the different islands
    Number of recursions requires redefining the islands which requires more time on single thread.
    Population size; unknown
    """

    output_directory = current_dir + '/pp_mgaso'
    julian_day = constants.JULIAN_DAY
    seed = 421
    no_of_points = 1000

    write_results_to_file = True

####################################################################
# MGASO Problem Setup ###############################################
####################################################################
    
    """
    ITBS - Initial Target Body Sequence
    PTBS - Predefined Target Body Sequence
    MGSA - Multiple Gravity Assist Sequence
    """
    subdirectory = f'/mgaso_testlocal{id}'
    max_no_of_gas = 3
    no_of_sequence_recursions = 2
    fraction_ss_evaluated = [frac for _ in range(no_of_sequence_recursions)]
    number_of_sequences_per_planet = [spp for _ in range(no_of_sequence_recursions)]
    elitist_fraction = 0.3
    islands_per_sequence = ips
    manual_base_functions = False
    dynamic_bounds = {'time_of_flight' : False,
            'orbit_ori_angle' : False,
            'swingby_outofplane' : False,
            'swingby_inplane' : False,
            'shaping_function' : False}
    manual_base_functions = False
    leg_exchange = False
    objectives = ['dv']
    zero_revs = False
    seed = 421
    # possible_ga_planets = ["Venus", "Earth", "Mars"] # optional
    possible_ga_planets = None
    
    ## Specific parameters
    departure_planet = "Earth"
    arrival_planet = "Jupiter"
    free_param_count = 2
    # num_gen = 2
    # pop_size = 52
    num_gen = 2
    pop_size = 100 # multiple of 12 makes the division also divisible by 4 if elitist fraction is 1/3
    # assert pop_size > 62 #only for gaco

    departure_date=  (61872 - 51544.5 - 30, 61872 - 51544.5  + 30) #10328 from paper
    time_of_flight = (100, 4500)
    incoming_velocity = (0, 5000)
    swingby_periapsis = (2e5, 1e9)
    orbit_ori_angle = (0, 2 * np.pi)
    swingby_inplane_angle = (0, 2 * np.pi)
    swingby_outofplane_angle = (-np.pi / 4, np.pi / 4)
    free_coefficient = (-1e4, 1e4)
    number_of_revs = (0, 2)
    Isp = 3200 #guess
    m0 = 1300 #guess

    bounds = [[departure_date[0], time_of_flight[0], incoming_velocity[0],
               swingby_periapsis[0], orbit_ori_angle[0], swingby_inplane_angle[0],
               swingby_outofplane_angle[0], free_coefficient[0], number_of_revs[0]], 
              [departure_date[1], time_of_flight[1], incoming_velocity[1],
               swingby_periapsis[1], orbit_ori_angle[1], swingby_inplane_angle[1],
               swingby_outofplane_angle[1], free_coefficient[1], number_of_revs[1]]]

    caldatelb = dateConversion(bounds[0][0]).mjd_to_date()
    caldateub = dateConversion(bounds[1][0]).mjd_to_date()
    print(f'Departure date bounds : [{caldatelb}, {caldateub}]')
    
    topo.run_mgaso_optimisation(departure_planet=departure_planet,
                                arrival_planet=arrival_planet,
                                free_param_count=free_param_count,
                                num_gen=num_gen,
                                pop_size=pop_size,
                                no_of_points=no_of_points,
                                bounds=bounds,
                                output_directory=output_directory,
                                subdirectory=subdirectory,
                                possible_ga_planets=possible_ga_planets,
                                max_no_of_gas=max_no_of_gas,
                                no_of_sequence_recursions=no_of_sequence_recursions,
                                islands_per_sequence=islands_per_sequence,
                                elitist_fraction=elitist_fraction,
                                number_of_sequences_per_planet=number_of_sequences_per_planet,
                                fraction_ss_evaluated=fraction_ss_evaluated,
                                seed=seed,
                                write_results_to_file=write_results_to_file,
                                manual_base_functions=manual_base_functions,
                                dynamic_bounds=dynamic_bounds,
                                leg_exchange=leg_exchange,
                                top_x_sequences=20,
                                objectives=objectives,
                                zero_revs=zero_revs)

