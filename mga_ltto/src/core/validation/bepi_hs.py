'''
Author: Sean Cowan
Purpose: MSc Thesis
Date Created: 14-02-2023

This module optimises the free coefficients of a hodographic shaping leg of a mission of choice (DAWN, BEPICOLOMBO, or DART).
'''

if __name__ == '__main__': 

    # General 
    import os
    import math
    import pygmo as pg
    import multiprocessing as mp
    import sys
    import numpy as np
    import warnings
    import argparse

    sys.path.append('/Users/sean/Desktop/tudelft/thesis/code/mga_ltto/src/') # this only works if you run ltto and mgso while in the directory that includes those files

    # Tudatpy
    from tudatpy.kernel import constants
    from tudatpy.kernel.astro import element_conversion

    #Local
    import core.multipurpose.pygmo_problem as prob
    import core.multipurpose.mga_low_thrust_utilities as util
    import core.multipurpose.create_files as post
    import core.multipurpose.perform_evolution as evol
    from misc.date_conversion import dateConversion

    current_dir = os.getcwd()
    bepi_state = np.loadtxt('/Users/sean/Desktop/tudelft/thesis/code/mga_ltto/pp_validation/bepi_data_test20/bepi_state.dat')
# If conda environment does not work
# import sys
# sys.path.insert(0, "/Users/sean/Desktop/tudelft/tudat/tudat-bundle/build/tudatpy")

#Argparse
    parser = argparse.ArgumentParser(description='This file runs a singe leg validation run')
    parser.add_argument('--id', default='0', dest='id', action='store', required=False)
# args = parser.parse_args(['--id'])
    args = parser.parse_args()
# print(args)
# print(args.id)
    id = args.id

###################################################################
# Simulation ######################################################
###################################################################

    # Convert bepi data from cartesian to keplerian
    central_body_mu = 1.3271244e20 # m^3 / s^2

    # departure_kep_state = list(bepi_data_kep[0, :])
    # arrival_kep_state = list(bepi_data_kep[-1, :])
    print(bepi_state[0, 1:])
    print(bepi_state[-1, 1:])

    # define output directory
    output_directory = current_dir + '/pp_validation'
    julian_day = constants.JULIAN_DAY
    seed = 421
    no_of_points = 100 # this number affects the state_history and thrust_acceleration, thus the accuracy of the delivery mass

## General parameters
    manual_tof_bounds = None
    dynamic_bounds = {'time_of_flight' : False,
            'orbit_ori_angle' : False,
            'swingby_outofplane' : False,
            'swingby_inplane' : False,
            'shaping_function' : False}
    write_results_to_file = True
    manual_base_functions = False
    topology_type = None #float or None
    zero_revs = False
    objectives = ['dv'] #dv, tof, pmf, dmf

    free_param_count = 1
    # num_gen = 50
    # pop_size = 500
    num_gen = 1
    pop_size = 5
    # cpu_count = os.cpu_count() // 2# not very relevant because different machines + asynchronous
    cpu_count = 1
# cpu_count = len(os.sched_getaffinity(0))
    print(f'CPUs used : {cpu_count}')
    number_of_islands = cpu_count # // 2 to only access physical cores.

    subdirectory = f'/bepi_hs_test{id}'

    transfer_body_order = ['Depart', 'Arriva']
    # begin_date = dateConversion(calendar_date="2018, 10, 20")
    # begin_epoch = begin_date.date_to_mjd() * 86400
    #
    # initial_pos_vector = []
    # final_pos_vector = []
    #
    # end_date = dateConversion(calendar_date="2020, 4, 10")
    # end_epoch = end_date.date_to_mjd() * 86400

    #Continuous low-thrust leg
    # begin_epoch = 6.3955*10**8
    # end_epoch = 6.4004*10**8
    # departure_date = begin_epoch
    
    # it 3 (based on esa article)
    begin_date = dateConversion(calendar_date="2018, 11, 20")
    begin_epoch = begin_date.date_to_mjd(time=[11, 33, 00]) * 86400 
    end_date = dateConversion(calendar_date="2018, 11, 20")
    end_epoch = begin_date.date_to_mjd(time=[16, 33, 00]) * 86400 


    departure_date = begin_epoch
    time_of_flight = end_epoch - begin_epoch
    print(time_of_flight)
    number_of_revs = 0
    Isp = 3200 #guess
    m0 = 1300 #guess


    free_coefficient = (-1e-3, 1e-3)
    bounds = [[free_coefficient[0]], [free_coefficient[1]]]

    # caldatelb = dateConversion(mjd2000=bounds[0][0]).mjd_to_date()
    # caldateub = dateConversion(mjd2000=bounds[1][0]).mjd_to_date()
    # print(f'Departure date bounds : [{begin_epoch.mjd_to_date()}, {end_epoch.mjd_to_date()}]')

    mga_low_thrust_problem = \
    prob.MGALowThrustTrajectoryOptimizationProblemSingleLeg(transfer_body_order,
                                    departure_date,
                                    time_of_flight,
                                    number_of_revs,
                                    no_of_free_parameters=free_param_count, 
                                    bounds=bounds, 
                                    manual_base_functions=manual_base_functions, 
                                    objectives=objectives, 
                                    Isp=Isp,
                                    m0=m0,
                                    no_of_points=no_of_points,
                                    zero_revs=zero_revs,
                                    dynamic_bounds=dynamic_bounds,
                                    manual_tof_bounds=manual_tof_bounds,
                                    bepi_state=bepi_state)

    prob = pg.problem(mga_low_thrust_problem)

    mp.freeze_support()
# number_of_islands = cpu_count

###################################################################
# LTTO Optimisation ###############################################
###################################################################

    my_optimiser = pg.nlopt(solver='neldermead')
    algorithm = pg.algorithm(my_optimiser)
    my_optimiser.maxeval = 2000
    my_optimiser.ftol_rel = 0.001
    my_optimiser.maxtime = 300

    # my_population = pg.population(prob, size=pop_size, seed=seed)
    # if len(objectives) == 1:
    #     algorithm = pg.algorithm(pg.sga(gen=1))
    #     # algorithm.set_verbosity(1)
    # elif len(objectives) == 2:
    #     algorithm = pg.algorithm(pg.nsga2(gen=1))
    #     modulus_pop = pop_size % 4
    #     if modulus_pop != 0:
    #         pop_size += (4-modulus_pop)
    #         print(f'Population size not divisible by 4, increased by {4-modulus_pop}')
    # else:
    #     raise RuntimeError('An number of objectives was provided that is not permitted')

    if isinstance(topology_type, float) and 0 < topology_type <= 1:
        topology = pg.fully_connected(w=topology_type)
    elif topology_type == None:
        topology = pg.unconnected()
    else:
        raise RuntimeError("The type of topology given is not permitted")

# my_island = pg.mp_island()
    print('Creating archipelago')
    archi = pg.archipelago(n=number_of_islands, t=topology, algo = algorithm, prob=prob, pop_size = pop_size)#, udi = my_island)

## New
    list_of_x_dicts, list_of_f_dicts, champions_x, \
    champions_f, ndf_x, ndf_f = evol.perform_evolution(archi,
                        number_of_islands,
                        num_gen,
                        objectives)

###########################################################################
# Post processing #########################################################
###########################################################################
    if write_results_to_file:
        post.create_files(type_of_optimisation='ltto',
                            number_of_islands=number_of_islands,
                            island_problem=mga_low_thrust_problem,
                            champions_x=champions_x,
                            champions_f=champions_f,
                            list_of_f_dicts=list_of_f_dicts,
                            list_of_x_dicts=list_of_x_dicts,
                            ndf_f=ndf_f,
                            ndf_x=ndf_x,
                            no_of_points=no_of_points,
                            Isp=Isp, 
                            m0=m0,
                            output_directory=output_directory,
                            subdirectory=subdirectory,
                            free_param_count=free_param_count,
                            num_gen=num_gen,
                            pop_size=pop_size,
                            cpu_count=cpu_count,
                            bounds=bounds,
                            archi=archi)

