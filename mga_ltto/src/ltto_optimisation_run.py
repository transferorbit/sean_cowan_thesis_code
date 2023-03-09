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
    
    # General 
    import os
    import math
    import pygmo as pg
    import multiprocessing as mp
    import sys
    import numpy as np
    import warnings
    import argparse
    
    # Tudatpy
    from tudatpy.kernel import constants

    #Local
    import core.multipurpose.pygmo_problem as prob
    import core.multipurpose.mga_low_thrust_utilities as util
    import core.multipurpose.create_files as post
    import core.multipurpose.perform_evolution as evol
    from misc.date_conversion import dateConversion
    
    # If conda environment does not work
    # import sys
    # sys.path.insert(0, "/Users/sean/Desktop/tudelft/tudat/tudat-bundle/build/tudatpy")
    
    #Argparse
    parser = argparse.ArgumentParser(description='This file runs an LTTO process')
    parser.add_argument('--id', default='0', dest='id', action='store', required=False)
    # args = parser.parse_args(['--id'])
    args = parser.parse_args()
    # print(args)
    # print(args.id)
    id = args.id

    current_dir = os.getcwd()
    sys.path.append(current_dir) # this only works if you run ltto and mgso while in the directory that includes those files

    
    
###########################################################################
# General parameters ######################################################
###########################################################################
    
    """
    Number of generations increases the amount of time spent in parallel computing the different islands
    Number of evolutions requires redefining the islands which requires more time on single thread.
    Population size; unknown
    """
    
    output_directory = current_dir + '/pp_ltto'
    julian_day = constants.JULIAN_DAY
    seed = 421
    # seed = 331249
    no_of_points = 100 # this number affects the state_history and thrust_acceleration, thus the accuracy of the delivery mass

    ## General parameters
    manual_tof_bounds = None
    dynamic_bounds = {'time_of_flight' : False,
            'orbit_ori_angle' : False,
            'swingby_outofplane' : False,
            'swingby_inplane' : False,
            'shaping_function' : False}
    write_results_to_file = False
    manual_base_functions = False
    topology_type = None #float or None
    zero_revs = False
    objectives = ['dv'] #dv, tof, pmf, dmf
    
####################################################################
# LTTO Problem Setup ###############################################
####################################################################


    # subdirectory_list = ['/EEMJ_g300p1200_test90', '/EEMJ_g300p1200_test90', '/EEMJ_g300p1200_test90',
    #                      '/EEMJ_g300p1200_test90', '/EEMJ_g300p1200_test90', '/EEMJ_g300p1200_test90',
    #                      '/EEMJ_g300p1200_test90', '/EEMJ_g300p1200_test90']

    free_param_count = 1
    num_gen = 10
    pop_size = 100
    cpu_count = os.cpu_count() // 2# not very relevant because different machines + asynchronous
    # cpu_count = len(os.sched_getaffinity(0))
    print(f'CPUs used : {cpu_count}')
    number_of_islands = cpu_count # // 2 to only access physical cores.

    #manual no_of_runs
    # no_of_runs = 1
    # departure_date_range = (61200, 61600) #MJD
    # ddate_step = (departure_date_range[1] - departure_date_range[0]) / no_of_runs
    # ddate_lb = departure_date_range[0]

    #manual ddate_step
    # ddate_step = 60
    # ddate_step = 400
    
    # departure_date_range = (62000, 62400) # EJ #MJD
    # departure_date_range = (62400, 62800) # EJ #MJD
    # departure_date_range = (62800, 63200) # EJ #MJD
    
    # departure_date_range = (61600, 62000) # EMJ #MJD
    # departure_date_range = (62000, 62400) # EMJ #MJD
    # departure_date_range = (62400, 62800) # EMJ #MJD
    
    # departure_date_range = (61200, 61600) # EEMJ #MJD
    # departure_date_range = (61600, 62000) # EEMJ #MJD
    # departure_date_range = (62000, 62400) # EEMJ #MJD
    
    # departure_date_range = (61600, 62000) # EEEMJ #MJD
    # departure_date_range = (62000, 62400) # EEEMJ #MJD
    # departure_date_range = (62400, 62800) # EEEMJ #MJD
    
    # no_of_runs = math.ceil((departure_date_range[1] - departure_date_range[0]) / ddate_step)
    # ddate_lb = departure_date_range[0]

    no_of_runs = 1

    for i in range(no_of_runs):
        #create various subdirectory names
        # # subdirectory = subdirectory_list[i]
        #
        # #create the departure date variation
        # ddate_ub = ddate_lb + ddate_step
        # departure_date = (ddate_lb - 51544.5, ddate_ub - 51544.5)
        # print(f'Current departure date window: [{ddate_lb}, {ddate_ub}]')
        #
        # # subdirectory = \
        # # f'/EEEMJ_lb{departure_date_range[0]}_ub{departure_date_range[1]}_test114/{int(ddate_lb)}_{int(ddate_ub)}'
        # # print(f'Current save directory: {subdirectory}')
        #
        # ddate_lb = ddate_ub

        subdirectory = f'/EEEMJ_optcharsalgo_test{id}'

        ## FAN ##

        # transfer_body_order = ["Earth", "Jupiter"]
        # 
        # # jd2000_dep_date_lb = 61420 - 51544.5
        # # jd2000_dep_date_ub = 63382 - 51544.5
        # # 
        # # departure_date=  (jd2000_dep_date_lb, jd2000_dep_date_ub)

        # # departure_date=  (62654 - 51544.5 - 30, 62654 - 51544.5 + 30)
        # departure_velocity = (0, 0)
        # departure_inplane_angle = (0, 0)
        # departure_outofplane_angle = (0, 0)
        # arrival_velocity = (0, 0)
        # arrival_inplane_angle = (0, 0)
        # arrival_outofplane_angle = (0, 0)
        # time_of_flight = (100, 4500)
        # incoming_velocity = (0, 5000)
        # swingby_periapsis = (2e5, 1e9)
        # orbit_ori_angle = (0, 2 * np.pi)
        # swingby_inplane_angle = (0, 2 * np.pi)
        # swingby_outofplane_angle = (-np.pi / 4, np.pi / 4)
        # free_coefficient = (-1e4, 1e4)
        # number_of_revs = (0, 2)
        # Isp = 3200 #guess
        # m0 = 1300 #guess
        # manual_tof_bounds = [[500], [4500]]

        # transfer_body_order = ["Earth", "Mars", "Jupiter"]

        # # jd2000_dep_date_lb = 61420 - 51544.5
        # # jd2000_dep_date_ub = 63382 - 51544.5
        # # departure_date = (jd2000_dep_date_lb, jd2000_dep_date_ub)

        # # departure_date=  (61872 - 51544.5 - 30, 61872 - 51544.5  + 30) #10328 from paper
        # departure_velocity = (0, 0)
        # departure_inplane_angle = (0, 0)
        # departure_outofplane_angle = (0, 0)
        # arrival_velocity = (0, 0)
        # arrival_inplane_angle = (0,0)
        # arrival_outofplane_angle = (0, 0)
        # time_of_flight = (100, 4500)
        # incoming_velocity = (0, 5000)
        # swingby_periapsis = (2e5, 1e9)
        # orbit_ori_angle = (0, 2 * np.pi)
        # swingby_inplane_angle = (0, 2 * np.pi)
        # swingby_outofplane_angle = (-np.pi / 4, np.pi / 4)
        # free_coefficient = (-1e4, 1e4)
        # number_of_revs = (0, 2)
        # Isp = 3200 #guess
        # m0 = 1300 #guess
        # manual_tof_bounds = [[100, 500], [1500, 4500]]

        # transfer_body_order = ["Earth", "Earth", "Mars", "Jupiter"]

        # # jd2000_dep_date_lb = 61420 - 51544.5
        # # jd2000_dep_date_ub = 63382 - 51544.5
        # # departure_date = (jd2000_dep_date_lb, jd2000_dep_date_ub)

        # # departure_date=  (61452 - 51544.5 - 30, 61452 - 51544.5  + 30) #10328 from paper
        # departure_velocity = (0, 0)
        # departure_inplane_angle = (0, 0)
        # departure_outofplane_angle = (0, 0)
        # arrival_velocity = (0, 0)
        # arrival_inplane_angle = (0, 0)
        # arrival_outofplane_angle = (0, 0)
        # time_of_flight = (100, 4500)
        # incoming_velocity = (0, 5000)
        # swingby_periapsis = (2e5, 1e9)
        # orbit_ori_angle = (0, 2 * np.pi)
        # swingby_inplane_angle = (0, 2 * np.pi)
        # swingby_outofplane_angle = (-np.pi / 4, np.pi / 4)
        # free_coefficient = (-1e4, 1e4)
        # number_of_revs = (0, 2)
        # Isp = 3200 #guess
        # m0 = 1300 #guess
        # manual_tof_bounds = [[20, 100, 500], [500, 1500, 4500]]

        transfer_body_order = ["Earth", "Earth", "Earth", "Mars", "Jupiter"]

        # jd2000_dep_date_lb = 61420 - 51544.5
        # jd2000_dep_date_ub = 63382 - 51544.5
        # departure_date = (jd2000_dep_date_lb, jd2000_dep_date_ub)

        departure_date=  (61872 - 51544.5 - 30, 61872 - 51544.5  + 30) #10328 from paper
        departure_velocity = (0, 0)
        departure_inplane_angle = (0, 0)
        departure_outofplane_angle = (0, 0)
        arrival_velocity = (0, 0)
        arrival_inplane_angle = (0, 0)
        arrival_outofplane_angle = (0, 0)
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
        manual_tof_bounds = [[20, 20, 100, 500], [500, 500, 1500, 4500]]

        #OptimAngles
        bounds = [[departure_date[0], time_of_flight[0], incoming_velocity[0],
                   swingby_periapsis[0], orbit_ori_angle[0], swingby_inplane_angle[0],
                   swingby_outofplane_angle[0], free_coefficient[0], number_of_revs[0]], 
                  [departure_date[1], time_of_flight[1], incoming_velocity[1],
                   swingby_periapsis[1], orbit_ori_angle[1], swingby_inplane_angle[1],
                   swingby_outofplane_angle[1], free_coefficient[1], number_of_revs[1]]]

        #AllAngles
        # bounds = [[departure_date[0], departure_velocity[0], departure_inplane_angle[0],
        #            departure_outofplane_angle[0], arrival_velocity[0], arrival_inplane_angle[0],
        #            arrival_outofplane_angle[0], time_of_flight[0], incoming_velocity[0],
        #            swingby_periapsis[0], orbit_ori_angle[0], swingby_inplane_angle[0],
        #            swingby_outofplane_angle[0], free_coefficient[0], number_of_revs[0]], 
        #           [departure_date[1], departure_velocity[1], departure_inplane_angle[1],
        #            departure_outofplane_angle[1], arrival_velocity[1], arrival_inplane_angle[1],
        #            arrival_outofplane_angle[1], time_of_flight[1], incoming_velocity[1],
        #            swingby_periapsis[1], orbit_ori_angle[1], swingby_inplane_angle[1],
        #            swingby_outofplane_angle[1], free_coefficient[1], number_of_revs[1]]]

        caldatelb = dateConversion(mjd2000=bounds[0][0]).mjd_to_date()
        caldateub = dateConversion(mjd2000=bounds[1][0]).mjd_to_date()
        print(f'Departure date bounds : [{caldatelb}, {caldateub}]')
        mga_sequence_characters = util.transfer_body_order_conversion.get_mga_characters_from_list(
                transfer_body_order)

        mga_low_thrust_problem = \
        prob.MGALowThrustTrajectoryOptimizationProblemOptimAngles(transfer_body_order=transfer_body_order,
                  no_of_free_parameters=free_param_count, 
                  bounds=bounds, 
                  manual_base_functions=manual_base_functions, 
                  objectives=objectives, 
                  Isp=Isp,
                  m0=m0,
                  no_of_points=no_of_points,
                  zero_revs=zero_revs,
                  dynamic_bounds=dynamic_bounds,
                  manual_tof_bounds=manual_tof_bounds)

        prob = pg.problem(mga_low_thrust_problem)
        
        mp.freeze_support()
        # number_of_islands = cpu_count

###################################################################
# LTTO Optimisation ###############################################
###################################################################

        my_population = pg.population(prob, size=pop_size, seed=seed)
        if len(objectives) == 1:
            algorithm = pg.algorithm(pg.sga(gen=1))
            # algorithm.set_verbosity(1)
        elif len(objectives) == 2:
            algorithm = pg.algorithm(pg.nsga2(gen=1))
            modulus_pop = pop_size % 4
            if modulus_pop != 0:
                pop_size += (4-modulus_pop)
                print(f'Population size not divisible by 4, increased by {4-modulus_pop}')
        else:
            raise RuntimeError('An number of objectives was provided that is not permitted')

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
                                mga_sequence_characters=mga_sequence_characters,
                                output_directory=output_directory,
                                subdirectory=subdirectory,
                                free_param_count=free_param_count,
                                num_gen=num_gen,
                                pop_size=pop_size,
                                cpu_count=cpu_count,
                                bounds=bounds,
                                archi=archi)
