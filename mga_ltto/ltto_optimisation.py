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
    import os
    import pygmo as pg
    import multiprocessing as mp
    import sys
    import numpy as np
    import warnings
    
    # If conda environment does not work
    # import sys
    # sys.path.insert(0, "/Users/sean/Desktop/tudelft/tudat/tudat-bundle/build/tudatpy")
    
    from tudatpy.kernel import constants
    
    current_dir = os.getcwd()
    sys.path.append(current_dir) # this only works if you run ltto and mgso while in the directory that includes those files
    from src.pygmo_problem import MGALowThrustTrajectoryOptimizationProblem, \
    MGALowThrustTrajectoryOptimizationProblemDSM, MGALowThrustTrajectoryOptimizationProblemOOA, \
    MGALowThrustTrajectoryOptimizationProblemOOADAAA
    import src.mga_low_thrust_utilities as util
    import src.manual_topology as topo
    from src.date_conversion import dateConversion
    
    
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
    no_of_points = 100 # this number affects the state_history and thrust_acceleration, thus the accuracy of the delivery mass

    ## General parameters
    manual_tof_bounds = None
    dynamic_shaping_functions = False
    dynamic_bounds = True
    write_results_to_file = True
    manual_base_functions = False
    zero_revs = False
    objectives = ['dv'] #dv, tof, pmf, dmf
    
####################################################################
# LTTO Problem Setup ###############################################
####################################################################


    subdirectory=  '/EMJ_gen50pop300_FAN_fullddate'
    free_param_count = 2
    num_gen = 50
    pop_size = 300
    cpu_count = os.cpu_count() // 2# not very relevant because differnent machines + asynchronous
    # cpu_count = len(os.sched_getaffinity(0))
    print(f'CPUs used : {cpu_count}')
    number_of_islands = cpu_count # // 2 to only access physical cores.
    bound_names= ['Departure date [mjd2000]', 'Departure velocity [m/s]', 'Arrival velocity [m/s]',
                'Time of Flight [s]', 'Incoming velocity [m/s]', 'Swingby periapsis [m]',
                'Free coefficient [-]', 'Number of revolutions [-]']
    # bound_names= ['Departure date [mjd2000]', 'Departure velocity [m/s]', 'Arrival velocity [m/s]',
    #         'Time of Flight [s]', 'Incoming velocity [m/s]', 'Swingby periapsis [m]', r'DSM $\Delta V$ [m/s]',
    #         'Free coefficient [-]', 'Number of revolutions [-]']
    # bound_names= ['Departure date [mjd2000]', 'Departure velocity [m/s]', 'Arrival velocity [m/s]',
    #         'Time of Flight [s]', 'Incoming velocity [m/s]', 'Swingby periapsis [m]', 
    #               'Orbit orientation angle [rad]', 'Free coefficient [-]', 'Number of revolutions [-]']
    # bound_names= ['Departure date [mjd2000]', 'Departure velocity [m/s]', 'Departure in-plane angle [rad]', 
    #               'Departure out-of-plane angle [rad]', 'Arrival velocity [m/s]', 'Arrival in-plane angle [rad]', 
    #               'Arrival out-of-plane angle [rad]', 'Time of Flight [s]', 'Incoming velocity [m/s]', 
    #               'Swingby periapsis [m]', 'Orbit orientation angle [rad]', 'Free coefficient [-]', 
    #               'Number of revolutions [-]']

    ## MORANTE ##

    # transfer_body_order = ["Earth", "Venus"]
    #
    # departure_date = (10592.5, 11321.5)
    # departure_velocity = (1999.9999, 2000)
    # arrival_velocity = (1000,3000)
    # time_of_flight = (100, 400)
    # incoming_velocity = (100, 15000)
    # swingby_periapsis = (2e5, 2e8)
    # free_coefficient = (-3e4, 3e4)
    # number_of_revs = (0, 4)
    # zero_revs = True
    # Isp = 3000
    # m0 = 360

    ## ENGLANDER ##
    # transfer_body_order = ["Earth", "Earth", "Venus", "Venus", "Mercury", "Mercury"]
    # Isp = 3200
    # m0 = 1300
    # dep_date_lb = dateConversion(calendar_date='2009, 8, 1').date_to_mjd()
    # dep_date_ub = dateConversion(calendar_date='2012, 4, 27').date_to_mjd()
    # bounds = [[dep_date_lb, 0, 0, 10, 100, 2e5, 0, -10**4, 0],
    #     [dep_date_ub, 1925, 500, 10000, 10000, 2e7, 2000, 10**4, 6]]

    ## FAN ##

    # transfer_body_order = ["Earth", "Jupiter"]
    #
    # jd2000_dep_date_lb = 61420 - 51544.5
    # jd2000_dep_date_ub = 63382 - 51544.5
    #
    # departure_date=  (jd2000_dep_date_lb, jd2000_dep_date_ub)
    # departure_velocity = (0, 0)
    # arrival_velocity = (0, 0)
    # time_of_flight = (2500, 3000)
    # incoming_velocity = (100, 15000)
    # swingby_periapsis = (2e5, 2e8)
    # free_coefficient = (-3e4, 3e4)
    # number_of_revs = (0, 4)
    # Isp = 3200 #guess
    # m0 = 1300 #guess
    
    transfer_body_order = ["Earth", "Mars", "Jupiter"]

    jd2000_dep_date_lb = 61420 - 51544.5
    jd2000_dep_date_ub = 63382 - 51544.5

    departure_date=  (jd2000_dep_date_lb, jd2000_dep_date_ub)
    departure_velocity = (0, 0)
    arrival_velocity = (0, 0)
    time_of_flight = (200, 3000)
    incoming_velocity = (100, 15000)
    swingby_periapsis = (2e5, 2e8)
    free_coefficient = (-3e4, 3e4)
    number_of_revs = (0, 4)
    Isp = 3200 #guess
    m0 = 1300 #guess


    # bounds = [[departure_date[0], departure_velocity[0], departure_inplane_angle[0],
    #            departure_outofplane_angle[0], arrival_velocity[0], arrival_inplane_angle[0],
    #            arrival_outofplane_angle[0], time_of_flight[0], incoming_velocity[0],
    #            swingby_periapsis[0], orbit_ori_angle[0], free_coefficient[0], number_of_revs[0]], 
    #           [departure_date[1], departure_velocity[1], departure_inplane_angle[1],
    #            departure_outofplane_angle[1], arrival_velocity[1], arrival_inplane_angle[1],
    #            arrival_outofplane_angle[1], time_of_flight[1], incoming_velocity[1],
    #            swingby_periapsis[1], orbit_ori_angle[1], free_coefficient[1], number_of_revs[1]]]
    
    # bounds = [[departure_date[0], departure_velocity[0], arrival_velocity[0], time_of_flight[0],
    #            incoming_velocity[0], swingby_periapsis[0], orbit_ori_angle[0], free_coefficient[0],
    #            number_of_revs[0]], 
    #           [departure_date[1], departure_velocity[1], arrival_velocity[1], time_of_flight[1],
    #            incoming_velocity[1], swingby_periapsis[1], orbit_ori_angle[1], free_coefficient[1],
    #            number_of_revs[1]]]

    bounds = [[departure_date[0], departure_velocity[0], arrival_velocity[0], time_of_flight[0],
               incoming_velocity[0], swingby_periapsis[0], free_coefficient[0], number_of_revs[0]], 
              [departure_date[1], departure_velocity[1], arrival_velocity[1], time_of_flight[1],
               incoming_velocity[1], swingby_periapsis[1], free_coefficient[1], number_of_revs[1]]]

    caldatelb = dateConversion(mjd2000=bounds[0][0]).mjd_to_date()
    caldateub = dateConversion(mjd2000=bounds[1][0]).mjd_to_date()
    print(f'Departure date bounds : [{caldatelb}, {caldateub}]')
    mga_sequence_characters = util.transfer_body_order_conversion.get_mga_characters_from_list(
            transfer_body_order)

    mga_low_thrust_problem = \
    MGALowThrustTrajectoryOptimizationProblem(transfer_body_order=transfer_body_order,
              no_of_free_parameters=free_param_count, 
              bounds=bounds, 
              manual_base_functions=manual_base_functions, 
              objectives=objectives, 
              Isp=Isp,
              m0=m0,
              no_of_points=no_of_points,
              zero_revs=zero_revs,
              dynamic_shaping_functions=dynamic_shaping_functions,
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
    elif len(objectives) == 2:
        algorithm = pg.algorithm(pg.nsga2(gen=1))
        modulus_pop = pop_size % 4
        if modulus_pop != 0:
            pop_size += (4-modulus_pop)
            print(f'Population size not divisible by 4, increased by {4-modulus_pop}')
    else:
        raise RuntimeError('An number of objectives was provided that is not permitted')

    # my_island = pg.mp_island()
    print('Creating archipelago')
    archi = pg.archipelago(n=number_of_islands, algo = algorithm, prob=prob, pop_size = pop_size)#, udi = my_island)

    ## New
    list_of_x_dicts, list_of_f_dicts, champions_x, \
    champions_f, ndf_x, ndf_f = topo.manualTopology.perform_evolution(archi,
                        number_of_islands,
                        num_gen,
                        objectives)

###########################################################################
# Post processing #########################################################
###########################################################################
    if write_results_to_file:
        topo.manualTopology.create_files(type_of_optimisation='ltto',
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
                            bound_names=bound_names)
