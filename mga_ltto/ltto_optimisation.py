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
    
    # If conda environment does not work
    # import sys
    # sys.path.insert(0, "/Users/sean/Desktop/tudelft/tudat/tudat-bundle/build/tudatpy")
    
    from tudatpy.kernel import constants
    
    current_dir = os.getcwd()
    sys.path.append(current_dir) # this only works if you run ltto and mgso while in the directory that includes those files
    from src.pygmo_problem import MGALowThrustTrajectoryOptimizationProblem
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
    no_of_points = 50 # this number affects the state_history and thrust_acceleration, thus the accuracy of the delivery mass

    ## General parameters
    dynamic_shaping_functions = False
    write_results_to_file = True
    manual_base_functions = False
    zero_revs = True
    objectives = ['mf', 'tof'] #dv, tof, mf
    
####################################################################
# LTTO Problem Setup ###############################################
####################################################################


    subdirectory=  '/EVEMJ_cpu4gen150pop300fp2_plot'
    Isp = 3000
    m0 = 360
    transfer_body_order = ["Earth", "Venus", "Earth", "Mars", "Jupiter"]
    free_param_count = 2
    num_gen = 10
    pop_size = 52
    cpu_count = os.cpu_count() // 2# not very relevant because differnent machines + asynchronous
    # cpu_count = len(os.sched_getaffinity(0))
    print(f'CPUs used : {cpu_count}')
    number_of_islands = cpu_count # // 2 to only access physical cores.
    bound_names= ['Departure date [mjd2000]', 'Departure velocity [m/s]', 'Arrival velocity [m/s]',
            'Time of Flight [s]', 'Incoming velocity [m/s]', 'Swingby periapsis [m]', 
            'Free coefficient [-]', 'Number of revolutions [-]']
    bounds = [[10592.5, 1999.999999, 5000, 100, 0, 2e5, -10**4, 0],
                [11321.5, 2000, 6000, 1500, 17000, 2e8, 10**4, 1]]
    # bounds = [[10592.5, 2000, 0, 100, 0, 2e5, -10**4, 0],
    #         [11321.5, 2000, 7000, 1500, 7000, 2e11, 10**4, 2]]
    caldatelb = dateConversion(bounds[0][0]).mjd_to_date()
    caldateub = dateConversion(bounds[1][0]).mjd_to_date()
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
            dynamic_shaping_functions=dynamic_shaping_functions)

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
                            bounds=bounds)
