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
    import sys
    
    # If conda environment does not work
    # import sys
    # sys.path.insert(0, "/Users/sean/Desktop/tudelft/tudat/tudat-bundle/build/tudatpy")
    
    from tudatpy.kernel import constants
    
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
    Number of evolutions requires redefining the islands which requires more time on single thread.
    Population size; unknown
    """
    
    current_dir = os.getcwd()
    output_directory = current_dir + '/pp_ltto'
    julian_day = constants.JULIAN_DAY
    seed = 421
    no_of_points = 500

    write_results_to_file = False
    manual_base_functions = False
    mo_optimisation = True
    
####################################################################
# LTTO Problem Setup ###############################################
####################################################################

    ## General parameters
    dynamic_shaping_functions = False


    # testing problem functionality
    # transfer_body_order = ["Earth", "Earth", "Venus", "Venus", "Mercury", "Mercury"]
    # free_param_count = 2
    # num_gen = 100
    # pop_size = 500
    # no_of_points = 500
    # bounds = [[3300, 0, 50, 0, 2e2, -10**4, 0],
    #         [4600, 3000, 500, 9000, 2e9, 10**4, 2]]
    # subdirectory=  '/tudat_example_EEVVYY_2'

    subdirectory=  '/EVEMJ_verification_ndf'
    Isp = 3000
    m0 = 360
    transfer_body_order = ["Earth", "Venus", "Earth", "Mars", "Jupiter"]
    free_param_count = 2
    num_gen = 2
    pop_size = 100
    cpu_count = os.cpu_count() # not very relevant because differnent machines + asynchronous
    number_of_islands = cpu_count
    bound_names= ['Departure date [mjd2000]', 'Departure velocity [m/s]', 'Arrival velocity [m/s]',
            'Time of Flight [s]', 'Incoming velocity [m/s]', 'Swingby periapsis [m]', 
            'Free coefficient [-]', 'Number of revolutions [-]']
    bounds = [[10592.5, 1999.999999, 0, 100, 0, 2e5, -10**4, 0],
            [11321.5, 2000, 7000, 1500, 7000, 2e11, 10**4, 1]]
    caldatelb = dateConversion(bounds[0][0]).mjd_to_date()
    caldateub = dateConversion(bounds[1][0]).mjd_to_date()
    print(f'Departure date bounds : [{caldatelb}, {caldateub}]')
    
    # verification Gondelach
    # transfer_body_order = ["Earth", "Mars"]
    # free_param_count = 2
    # num_gen = 100
    # pop_size = 30
    # num_gen = 1
    # pop_size = 100
    # no_of_points = 500
    # bounds = [[7304*julian_day, 0, 500*julian_day, -10**4, 2],
    #         [10225*julian_day, 0, 2000*julian_day, 10**4, 2]]
    # subdirectory = '/verification/gondelach_N2'

    # transfer_body_order = ["Earth", "Mars"]
    # free_param_count = 0
    # num_gen = 30
    # pop_size = 500
    # no_of_points = 500
    # bounds = [[10025, 0, 1050, 0, 2e2, -10**4, 2], #0fp
    #         [10025, 0, 1050, 7000, 2e11, 10**4, 2]]
    # bounds = [[9985, 0, 1100, 0, 2e2, -10**4, 2], #2fp
    #         [9985, 0, 1100, 7000, 2e11, 10**4, 2]]
    # subdirectory = '/verification/ltto_0fp_planstates'
    
    # TGRRoegiers p.116
    # mjd_depart_lb = 58849
    # mjd_depart_ub = 61770
    # mjd_2000=  51544.5
    # bounds = [[(mjd_depart_lb-mjd_2000)*julian_day, 1, 500*julian_day, -10**6, 1],
    #         [(mjd_depart_ub-mjd_2000)*julian_day, 1, 2000*julian_day, 10**6, 4]]
    # subdirectory = '/verification/roegiers_test5/'
    
    
    # bounds = [[9265*julian_day, 1, 1070*julian_day, -10**6, 0],
    #         [9265*julian_day, 1, 1070*julian_day, 10**6, 6]]
    # bounds = [[9300*julian_day, 150, 1185*julian_day, -10**6, 2],
    #         [9300*julian_day, 150, 1185*julian_day, 10**6, 4]]
    # subdirectory = '/verification/verification_results/'

    # Nathan
    # transfer_body_order = ["Earth", "Mars"]
    # free_param_count = 2
    # num_gen = 15
    # pop_size = 300
    # no_of_points = 500
    #
    # bounds = [[10000, 0, 50*julian_day, -10**4, 0], #seconds since J2000
    #         [10100, 0, 1000*julian_day, 10**4, 6]]
    # subdirectory=  '/nathan_2fp'
    
    # validation

    mga_sequence_characters = util.transfer_body_order_conversion.get_mga_characters_from_list(
            transfer_body_order)

    mga_low_thrust_problem = \
    MGALowThrustTrajectoryOptimizationProblem(transfer_body_order=transfer_body_order,
            no_of_free_parameters=free_param_count, 
            bounds=bounds, 
            manual_base_functions=manual_base_functions, 
            mo_optimisation=mo_optimisation, 
            Isp=Isp,
            m0=m0,
            no_of_points=no_of_points)

    prob = pg.problem(mga_low_thrust_problem)
    
    mp.freeze_support()
    # number_of_islands = cpu_count

###################################################################
# LTTO Optimisation ###############################################
###################################################################

    my_population = pg.population(prob, size=pop_size, seed=seed)
    if not mo_optimisation:
        algorithm = pg.algorithm(pg.sga(gen=1))
    else:
        algorithm = pg.algorithm(pg.nsga2(gen=1))
        modulus_pop = pop_size % 4
        if modulus_pop != 0:
            pop_size += (4-modulus_pop)
            print(f'Population size not divisible by 4, increased by {4-modulus_pop}')

    # my_island = pg.mp_island()
    print('Creating archipelago')
    archi = pg.archipelago(n=number_of_islands, algo = algorithm, prob=prob, pop_size = pop_size)#, udi = my_island)

    ## New
    list_of_x_dicts, list_of_f_dicts, champions_x, \
    champions_f, ndf_x, ndf_f = topo.manualTopology.perform_evolution(archi,
                        number_of_islands,
                        num_gen,
                        mo_optimisation)
    ## Old
    # list_of_f_dicts = []
    # list_of_x_dicts = []
    # for i in range(num_gen): # step between which topology steps are executed
    #     print('Evolving Gen : %i / %i' % (i, num_gen))
    #     archi.evolve()
    #     # archi.status
    #     # archi.wait_check()
    #     champs_dict_per_gen = {}
    #     champ_f_dict_per_gen = {}
    #     for j in range(number_of_islands):
    #         champs_dict_per_gen[j] = archi.get_champions_x()[j]
    #         champ_f_dict_per_gen[j] = archi.get_champions_f()[j]
    #     list_of_x_dicts.append(champs_dict_per_gen)
    #     list_of_f_dicts.append(champ_f_dict_per_gen)
    #     # champion_fitness_dict_per_gen[i] = archi.get_champions_f()
    #     archi.wait_check()
    # print('Evolution finished')
    # # print(list_of_f_dicts, list_of_x_dicts)



###########################################################################
# Post processing #########################################################
###########################################################################
    print(ndf_f, ndf_x)

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
