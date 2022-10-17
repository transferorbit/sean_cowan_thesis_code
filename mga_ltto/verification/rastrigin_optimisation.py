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
    # from tudatpy.kernel.interface import spice
    # spice.load_standard_kernels()
    
    current_dir = os.getcwd()
    output_directory = current_dir + '/verification_results'
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
# LTTO Problem Setup ###############################################
####################################################################

    bound_names= ['Departure date [mjd2000]', 'Departure velocity [m/s]', 'Time of Flight [s]',
            'Free coefficient [-]', "Number of revolutions [-]"]

    # verification optimisation paramters
    # test minlp optimization
    my_problem = pg.minlp_rastrigin(5,5) 
    # my_problem=  pg.rastrigin(5)
    num_gen = 300
    pop_size = 3000
    no_of_points = 500
    subdirectory = '/minlp_rastrigin'

    prob = my_problem #optimisation verification
    mp.freeze_support()
    cpu_count = os.cpu_count() # not very relevant because differnent machines + asynchronous
    number_of_islands = cpu_count
###################################################################
# LTTO Optimisation ###############################################
###################################################################

    my_population = pg.population(prob, size=pop_size, seed=seed)
    my_algorithm = pg.algorithm(pg.sga(gen=1))
    # my_island = pg.mp_island()
    print('Creating archipelago')
    archi = pg.archipelago(n=number_of_islands, algo = my_algorithm, prob=prob, pop_size = pop_size)#, udi = my_island)

    list_of_f_dicts = []
    list_of_x_dicts = []
    for i in range(num_gen): # step between which topology steps are executed
        print('Evolving Gen : %i / %i' % (i, num_gen))
        archi.evolve()
        # archi.status
        # archi.wait_check()
        champs_dict_per_gen = {}
        champ_f_dict_per_gen = {}
        for j in range(number_of_islands):
            champs_dict_per_gen[j] = archi.get_champions_x()[j]
            champ_f_dict_per_gen[j] = archi.get_champions_f()[j]
        list_of_x_dicts.append(champs_dict_per_gen)
        list_of_f_dicts.append(champ_f_dict_per_gen)
        # champion_fitness_dict_per_gen[i] = archi.get_champions_f()
        archi.wait_check()
    print('Evolution finished')
    # print(list_of_f_dicts, list_of_x_dicts)



###########################################################################
# Post processing #########################################################
###########################################################################

    champions = archi.get_champions_x()
    champion_fitness = archi.get_champions_f()

    # for optimisation verification
    print(champions)
    print(champion_fitness)

    champions_dict = {}
    champion_fitness_dict = {}
    thrust_acceleration_list=  []
    for i in range(number_of_islands):
            current_island_f = {}
            current_island_x = {}
            unique_identifier = "/island_" + str(i) + "/"
            for j in range(num_gen):
                current_island_f[j] = list_of_f_dicts[j][i]
                current_island_x[j] = list_of_x_dicts[j][i]
            save2txt(current_island_f, 'champ_f_per_gen.dat', output_directory
                    + subdirectory + unique_identifier)
            save2txt(current_island_x, 'champs_per_gen.dat', output_directory +
                    subdirectory + unique_identifier)
