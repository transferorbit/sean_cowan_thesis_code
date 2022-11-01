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
    
    
    sys.path.append('../mga_ltto/src/')
    from pygmo_problem import MGALowThrustTrajectoryOptimizationProblem
    import mga_low_thrust_utilities as util
    
    current_dir = os.getcwd()
    output_directory = current_dir + '/pp_ltto'
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

    bound_names= ['Departure date [mjd2000]', 'Departure velocity [m/s]', 'Arrival velocity [m/s]',
            'Time of Flight [s]', 'Incoming velocity [m/s]', 'Swingby periapsis [m]', 
            'Free coefficient [-]', 'Number of revolutions [-]']
    
    # testing problem functionality
    # transfer_body_order = ["Earth", "Earth", "Venus", "Venus", "Mercury", "Mercury"]
    # free_param_count = 2
    # num_gen = 100
    # pop_size = 500
    # no_of_points = 500
    # bounds = [[3300, 0, 50, 0, 2e2, -10**4, 0],
    #         [4600, 3000, 500, 9000, 2e9, 10**4, 2]]
    # subdirectory=  '/tudat_example_EEVVYY_2'
    transfer_body_order = ["Earth", "Mars", "Jupiter"]
    Isp = 3200
    m0 = 1300
    free_param_count = 2
    num_gen = 3
    pop_size = 100
    no_of_points = 500
    bounds = [[10000, 0, 0, 200, 300, 2e2, -10**4, 0],
            [12000, 0, 0, 1200, 7000, 2e9, 10**4, 2]]
    subdirectory=  '/EMJ_deliv_mass_test'

    mo_optimisation = False
    
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
            Isp=Isp, 
            m0=m0,
            no_of_points=no_of_points,
            mo_optimisation=mo_optimisation)

    prob = pg.problem(mga_low_thrust_problem)
    
    mp.freeze_support()
    cpu_count = os.cpu_count() # not very relevant because differnent machines + asynchronous
    # number_of_islands = cpu_count
    number_of_islands = 1

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

    champions_dict = {}
    champion_fitness_dict = {}
    thrust_acceleration_list=  []
    for i in range(number_of_islands):
        mga_low_thrust_problem.fitness(champions[i], post_processing=True)

        # State history
        state_history = \
        mga_low_thrust_problem.transfer_trajectory_object.states_along_trajectory(no_of_points)

        # Thrust acceleration
        thrust_acceleration = \
        mga_low_thrust_problem.transfer_trajectory_object.inertial_thrust_accelerations_along_trajectory(no_of_points)

        mass_history, delivery_mass = util.get_mass_propagation(thrust_acceleration, Isp, m0)

        # Node times
        node_times_list = mga_low_thrust_problem.node_times
        node_times_days_list = [i / constants.JULIAN_DAY for i in node_times_list]

        # print(state_history)
        # print(node_times_list)
        # print(node_times_days_list)

        node_times = {}
        for it, time in enumerate(node_times_list):
            node_times[it] = time

        # Auxiliary information
        delta_v = mga_low_thrust_problem.transfer_trajectory_object.delta_v
        delta_v_per_leg = mga_low_thrust_problem.transfer_trajectory_object.delta_v_per_leg
        number_of_legs = mga_low_thrust_problem.transfer_trajectory_object.number_of_legs
        number_of_nodes = mga_low_thrust_problem.transfer_trajectory_object.number_of_nodes
        time_of_flight = mga_low_thrust_problem.transfer_trajectory_object.time_of_flight

        if write_results_to_file:
            auxiliary_info = {}
            auxiliary_info['Number of legs,'] = number_of_legs 
            auxiliary_info['Number of nodes,'] = number_of_nodes 
            auxiliary_info['Total ToF (Days),'] = time_of_flight / 86400.0
            departure_velocity = delta_v
            for j in range(number_of_legs):
                auxiliary_info['Delta V for leg %s,'%(j)] = delta_v_per_leg[j]
                departure_velocity -= delta_v_per_leg[j]
            auxiliary_info['Delta V,'] = delta_v 
            auxiliary_info['Departure velocity,'] = departure_velocity
            auxiliary_info['MGA Sequence,'] = mga_sequence_characters
            auxiliary_info['Maximum thrust,'] = np.max([np.linalg.norm(j[1:]) for _, j in
                enumerate(thrust_acceleration.items())])
            auxiliary_info['Delivery mass,'] = delivery_mass

            unique_identifier = "/island_" + str(i) + "/"
            save2txt(state_history, 'state_history.dat', output_directory + subdirectory +
                    unique_identifier)
            save2txt(thrust_acceleration, 'thrust_acceleration.dat', output_directory + subdirectory +
                unique_identifier)
            save2txt(mass_history, 'mass_history.dat', output_directory + subdirectory + unique_identifier)
            save2txt(node_times, 'node_times.dat', output_directory + subdirectory + unique_identifier)
            save2txt(auxiliary_info, 'auxiliary_info.dat', output_directory + subdirectory +
                unique_identifier)

            current_island_f = {}
            current_island_x = {}
            for j in range(num_gen):
                current_island_f[j] = list_of_f_dicts[j][i]
                current_island_x[j] = list_of_x_dicts[j][i]
            save2txt(current_island_f, 'champ_f_per_gen.dat', output_directory
                    + subdirectory + unique_identifier)
            save2txt(current_island_x, 'champs_per_gen.dat', output_directory +
                    subdirectory + unique_identifier)

            champions_dict[i] = champions[i]
            champion_fitness_dict[i] = champion_fitness[i]
            # champions_dict[i][0] /= 86400.0
            # for x in range(len(transfer_body_order)-1):
            #     champions_dict[i][2+x] /= 86400.0
                # if x != 0:
                #     champions_dict[i][2 + len(transfer_body_order)-1 + (x-1)] = \
                #     10**champions_dict[i][2 + len(transfer_body_order)-1 + (x-1)]


    if write_results_to_file:

        optimisation_characteristics = {}
        optimisation_characteristics['Transfer body order,'] = mga_sequence_characters
        optimisation_characteristics['Free parameter count,'] = free_param_count
        optimisation_characteristics['Number of generations,'] = num_gen
        optimisation_characteristics['Population size,'] = pop_size
        optimisation_characteristics['CPU count,'] = cpu_count
        optimisation_characteristics['Number of islands,'] = number_of_islands
        for j in range(len(bounds[0])):
            for k in range(len(bounds)):
                if k == 0:
                    min = ' LB'
                else:
                    min = ' UB'
                optimisation_characteristics[bound_names[j] + min + ','] = bounds[k][j]
        # optimisation_characteristics['Bounds'] = bounds
        
        # This should be for the statistics, we already save all information per
        unique_identifier = "/champions/"
        save2txt(champion_fitness_dict, 'champion_fitness.dat', output_directory + subdirectory +
                unique_identifier)
        save2txt(champions_dict, 'champions.dat', output_directory + subdirectory +
                unique_identifier)
        unique_identifier = ""
        save2txt(optimisation_characteristics, 'optimisation_characteristics.dat', output_directory +
                subdirectory + unique_identifier)
