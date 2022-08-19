'''
Author: Sean Cowan
Purpose: MSc Thesis
Date Created: 26-07-2022

This module performs the optimization calculations using the help modules from mga-low-thrust-utilities.py and
pygmo-utilities.py
'''
###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import numpy as np
import os
import pygmo as pg
import dill
import multiprocessing as mp

# Tudatpy imports
import tudatpy
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.math import interpolators
from tudatpy.kernel.trajectory_design import shape_based_thrust
from tudatpy.kernel.trajectory_design import transfer_trajectory

# import mga_low_thrust_utilities as mga_util
import pygmo_island as isl
from pygmo_problem import MGALowThrustTrajectoryOptimizationProblem

current_dir = os.getcwd()
write_results_to_file = True
subdirectory = '/test_optimization_results/'

###########################################################################
# CREATE ENVIRONMENT ######################################################
###########################################################################

# Define settings for celestial bodies
bodies_to_create = ['Mercury',
                    'Venus',
                    'Earth',
                    'Mars',
                    'Jupiter',
                    'Saturn',
                    'Uranus',
                    'Neptune',
                    'Sun']

# Define coordinate system
global_frame_origin = 'SSB'
global_frame_orientation = 'ECLIPJ2000'
# Create body settings
body_settings = environment_setup.get_default_body_settings(bodies_to_create,
                                                            global_frame_origin,
                                                            global_frame_orientation)
# Create bodies
bodies = environment_setup.create_system_of_bodies(body_settings)
central_body = 'Sun'

########################################################################################
# DEFINE PARAMETERS AND PROBLEM ########################################################
########################################################################################

## definitions
julian_day = constants.JULIAN_DAY

## constraints
max_no_of_gas = 8

## trajectory information
departure_date = (-789.8117 - 0.5)  * julian_day
departure_velocity_magnitude = 2000 # m/s

begin_body = "Earth"
target_body = "Jupiter"

"""
transfer_body_order = ["Earth", "Mars", "Jupiter"]
"""
transfer_body_order = np.array([3, 0, 4, 0, 0, 0], dtype=int)
number_of_gas = len(transfer_body_order)

time_of_flight = np.array([500, 2500, 0, 0, 0, 0, 0]) * julian_day
tof_total = np.sum(time_of_flight)
# swingby_periapses=  np.array([50e6, 200e6])
# number_of_revolutions = np.array([0, 0])

#no_of_free_coefficients = 2
# velocity_coefficients = np.array([np.ones(6), np.ones(6)])

#constructing the design parameter vector
#velocity_coefficients = np.array([np.ones(6) for i in range(max_no_of_gas)]) #get all velocity coefficients

design_parameter_vector = np.zeros(17) #without periapses, revolutions, and free coefficients


"""
0: departure_date
1: departure_velocity_magnitude
2-9:time of flight
9-15:transfer_body_order
"""
design_parameter_vector[0] = departure_date
design_parameter_vector[1] = departure_velocity_magnitude
design_parameter_vector[2:9] = time_of_flight
design_parameter_vector[9:15] = transfer_body_order
# print(design_parameter_vector)
# print(type(transfer_body_order[0]))

freq = 2.0 * np.pi / tof_total
scale = 1.0 / tof_total

# mga_ltto_problem = \
# MGALowThrustTrajectoryOptimizationProblem(design_parameter_vector)
# print(mga_ltto_problem)

mga_low_thrust_problem = MGALowThrustTrajectoryOptimizationProblem('Jupiter', bodies_to_create)
prob = pg.problem(mga_low_thrust_problem)
#prob.c_tol = [0]*17


###########################################################################
# OPTIMIZE PROBLEM ########################################################
###########################################################################

parallel = True
if parallel:
    #my_population = pg.population(mga_low_thrust_problem, size = 100, seed=33333)
    if __name__ == '__main__':
        # mp.freeze_support()

        num_gen = 20
        pop_size = 1000

        # my_problem = pg.minlp_rastrigin(300, 60)
        my_problem = prob
        my_population = pg.population(my_problem, size=pop_size)
        my_algorithm = pg.algorithm(pg.gaco(gen=num_gen))
        my_island = pg.mp_island()
        # my_island = isl.userdefinedisland_v3()
        # my_island = isl.my_isl()
        # print(prob)
        archi = pg.archipelago(n=8, t=pg.unconnected(), algo = my_algorithm, pop=my_population, udi = my_island)
        print(archi)
        archi.evolve()
        print(archi)
        archi.wait()
        print(archi)
        #
        # champions_x = archi.get_champions_x()
        # print(champions_x[0])
        # best_trajectory = mga_low_thrust_problem.fitness(champions_x[0])
        # print(mga_low_thrust_problem.delta_v)
        # my_island.run_evolve(algo=my_algorithm, pop=my_population)
        #
        # Only try with island to reproduce problem
        # my_island = isl.my_isl()
        # my_island = pg.mp_island()
        # isl = pg.island(algo = my_algorithm, pop=my_population, udi = my_island)
        # isl.status
        # isl.evolve()
        # isl.status
        # my_island.shutdown_pool()

else:
    num_evol = 50
    pop_size = 100
    pop = pg.population(prob, size=pop_size)
    algo = pg.algorithm(pg.sga()) #multiple algorithms possible, sga, gaco, de
    # x_best = pop.get_x()
    # x_best[-1] = 0; x_best[-2] = 0
    # pop.push_back(x_best)

    fitness_list=  []
    population_list=  []
    for i in range(num_evol):
        pop = algo.evolve(pop)
        fitness_list.append(pop.get_f())
        population_list.append(pop.get_x())
        print(pop.champion_f)
        print('Evolving population; at generation ' + str(i))


    print(pop.champion_x)
    best_trajectory = mga_low_thrust_problem.fitness(pop.champion_x)

    state_history = mga_low_thrust_problem.get_states_along_trajectory(500)
    thrust_acceleration = mga_low_thrust_problem.get_inertial_thrust_accelerations_along_trajectory(500)
    node_times_list = mga_low_thrust_problem.get_node_times()

    node_times = {}
    for it, time in enumerate(node_times_list):
        node_times[it] = time

    print(node_times)

    if write_results_to_file:
        output_path = current_dir + subdirectory
    else:
        output_path = None

    if write_results_to_file:
        save2txt(state_history, 'state_history.dat', output_path)
        save2txt(thrust_acceleration, 'thrust_acceleration.dat', output_path)
        save2txt(node_times, 'node_times.dat', output_path)

    # # Select algorithm from pygmo, with one generation
    # algo = pg.algorithm(pg.de())
    # # Create pygmo problem
    # prob = pg.problem(current_shape_optimization_problem)
    # # Initialize pygmo population with 50 individuals
    # population_size = 50
    # pop = pg.population(prob, size=population_size)
    # # Set the number of evolutions
    # number_of_evolutions = 50
    # # Evolve the population recursively
    # fitness_list = []
    # population_list = []
    # 
    # fitness_list.append(pop.get_f())
    # population_list.append(pop.get_x())
    # 
    # for i in range(number_of_evolutions):
    #     # Evolve the population
    #     pop = algo.evolve(pop)
    #     # Store the fitness values for all individuals in a list
    #     fitness_list.append(pop.get_f())
    #     population_list.append(pop.get_x())
    #     print(pop.champion_f)
    #     print('Evolving population; at generation ' + str(i))
    # 
