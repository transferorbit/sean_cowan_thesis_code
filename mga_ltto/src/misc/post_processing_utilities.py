'''
Author: Sean Cowan
Purpose: MSc Thesis
Date Created: 31-01-2023

This file includes all post-processing utilities, plotting and local optimisation.
'''
###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General
import numpy as np
import os
import pygmo as pg
import multiprocessing as mp
import random

# Tudatpy
from tudatpy.io import save2txt
from tudatpy.kernel import constants

# Local
import core.multipurpose.pygmo_problem as prob
import core.multipurpose.mga_low_thrust_utilities as util

# import sys
# sys.path.insert(0, "/Users/sean/Desktop/tudelft/tudat/tudat-bundle-test/build/tudatpy")

def perform_local_optimisation(dir_of_dir=None, 
                               no_of_islands=4,
                               max_eval=2000,
                               max_time=300,
                               set_verbose=False,
                               print_results=False):
    """
    dir_of_dir : str
    A directory as a string containing champions, islands, and optimisation_characteristics
    no_of_islands : int
    An integer representing the number of islands that were used for the optimisation
    """

    dir_list = dir_of_dir.split('/')
    fitness_value_dict = {}
    variable_value_dict = {}
    auxiliary_info = np.loadtxt(dir_of_dir + 'islands/island_0/auxiliary_info.dat', delimiter=',', dtype=str)
    optim_chars = np.loadtxt(dir_of_dir + 'optimisation_characteristics.dat', delimiter=',', dtype=str)

    auxiliary_info_dict = {}
    for i in range(len(auxiliary_info)):
        auxiliary_info_dict[auxiliary_info[i][0]] = auxiliary_info[i][1].replace('\t', '')

    optim_chars_dict = {}
    for i in range(len(optim_chars)):
        optim_chars_dict[optim_chars[i][0]] = optim_chars[i][1].replace('\t', '')

    transfer_body_order = util.transfer_body_order_conversion.get_mga_list_from_characters(
                                                                    auxiliary_info_dict['MGA Sequence'])
    free_param_count = int(optim_chars_dict["Free parameter count"])
    bounds = [[float(value) for key, value in optim_chars_dict.items() if key.split(' ')[-1] == 'LB' 
               and key.split(' ')[0] != 'Manual'],
              [float(value) for key, value in optim_chars_dict.items() if key.split(' ')[-1] == 'UB'
               and key.split(' ')[0] != 'Manual']]

    manual_base_functions = optim_chars_dict["Manual base functions"]
    objectives = [optim_chars_dict["Objective 1"]]
    no_of_points = int(optim_chars_dict["No of points"])
    zero_revs = True if optim_chars_dict["Zero revs"] == "True" else False

    dynamic_bounds = {key.split(' ')[1] : True if value == "True" else False for key, value in optim_chars_dict.items() if key.split(' ')[0] == 'Dynamic'}

    manual_tof_bounds = [[float(value) for key, value in optim_chars_dict.items() if key.split(' ')[0] == 'Manual' and
                            key.split(' ')[-1] == 'LB'], 
                         [float(value) for key, value in optim_chars_dict.items() if key.split(' ')[0] == 'Manual' and
                            key.split(' ')[-1] == 'UB']]
    # print('Bounds : ', bounds)

    Isp = int(auxiliary_info_dict["Isp"])
    m0 = int(auxiliary_info_dict["m0"])


    mga_low_thrust_problem = \
    prob.MGALowThrustTrajectoryOptimizationProblemAllAngles(transfer_body_order=transfer_body_order,
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

    # pop_size = int(optim_chars_dict['Population size'])
    # pop_size = no_of_islands # because deterministic method, so 1 island converges to same thing
    prob = pg.problem(mga_low_thrust_problem)

    my_optimiser = pg.nlopt(solver='neldermead')
    my_optimiser.maxeval = max_eval
    my_optimiser.ftol_rel = 0.001
    my_optimiser.maxtime = max_time

    my_algorithm = pg.algorithm(my_optimiser)
    if set_verbose:
        my_algorithm.set_verbosity(1)
    archi = pg.archipelago(t=pg.topology(pg.unconnected()))#, udi = my_island)

    for i in range(no_of_islands):
        my_population = pg.population(prob)
        fitness_value_dict[i] = np.loadtxt(dir_of_dir + f'islands/island_{i}/champ_f_per_gen.dat')
        variable_value_dict[i] = np.loadtxt(dir_of_dir + f'islands/island_{i}/champs_per_gen.dat')
        # print(fitness_value_dict[i][-1, 1:])
        my_population.push_back(variable_value_dict[i][-1, 1:], f=fitness_value_dict[i][-1, 1:])
        my_island = pg.island(algo=my_algorithm, pop=my_population)
        archi.push_back(my_island)

    initial_population = np.array(archi.get_champions_f()).copy()
    # mp.freeze_support()
    if print_results:
        print('Creating archipelago')
        print('Evolving ..')
    archi.evolve()
    archi.wait_check()

    final_population = np.array(archi.get_champions_f())
    final_population_dict = {it: fitness for it, fitness in enumerate(final_population)}

    final_population_x = np.array(archi.get_champions_x())
    final_population_x_dict = {it: dpv for it, dpv in enumerate(final_population_x)}

    diff_pop = np.array([initial_population[i] - final_population[i] for i in range(len(initial_population))])
    diff_pop_dict = {it: fitness for it, fitness in enumerate(diff_pop)}

    perc_pop = np.array([diff_pop[i] * 100 / initial_population[i] for i in range(len(initial_population))])
    perc_pop_dict = {it: fitness for it, fitness in enumerate(perc_pop)}

    if print_results:
        print("""
==================
Evolution finished
==================
              """)

        print('Initial fitness population', initial_population.reshape(1, 24))
        print('Final fitness population', final_population.reshape(1, 24))

        print('Absolute improvement :', diff_pop.reshape(1, 24))
        print('Improvement percentage :', perc_pop.reshape(1, 24))

        print('Initial minimum : ', np.min(initial_population))
        print('Final minimum : ', np.min(final_population))

    save2txt(final_population_dict, 'final_population.dat', dir_of_dir + 'local_optimisation/')
    save2txt(final_population_x_dict, 'final_population_x.dat', dir_of_dir + 'local_optimisation/')
    save2txt(diff_pop_dict, 'absolute_improvement.dat', dir_of_dir + 'local_optimisation/')
    save2txt(perc_pop_dict, 'percentage_improvement.dat', dir_of_dir + 'local_optimisation/')

    # return archi, initial_population

def hodographic_shaping_visualisation(dir=None , dir_of_dir=None, quiver=False, projection=None):
    """
    Function that plots the relevant data provided by the input parameters. Generally this function
    is designed for MGA trajectories, and is specifically adapted to use data created by multiple
    islands with PyGMO

    Parameters
    ----------
    dir : directory in which to look for state_history, thrust_acceleration, node_times, and auxiliary
    information
    dir_of_dir : directory in which to look for directories of various islands that all have
    to-be plotted data

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the 3D plot.
    ax : matplotlib.axes.Axes
        3D axis system used to plot the trajectory.
    """

    if dir == None and dir_of_dir == None:
        raise RuntimeError('No directory has been provided that contains to-be plotted data') 
    if dir != None and dir_of_dir != None:
        raise RuntimeError('Too many resources are provided. Please only pass a directory or \
                directory of directories to dir or dir_of_dir, respectively') 

    # TO-DO make dir_of_dir plots where you can take up to 8 islands and plot them in one figure

    if dir != None:
        input_directory = dir
    elif dir_of_dir != None:
        input_directory = dir_of_dir
    else:
        raise RuntimeError('Something went wrong, check source')

    state_history = np.loadtxt(dir + 'state_history.dat')
    node_times = np.loadtxt(dir + 'node_times.dat')
    auxiliary_info = np.loadtxt(dir + 'auxiliary_info.dat', delimiter=',', dtype=str)

    thrust_acceleration = np.loadtxt(dir + 'thrust_acceleration.dat')
    # print(auxiliary_info[0][0])
    # print(auxiliary_info[1])
    # aux_info_list= [i.replace('\t') for i in auxiliary_info]
    # print(aux_info_list)

    state_history_dict = {}
    thrust_history_dict = {}
    for i in range(len(state_history)):
        state_history_dict[state_history[i, 0]] = state_history[i,1:]
        thrust_history_dict[thrust_acceleration[i, 0]] = thrust_acceleration[i,1:]
    # print(state_history_dict)
    auxiliary_info_dict = {}
    for i in range(len(auxiliary_info)):
        auxiliary_info_dict[auxiliary_info[i][0]] = auxiliary_info[i][1].replace('\t', '')

    # print(node_times[0, 1])
    # print(state_history_dict)
    # print(node_times)
    fly_by_states = np.array([state_history_dict[node_times[i, 1]] for i in range(len(node_times))])
    # print(fly_by_states)
    if quiver==False:
        thrust_history_dict = None

    #Only plot relevant bodies
    tbo_list = \
    util.transfer_body_order_conversion.get_mga_list_from_characters(auxiliary_info_dict['MGA Sequence'])

    fig, ax = trajectory_3d(
            state_history_dict,
            vehicles_names=["Spacecraft"],
            central_body_name="SSB",
            # bodies=["Sun", "Mercury", "Venus", "Earth", "Mars"],
            bodies=["Sun"] + tbo_list,
            frame_orientation= 'ECLIPJ2000',
            thrust_history=thrust_history_dict,
            projection=projection)
    plt.figure(constrained_layout=True)
    # print(auxiliary_info_dict)
    ax.set_title(auxiliary_info_dict['MGA Sequence'], fontweight='semibold', fontsize=18)
    # ax.scatter(fly_by_states[0, 0] , fly_by_states[0, 1] , fly_by_states[0,
    #         2] , marker='+', color='yellow', label='Earth departure')
    # ax.scatter(fly_by_states[1, 0] , fly_by_states[1, 1] , fly_by_states[1,
    #         2] , marker='+', color='yellow', label='Mars arrival')
    # ax.scatter(fly_by_states[2, 0] , fly_by_states[2, 1] , fly_by_states[2,
    #         2] , marker='+', color='yellow', label='Mars fly-by')
    # ax.scatter(fly_by_states[3, 0] , fly_by_states[3, 1] , fly_by_states[3,
    #         2] , marker='+', color='yellow', label='Mars fly-by')

# Change the size of the figure
    fig.set_size_inches(8, 8)
# Show the plot
    # plt.show()

def objective_per_generation_visualisation(dir=None, 
                                           dir_of_dir=None, 
                                           no_of_islands=4,
                                           title=1,
                                           add_local_optimisation=False,
                                           save_fig=False):
    """
    Function that takes directories and returns the champions per generation for one or multiple islands

    Parameters
    ----------
    dir : str
        String representing directory of island_x
    dir_of_dir : str
        String representing directories that include the islands/ champions/ (and local_optimisation/) directories 

    Returns
    -------
    fig, ax : plt.figure
        Figure plotting the champion fitness per generation.
    """

    if dir != None:
        fitness_values = np.loadtxt(dir + 'champ_f_per_gen.dat')
        variable_values = np.loadtxt(dir + 'champs_per_gen.dat')
    if dir_of_dir != None:
        dir_list = dir_of_dir.split('/')
        fitness_value_dict = {}
        variable_value_dict = {}
        if add_local_optimisation:
            try:
                champion_fitness_local = np.loadtxt(dir_of_dir + "/local_optimisation/final_population.dat")[:, 1]
            except FileNotFoundError:
                raise RuntimeError("The local optimisation has not been performed properly yet.")
        for i in range(no_of_islands):
            fitness_value_dict[i] = np.loadtxt(dir_of_dir + f'islands/island_{i}/champ_f_per_gen.dat')
            variable_value_dict[i] = np.loadtxt(dir_of_dir + f'islands/island_{i}/champs_per_gen.dat')
            if add_local_optimisation:
                fitness_value_dict[i] = np.vstack([fitness_value_dict[i], [fitness_value_dict[i][-1, 0]+11,
                                                                           champion_fitness_local[i]]])
        auxiliary_info = np.loadtxt(dir_of_dir + 'islands/island_0/auxiliary_info.dat', delimiter=',', dtype=str)
            # champions_local[it] = np.loadtxt(root + dir + "/local_optimisation/final_population_x.dat")[:, 1] / 86400 + 51544.5
    auxiliary_info_dict = {}
    for i in range(len(auxiliary_info)):
        auxiliary_info_dict[auxiliary_info[i][0]] = auxiliary_info[i][1].replace('\t', '')

    fitness_array = np.zeros((len(fitness_value_dict[0][:,0]), no_of_islands))
    for i in range(no_of_islands):
        fitness_array[:, i] = fitness_value_dict[i][:, 1] #if not add_local_optimisation else fitness_value_dict[i][:, 1]

    min_deltav_value_before_local = np.min(fitness_array[:-1, :])
    min_deltav_value = np.min(fitness_array)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plt.figure(constrained_layout=True)
    for island in range(no_of_islands):
        ax.plot(fitness_value_dict[island][:,0], fitness_value_dict[island][:,1], label=f'Island {island}')
        # ax.legend()
    ax.axhline(min_deltav_value, c='k', linestyle='-', label=f'Minimum : {int(min_deltav_value / 1000)}')
    if add_local_optimisation:
        ax.axhline(min_deltav_value_before_local, c='k', linestyle='-', label=f'Minimum : {int(min_deltav_value_before_local / 1000)}')
    trans = transforms.blended_transform_factory(ax.get_yticklabels()[0].get_transform(), ax.transData)
    ax.text(0,min_deltav_value, "{:.0f}".format(min_deltav_value), color="red", transform=trans, 
            ha="right", va="center")
    if add_local_optimisation:
        ax.text(0,min_deltav_value_before_local, "{:.0f}".format(min_deltav_value_before_local),
                color="red", transform=trans, ha="right", va="center")
        ax.fill_between([len(fitness_value_dict[0]), len(fitness_value_dict[0])+10], 10000, 40000, color='lightskyblue')


    ax.set_ylabel(r'$\Delta V$ [m / s]')
    ax.set_xlabel(r'Generation count [-]')
    # ax.legend()
    ax.grid()
    # ax.set_title(auxiliary_info_dict['MGA Sequence'], fontweight='semibold', fontsize=18)
    title_elem = dir_of_dir.split('/')[title].split('_')
    ax.set_title(f'{title_elem[0]} — 24 islands')
    # ax.set_yscale('log')
    ax.set_ylim([10000, 40000])
    if save_fig:
        if add_local_optimisation:
            fig.savefig('figures/' + f"{dir_of_dir.split('/')[title]}_local.jpg", bbox_inches="tight")
        else:
            fig.savefig('figures/' + f"{dir_of_dir.split('/')[title]}.jpg", bbox_inches="tight")

def get_scattered_objectives(dir_of_dir_of_dir=None,
                             dir_of_dir=None,
                             title=2,
                             add_local_optimisation=False,
                             no_of_islands=4,
                             save_fig=False):

    if dir_of_dir_of_dir != None:
        # dir_of_dir_of_dir_list = dir_of_dir_of_dir.split('/')[2].split('_')
        for root, dirs, files in os.walk(dir_of_dir_of_dir):
            directory_list = dirs
            break
        grid_search = True
    elif dir_of_dir != None:
        dir_of_dir_list = dir_of_dir.split('/')
        directory_list = [dir_of_dir_list[2]]
        root = dir_of_dir_list[0] +"/" + dir_of_dir_list[1] + "/"
        grid_search = False

    color_list = \
    ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan',]
    
    champion_fitness = {}
    champion_fitness_local = {}
    champions = {}
    champions_local = {}
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    for it, dir in enumerate(directory_list):
        if grid_search:
            dir_list = dir.split('_')
            lb = dir_list[0]
            ub = dir_list[1]

        champions[it] = np.loadtxt(root + dir + "/champions/champions.dat")[:, 1] / 86400 + 51544.5
        champion_fitness[it] = np.loadtxt(root + dir + "/champions/champion_fitness.dat")[:, 1] 
        optim_chars = np.loadtxt(dir_of_dir + 'optimisation_characteristics.dat', delimiter=',', dtype=str)
        optim_chars_dict = {}
        for i in range(len(optim_chars)):
            optim_chars_dict[optim_chars[i][0]] = optim_chars[i][1].replace('\t', '')

        ax.scatter(champions[it], champion_fitness[it], c=color_list[it], label=f'{lb} - {ub} [JD]' if grid_search else '')
        if add_local_optimisation:
            champion_fitness_local[it] = np.loadtxt(root + dir + "/local_optimisation/final_population.dat")[:, 1]
            champions_local[it] = np.loadtxt(root + dir + "/local_optimisation/final_population_x.dat")[:, 1] / 86400 + 51544.5
            ax.scatter(champions_local[it], champion_fitness_local[it], c=color_list[it], marker="*")
            for it2 in range(no_of_islands):
                # ax.axvline((champions_local[it][it2] / 86400) + 51544.5, ymin=(champion_fitness_local[it][it2] - 15000) / 5000,
                #            ymax=(champion_fitness[it][it2] - 15000) / 5000, c=color_list[it], linestyle='--', linewidth=0.5)
                x_points = [champions[it][it2], champions_local[it][it2]]
                f_points = [champion_fitness[it][it2], champion_fitness_local[it][it2]]
                ax.plot(x_points, f_points, c=color_list[it], linestyle='--', linewidth=0.5)
        # ax.axvline(float(lb), c='k', linestyle='-', linewidth=1)
        # ax.axvline(float(ub), c='k', linestyle='-', linewidth=1)
        ax.set_ylabel(r'$\Delta V$ [m / s]')
        ax.set_xlabel(r'Departure Date [JD]')

    fitness_array = np.array(list(champion_fitness.values())) if not add_local_optimisation else \
                            np.array(list(champion_fitness_local.values())) 
    min_deltav_value = np.min(fitness_array)
    ax.axhline(min_deltav_value, c='k', linestyle='-', label=f'Minimum : {int(min_deltav_value)} [m/s]')
    trans = transforms.blended_transform_factory(
    ax.get_yticklabels()[0].get_transform(), ax.transData)
    ax.text(0,min_deltav_value, f"{int(min_deltav_value)}", color="red", transform=trans, 
            ha="right", va="center")
    ax.legend()
    ax.grid()
    if grid_search:
        title_elem = dir_of_dir_of_dir.split('/')[title].split('_')
        ax.set_title(f'{title_elem[0]} — {title_elem[1].replace("lb","")} - {title_elem[2].replace("ub","")} [JD] — {title_elem[3]}')
    else:
        title_elem = directory_list[0].split('_')
        lb = optim_chars_dict['Departure date [mjd2000] LB'] 
        lb_float = float(lb)+ 51544.5
        ub = optim_chars_dict['Departure date [mjd2000] UB'] 
        ub_float = float(ub)+ 51544.5
        ax.set_title(f'{title_elem[0]} — {lb} - {ub} [JD]')

    ax.set_ylim([20000, 35000])
    if not grid_search:
        ax.set_xlim([lb_float, ub_float])
    if save_fig:
        if add_local_optimisation:
            fig.savefig('figures/' + f"{dir_of_dir_of_dir.split('/')[title] if grid_search else dir_of_dir.split('/')[title]}_local.jpg", bbox_inches="tight")
        else:
            fig.savefig('figures/' + f"{dir_of_dir_of_dir.split('/')[title] if grid_search else dir_of_dir.split('/')[title]}.jpg", bbox_inches="tight")
    

def pareto_front(dir=None,
                 deltav_as_obj=False,
                 dmf_as_obj=False,
                 pmf_as_obj=False):

    if dir != None:
        input_directory = dir
    elif dir_of_dir != None:
        input_directory = dir_of_dir
    else:
        raise RuntimeError('Something went wrong, check source')


    #import data
    pareto_front = np.loadtxt(dir + 'pareto_front.dat')

    auxiliary_info = np.loadtxt(dir + 'auxiliary_info.dat', delimiter=',', dtype=str)
    auxiliary_info_dict = {}
    for i in range(len(auxiliary_info)):
        auxiliary_info_dict[auxiliary_info[i][0]] = auxiliary_info[i][1].replace('\t', '')

    # thrust_acceleration = np.loadtxt(dir + 'thrust_acceleration.dat')
    # Isp = auxiliary_info_dict['Isp']
    # m0 = auxiliary_info_dict['m0']
    # mass_history, delivery_mass, invalid_trajectory = get_mass_propagation(thrust_acceleration,
    #                                                                             Isp, m0)

    y_values = pareto_front[:, 1]
    if deltav_as_obj:
        y_values /= 1000 #to km/s
    tof_values = pareto_front[:, 2]
    tof_values /= 86400 # to days

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plt.figure(constrained_layout=True)
    ax.plot(tof_values, y_values)
    if deltav_as_obj:
        ax.set_ylabel(r'$\Delta V$ [km / s]')
        ax.set_ylim([150, 1000])
    elif dmf_as_obj:
        ax.set_ylabel(r'Delivery mass fraction $\frac{m_d}{m_0}$')
    elif pmf_as_obj:
        ax.set_ylabel(r'Propellant mass fraction $\frac{m_p}{m_0}$')
        # ax.set_ylim([0, 1])
    ax.set_xlabel(' ToF [days]' )
    # ax.set_xlim([350, 1000])
    # ax.legend()
    ax.set_title(auxiliary_info_dict['MGA Sequence'], fontweight='semibold', fontsize=18)
    ax.grid()

    # plt.show()

def thrust_propagation(dir=None):

    auxiliary_info = np.loadtxt(dir + 'auxiliary_info.dat', delimiter=',', dtype=str)
    auxiliary_info_dict = {}
    for i in range(len(auxiliary_info)):
        auxiliary_info_dict[auxiliary_info[i][0]] = auxiliary_info[i][1].replace('\t', '')

    thrust_acceleration = np.loadtxt(dir + 'thrust_acceleration.dat')
    thrust_norm = np.array([np.linalg.norm(thrust_acceleration[i, 1:4]) for i in 
                                           range(len(thrust_acceleration))])
    # print(np.min(thrust_norm))
    # print(np.max(thrust_norm))
    # print(min(thrust_norm))
    # print(max(thrust_norm))

    initial_epoch = thrust_acceleration[0, 0]
    # print(initial_epoch)
    time_history = thrust_acceleration[:, 0].copy()
    time_history -= initial_epoch
    time_history /= 86400
    # print(time_history[0:10])

    node_times = np.loadtxt(dir + 'node_times.dat')
    node_times[:, 1] -= initial_epoch
    node_times[:, 1] /= 86400
    # print(node_times)

    mga_sequence_characters = auxiliary_info_dict['MGA Sequence']
    list_of_mga_sequence_char = util.transfer_body_order_conversion.get_mga_list_from_characters(mga_sequence_characters)
    color_list = \
    ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan',]
    linestyle = '--'

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plt.figure(constrained_layout=True)
    ax.plot(time_history, thrust_acceleration[:, 1], linestyle=':', label=r'$a_r$')
    ax.plot(time_history, thrust_acceleration[:, 2], linestyle='-.', label=r'$a_\theta$')
    ax.plot(time_history, thrust_acceleration[:, 3], linestyle='--', label=r'$a_z$')
    ax.plot(time_history, thrust_norm, label='a')
    for i in range(len(list_of_mga_sequence_char)):
        ax.axvline(node_times[i, 1], c='k', linestyle='-', linewidth=0.5)
    # ax.axvline(node_times[0, 1], c=color_list[0], linestyle=linestyle,
    #            label=list_of_mga_sequence_char[0], ymin = np.min(thrust_norm), ymax =
    #            np.max(thrust_norm))
    ax.set_ylabel(r'Thrust norm [m / s$^2$]')
    ax.set_xlabel(' Epoch [days]' )
    # ax.set_ylim([150, 1000])
    # ax.set_xlim([350, 1000])
    # ax.set_yscale('log')
    ax.legend()
    ax.set_title(mga_sequence_characters, fontweight='semibold', fontsize=18)
    ax.grid()

def get_stats(dir_of_dir=None, 
              print_stats=False,
              plot_quantity=False,
              save_fig=False,
              quantity_to_analyse=None,
              title=1,
              bins=20):

    optim_chars = np.loadtxt(dir_of_dir + 'optimisation_characteristics.dat', delimiter=',', dtype=str)
    optim_chars_dict = {}
    for i in range(len(optim_chars)):
        optim_chars_dict[optim_chars[i][0]] = optim_chars[i][1].replace('\t', '')
    free_param_count = int(optim_chars_dict["Free parameter count"])

    auxiliary_info = np.loadtxt(dir_of_dir + 'islands/island_0/auxiliary_info.dat', delimiter=',', dtype=str)
    auxiliary_info_dict = {}
    for i in range(len(auxiliary_info)):
        auxiliary_info_dict[auxiliary_info[i][0]] = auxiliary_info[i][1].replace('\t', '')

    mga_chars = auxiliary_info_dict['MGA Sequence']
    no_of_legs = len(mga_chars) - 1
    no_of_gas = len(mga_chars) - 2
    time_of_flight_index = 8 + no_of_legs
    incoming_velocity_index = time_of_flight_index + no_of_gas
    swingby_altitude_index = incoming_velocity_index + no_of_gas
    orbit_ori_angle_index = swingby_altitude_index + no_of_gas
    swingby_inplane_angle_index = orbit_ori_angle_index + no_of_gas
    swingby_outofplane_angle_index = swingby_inplane_angle_index + no_of_gas
    free_coefficient_index = swingby_outofplane_angle_index + 3*free_param_count*no_of_legs
    revolution_index = free_coefficient_index + no_of_legs

    if quantity_to_analyse == 'dv':
        file_to_analyse = np.loadtxt(dir_of_dir + 'champions/champion_fitness.dat', delimiter='\t')
    else:
        file_to_analyse = np.loadtxt(dir_of_dir + 'champions/champions.dat', delimiter='\t')
    
    bound_names = ['Departure date [mjd2000]', 'Time of Flight [s]', 'Incoming velocity [m/s]', 
              'Swingby periapsis [m]', 'Orbit orientation angle [rad]', 'Swingby in-plane Angle [rad]', 
              'Swingby out-of-plane angle [rad]', 'Free coefficient [-]', 'Number of revolutions [-]']

    bound_quantity = {'dv' : 'Departure Velocity [m/s]', 
                      'dep_date' : bound_names[0],
                      'tof' : bound_names[1],
                      'inc_vel' : bound_names[2],
                      'swi_per' : bound_names[3],
                      'ooa' : bound_names[4],
                      'ip_ang' : bound_names[5],
                      'oop_ang' : bound_names[6],
                      'fc_legs' : bound_names[7],
                      'fc_axes' : bound_names[7],
                      'revs' : bound_names[8]}

    dict_of_indices = {'dv' : [1], 
                       'dep_date' : [1], 
                       'tof' : [8 + i for i in range(time_of_flight_index-8)], 
                       'inc_vel' : [time_of_flight_index + i for i in range(incoming_velocity_index -
                                                                            time_of_flight_index)],
                       'swi_per' : [incoming_velocity_index + i for i in range(swingby_altitude_index -
                                                                               incoming_velocity_index)],
                       'ooa' : [swingby_altitude_index + i for i in range(orbit_ori_angle_index -
                                                                          swingby_altitude_index)],
                       'ip_ang' : [orbit_ori_angle_index + i for i in range(swingby_inplane_angle_index -
                                                                            orbit_ori_angle_index)],
                       'oop_ang' : [swingby_inplane_angle_index + i for i in range(swingby_outofplane_angle_index -
                                                                                   swingby_inplane_angle_index)],
                       'fc_legs' : [swingby_outofplane_angle_index + i for i in range(free_coefficient_index -
                                                                                 swingby_outofplane_angle_index)],
                       'fc_axes' : [swingby_outofplane_angle_index + i for i in range(free_coefficient_index -
                                                                                 swingby_outofplane_angle_index)],
                       'revs' : [free_coefficient_index+ i for i in range(revolution_index - free_coefficient_index)]
                       }

    values_to_analyse = file_to_analyse[:, dict_of_indices[quantity_to_analyse]]

    if quantity_to_analyse == 'tof' or quantity_to_analyse == 'dep_date':
        values_to_analyse = values_to_analyse / 86400

    min_stats = np.min(values_to_analyse)
    mean_stats = np.mean(values_to_analyse)

    if quantity_to_analyse == 'fc_legs':
        values_to_analyse = np.abs(values_to_analyse)
        mean_stats = np.mean(values_to_analyse)

        values_to_analyse = [[values_to_analyse[j][i + p*3*free_param_count] for i in range(3) for j in
                             range(len(values_to_analyse))] for p in range(no_of_legs)]


    if quantity_to_analyse == 'fc_axes':
        values_to_analyse = np.abs(values_to_analyse)
        mean_stats = np.mean(values_to_analyse)
        radial_values_to_analyse = [values_to_analyse[j][i*3*free_param_count] for i in
                                    range(no_of_legs) for j in range(len(values_to_analyse))]
        normal_values_to_analyse = [values_to_analyse[j][1 + i*3*free_param_count] for i in
                                    range(no_of_legs) for j in range(len(values_to_analyse))]
        axial_values_to_analyse = [values_to_analyse[j][2 + i*3*free_param_count] for i in
                                    range(no_of_legs) for j in range(len(values_to_analyse))]
        values_to_analyse = [radial_values_to_analyse, normal_values_to_analyse, axial_values_to_analyse]

    dict = {f'min {quantity_to_analyse},' : min_stats, f'mean {quantity_to_analyse},' : mean_stats}
    if quantity_to_analyse != 'dv':
        max_stats = np.max(values_to_analyse)
        dict = {f'min {quantity_to_analyse},' : min_stats, f'mean {quantity_to_analyse},' : mean_stats,
                f'max {quantity_to_analyse},' : max_stats}
    if print_stats:
        print(dict)

    list_of_gas = util.transfer_body_order_conversion.get_mga_list_from_characters(mga_chars)[1:-1]
    list_of_legs = util.transfer_body_order_conversion.get_list_of_legs_from_characters(mga_chars)

    direction_list = ["Radial", "Normal", "Axial"]
    labels = {'dv' : [],
              'tof' : [f"{i}" for i in list_of_legs],
              'inc_vel' : [f"{i}" for i in list_of_gas],
              'swi_per' : [f"{i}" for i in list_of_gas],
              'ooa' : [f"{i}" for i in list_of_gas],
              'ip_ang' : [f"{i}" for i in list_of_gas],
              'oop_ang' : [f"{i}" for i in list_of_gas],
              'fc_legs' : [f"{i}" for i in list_of_legs],
              'fc_axes' : direction_list,
              'revs' : [f"{i}" for i in list_of_legs]}

    if plot_quantity:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.hist(values_to_analyse, bins=bins, label=labels[quantity_to_analyse])
        ax.set_ylabel('Frequency [-]')
        ax.set_xlabel(bound_quantity[quantity_to_analyse])
        title_elem = dir_of_dir.split('/')[title].split('_')
        ax.set_title(f'{title_elem[0]} — 24 islands')
        # ax.set_xticks([])
        # if quantity_to_analyse == 'swi_per':
        #     ax.set_xscale('log')
        ax.grid()
        ax.legend()
    if save_fig:
        fig.savefig('figures/' + f"{dir_of_dir.split('/')[title]}_{quantity_to_analyse}histo.jpg", bbox_inches="tight")
        

    save2txt(dict, 'get_stats.dat', dir_of_dir)

