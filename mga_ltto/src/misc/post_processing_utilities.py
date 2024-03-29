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
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

# for custom legend
from matplotlib.lines import Line2D

# for many colors
import matplotlib.colors as mcolors

# Tudatpy
from tudatpy.io import save2txt
from tudatpy.kernel import constants

# Local
import core.multipurpose.mga_low_thrust_utilities as util
from misc.trajectory3d import trajectory_3d

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
            bodies=["Sun", "Mercury", "Venus", "Earth"],
            # bodies=["Sun"] + tbo_list,
            frame_orientation= 'ECLIPJ2000',
            thrust_history=thrust_history_dict,
            projection=projection)
    # plt.figure(constrained_layout=True)
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
    # fig.set_size_inches(8, 8)
    # plt.show()

def objective_per_generation_visualisation(dir=None, 
                                           dir_of_dir=None, 
                                           no_of_islands=4,
                                           title=1,
                                           add_local_optimisation=False,
                                           save_fig=False, 
                                           seq=0,
                                           num_gen=200,
                                           ips=14):
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

    fitness_value_dict = {}
    variable_value_dict = {}
    auxiliary_info_dict = {}
    if dir != None:
        dir_list = dir.split('/')
        for i in range(no_of_islands):
            island = i + seq*no_of_islands
            auxiliary_info_dict[i] = np.loadtxt(dir + f'layer_{seq}/islands/island_{island}/auxiliary_info.dat',
                                                delimiter=',', dtype=str)
            fitness_value_dict[i] = np.loadtxt(dir + f'layer_{seq}/islands/island_{island}/champ_f_per_gen.dat')
            variable_value_dict[i] = np.loadtxt(dir + f'layer_{seq}/islands/island_{island}/champs_per_gen.dat')
            evaluated_sequences_database = np.loadtxt(dir + f'topology_database/evaluated_sequences_database.txt',
                                                      delimiter=',', dtype=str)
            # print(fitness_value_dict)

    if dir_of_dir != None:
        dir_list = dir_of_dir.split('/')
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

    # print(auxiliary_info_dict[0])
    # auxiliary_info_dict = {}
    # for i in range(len(auxiliary_info)):
    #     auxiliary_info_dict[auxiliary_info[i][0]] = auxiliary_info[i][1].replace('\t', '')
    # print(evaluated_sequences_database)

    fitness_array = np.zeros((len(fitness_value_dict[0][:,0]), no_of_islands))
    for i in range(no_of_islands):
        fitness_array[:, i] = fitness_value_dict[i][:, 1] #if not add_local_optimisation else fitness_value_dict[i][:, 1]

    min_deltav_value_before_local = np.min(fitness_array[:-1, :])
    min_deltav_value = np.min(fitness_array) / 1000

    c_list = ['b', 'r', 'y', 'g', 'c', 'm', 'k']

    custom_lines = [Line2D([0], [0], color=c_list[0], lw=2),
                    Line2D([0], [0], color=c_list[1], lw=2),
                    Line2D([0], [0], color=c_list[2], lw=2),
                    Line2D([0], [0], color=c_list[3], lw=2),
                    Line2D([0], [0], color=c_list[4], lw=2)]
    custom_names = [f'{evaluated_sequences_database[1][0]}',
                    f'{evaluated_sequences_database[2][0]}',
                    f'{evaluated_sequences_database[3][0]}',
                    f'{evaluated_sequences_database[4][0]}',
                    f'{evaluated_sequences_database[5][0]}']
    
    fig, ax = plt.subplots(1, 1)
    for island in range(no_of_islands):
        ax.plot(fitness_value_dict[island][:num_gen,0], fitness_value_dict[island][:num_gen,1] / 1000, 
                c=c_list[(island // ips) % no_of_islands])
    ax.axhline(min_deltav_value, c='k', linestyle='-', label=f'Minimum : {int(min_deltav_value / 1000)}')
    if add_local_optimisation:
        ax.axhline(min_deltav_value_before_local, c='k', linestyle='-', label=f'Minimum : {int(min_deltav_value_before_local / 1000)}')
    trans = transforms.blended_transform_factory(ax.get_yticklabels()[0].get_transform(), ax.transData)
    ax.text(0,min_deltav_value, "{:.3f}".format(min_deltav_value), color="red", transform=trans, 
            ha="right", va="center")
    if add_local_optimisation:
        ax.text(0,min_deltav_value_before_local, "{:.0f}".format(min_deltav_value_before_local),
                color="red", transform=trans, ha="right", va="center")
        ax.fill_between([len(fitness_value_dict[0]), len(fitness_value_dict[0])+10], 10000, 40000, color='lightskyblue')


    ax.legend(custom_lines, custom_names, loc='upper right')
    ax.set_ylabel(r'$\Delta V$ [km / s]')
    ax.set_xlabel(r'Generation count [-]')
    # ax.legend()
    ax.grid()
    # ax.set_title(auxiliary_info_dict['MGA Sequence'], fontweight='semibold', fontsize=18)

    # dir_of_dir title
    # title_elem = dir_of_dir.split('/')[title].split('_')
    # ax.set_title(f'{title_elem[0]} — {no_of_islands} islands')

    #dir title
    # ax.set_title(title)

    ax.set_yscale('log')
    # ax.set_ylim([10000, 40000])
    if save_fig:
        if add_local_optimisation:
            fig.savefig('figures/' + f"{dir.split('/')[title]}_local.jpg")
        # else:
        #     fig.savefig('figures/' + f"{dir.split('/')[title]}.jpg")
        else:
            fig.savefig('figures/' + f"{title}.jpg")

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

        ax.scatter(champions[it], champion_fitness[it], c=color_list[it], label=f'{lb} - {ub} [MJD]' if grid_search else '')
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
        ax.set_xlabel(r'Departure Date [MJD]')

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
        ax.set_title(f'{title_elem[0]} — {title_elem[1].replace("lb","")} - {title_elem[2].replace("ub","")} [MJD] — {title_elem[3]}')
    else:
        title_elem = directory_list[0].split('_')
        lb = optim_chars_dict['Departure date [mjd2000] LB'] 
        lb_float = float(lb)+ 51544.5
        ub = optim_chars_dict['Departure date [mjd2000] UB'] 
        ub_float = float(ub)+ 51544.5
        ax.set_title(f'{title_elem[0]} — {lb} - {ub} [MJD]')

    ax.set_ylim([20000, 35000])
    if not grid_search:
        ax.set_xlim([lb_float, ub_float])
    if save_fig:
        if add_local_optimisation:
            fig.savefig('figures/' + f"{dir_of_dir_of_dir.split('/')[title] if grid_search else dir_of_dir.split('/')[title]}_local.jpg", bbox_inches="tight")
        else:
            fig.savefig('figures/' + f"{dir_of_dir_of_dir.split('/')[title] if grid_search else dir_of_dir.split('/')[title]}.jpg", bbox_inches="tight")
    
def get_scattered_objectives_extended(dir1,
                                      dir2,
                                      dir3,
                                      title=2,
                                      add_local_optimisation=False,
                                      no_of_islands=4,
                                      save_fig=False):


    directory_list = []
    for dir in [dir1, dir2, dir3]:
        for root, dirs, files in os.walk(dir):
            dirs_2 = [root + dir for dir in sorted(dirs)]
            directory_list.extend(dirs_2)
            # print(directory_list)
            break

    color_list = \
    ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan',]
    
    champion_fitness = {}
    champion_fitness_local = {}
    champions = {}
    champions_local = {}
    fig, ax = plt.subplots(1, 1)
    for it, dir in enumerate(directory_list):

        champions[it] = np.loadtxt(dir + "/champions/champions.dat")[:, 1] / 86400 + 51544.5
        champion_fitness[it] = np.loadtxt(dir + "/champions/champion_fitness.dat")[:, 1] 
        optim_chars = np.loadtxt(dir + '/optimisation_characteristics.dat', delimiter=',', dtype=str)
        optim_chars_dict = {}
        for i in range(len(optim_chars)):
            optim_chars_dict[optim_chars[i][0]] = optim_chars[i][1].replace('\t', '')

        ax.scatter(champions[it], champion_fitness[it] / 1000, c=color_list[it%len(color_list)], s=20)#, label=f'{lb} - {ub} [MJD]' if grid_search else '')
        if add_local_optimisation:
            champion_fitness_local[it] = np.loadtxt(dir + "/local_optimisation/final_population.dat")[:, 1]
            champions_local[it] = np.loadtxt(dir + "/local_optimisation/final_population_x.dat")[:, 1] / 86400 + 51544.5
            ax.scatter(champions_local[it], champion_fitness_local[it]/ 1000, c=color_list[it%len(color_list)], marker="+",
                       s=15)
            for it2 in range(no_of_islands):
                # ax.axvline((champions_local[it][it2] / 86400) + 51544.5, ymin=(champion_fitness_local[it][it2] - 15000) / 5000,
                #            ymax=(champion_fitness[it][it2] - 15000) / 5000, c=color_list[it], linestyle='--', linewidth=0.5)
                x_points = [champions[it][it2], champions_local[it][it2]]
                f_points = [champion_fitness[it][it2], champion_fitness_local[it][it2]]
                ax.plot(x_points, f_points, c=color_list[it%len(color_list)], linestyle='--', linewidth=0.5)
        # ax.axvline(float(lb), c='k', linestyle='-', linewidth=1)
        # ax.axvline(float(ub), c='k', linestyle='-', linewidth=1)
        ax.set_ylabel(r'$\Delta V$ [km / s]')
        ax.set_xlabel(r'Departure Date [MJD]')

    # Add value + line for minimum
    fitness_array = np.array(list(champion_fitness.values())) / 1000 if not add_local_optimisation else \
                            np.array(list(champion_fitness_local.values()))  / 1000
    min_deltav_value = np.min(fitness_array)
    ax.axhline(min_deltav_value, c='k', linestyle='-', label=f'Minimum : {min_deltav_value} [km/s]')
    trans = transforms.blended_transform_factory(
    ax.get_yticklabels()[0].get_transform(), ax.transData)
    ax.text(0,min_deltav_value,  f"{round(min_deltav_value, 3)}", color="red", transform=trans, 
            ha="right", va="center")

    # custom legend
    custom_lines = [Line2D([0], [0], color='k', lw=2),
                    Line2D([], [], c='k',  marker='o', linestyle='None'),
                    Line2D([], [], c='k',  marker='+', linestyle='None')]
    custom_names=[f'Minimum : {round(min_deltav_value, 3)} [km/s]', 'GA optima', 'locally optimised']
    if not add_local_optimisation:
        custom_lines.pop(-1)
        custom_names.pop(-1)
    ax.legend(custom_lines, custom_names)#, loc='upper right')
    # ax.legend()
    ax.grid()
    x_data = ax.collections[0].get_offsets()
    lb = int(x_data[0][0])
    x_data = ax.collections[-1].get_offsets()
    ub = int(x_data[-1][0])
    title_elem = directory_list[0].split('/')[2].split('_')
    # lb = optim_chars_dict['Departure date [mjd2000] LB'] 
    # lb_float = float(lb)+ 51544.5
    # ub = optim_chars_dict['Departure date [mjd2000] UB'] 
    # ub_float = float(ub)+ 51544.5
    ax.set_title(f'{title_elem[0]} transfer')
    # if title_elem[0] == 'EJ':
    #     ax.set_xlim([61400, 63400])
    # elif title_elem[0] == 'EMJ':
    #     ax.set_xlim([61400, 63400])
    # elif title_elem[0] == 'EEMJ':
    #     ax.set_xlim([61400, 63200])
    # elif title_elem[0] == 'EEEMJ':
    #     ax.set_xlim([61400, 63400])

    ax.set_ylim([15, 25])
    if save_fig:
        if add_local_optimisation:
            fig.savefig('figures/' + f"{title_elem[0]}_{lb}_{ub}_local.jpg", bbox_inches="tight")
        else:
            fig.savefig('figures/' + f"{title_elem[0]}_{lb}_{ub}.jpg", bbox_inches="tight")

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

def thrust_propagation(dir=None, mass=1):

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

    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    ax.plot(time_history, thrust_acceleration[:, 1] * 1000 * mass, linestyle=':', label=r'$a_r$')
    ax.plot(time_history, thrust_acceleration[:, 2] * 1000 * mass, linestyle='-.', label=r'$a_\theta$')
    # ax.plot(time_history, thrust_acceleration[:, 3] * 1000 * mass, linestyle='--', label=r'$a_z$')
    # ax.plot(time_history, thrust_norm * 1000 * mass, label='a')
    for i in range(len(list_of_mga_sequence_char)):
        ax.axvline(node_times[i, 1], c='k', linestyle='-', linewidth=0.5)
    # ax.axvline(node_times[0, 1], c=color_list[0], linestyle=linestyle,
    #            label=list_of_mga_sequence_char[0], ymin = np.min(thrust_norm), ymax =
    #            np.max(thrust_norm))
    # ax.set_ylabel(r'Thrust norm [m / s$^2$]')
    ax.set_ylabel(r'Thrust norm [mN]')
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

    if quantity_to_analyse == 'revs':
        values_to_analyse.astype('int32')

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
        fig, ax = plt.subplots(1, 1)
        ax.hist(values_to_analyse, bins=bins, label=labels[quantity_to_analyse])
        ax.set_ylabel('Frequency [-]')
        ax.set_xlabel(bound_quantity[quantity_to_analyse])
        title_elem = dir_of_dir.split('/')[title].split('_')
        ax.set_title(f'{title_elem[0]} transfer')
        if quantity_to_analyse == 'revs':
            ax.set_xticks([0.0, 1.0, 2.0])#, label['0', '1', '2'])
        # if quantity_to_analyse == 'swi_per':
        #     ax.set_xscale('log')
        ax.grid()
        ax.legend()
    if save_fig:
        fig.savefig('figures/' + f"{dir_of_dir.split('/')[title]}_{quantity_to_analyse}histo.jpg", bbox_inches="tight")
        

    save2txt(dict, 'get_stats.dat', dir_of_dir)

def compare_data_to_hs(data_file, hs_file):
    """
    This function plots data that compares validation runs between real BEPICOLOMBO data and a recreation using hodographic shaping
    """

    #Load data from file
    data_state = np.loadtxt(data_file)
    hs_state = np.loadtxt(hs_file)

    #Plot data
    fig = plt.figure(constrained_layout=True)
    ax1 = fig.add_subplot(1, 1, 1, projection="3d")
    ax1.plot(data_state[:, 1], data_state[:, 2], data_state[:, 3], label='BEPI data')
    ax1.plot(hs_state[:, 1], hs_state[:, 2], hs_state[:, 3], label='HS recreation')
    # ax1.set_aspect('equal')
    ax1.set_aspect('equal')

    ax1.set_xlabel("x [m]", labelpad=15)
    ax1.set_ylabel("y [m]", labelpad=15)
    ax1.set_zlabel("z [m]", labelpad=15)
    ax1.legend()

    hs_state_norm = [np.linalg.norm(i) for i in hs_state[:, 1:4]]
    data_state_norm = [np.linalg.norm(i) for i in data_state[:, 1:4]]
    diff = [hs_state_norm[i] - data_state_norm[i] for i in range(len(hs_state_norm))]
    diff_axes = {}
    for i in range(1, 4):
        diff_axes[i] = [hs_state[j, i] - data_state[j, i] for j in range(len(hs_state))]

    fig2 = plt.figure(constrained_layout=True)
    ax2 = fig2.add_subplot(1, 1, 1)
    # ax2.plot(data_state[:, 0], diff)
    ax2.plot(data_state[:, 0], diff_axes[1], label='Radial')
    ax2.plot(data_state[:, 0], diff_axes[2], label='Normal')
    ax2.plot(data_state[:, 0], diff_axes[3], label='Axial')
    ax2.grid()
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Position difference [m]")
    ax2.legend()
    return fig, fig2, ax1, ax2


def mgaso_scatter_per_fitprop(data_directory, fitprop_values=None, frac_values=None, save_fig=False, title=1):
    # OLD FUNCTION

    # Helper class
    class mgaso_run:
        def __init__(self,
                     frac=None, 
                     fitprop=None,
                     sequences=None,
                     fitnesses=None,
                     min_dvs=None,
                     mean_dvs=None) -> None:

            self.frac = frac
            self.fitprop = fitprop
            self.sequences = sequences
            self.min_dvs = min_dvs
            self.mean_dvs = mean_dvs
            self.fitnesses = fitnesses if fitnesses != None else self.fitprop*self.min_dvs + (1-self.fitprop)*self.mean_dvs

    #Get data
    for root, dirs, files in os.walk(data_directory):
        directory_list = sorted(dirs)
        break

    number_of_sims = len(fitprop_values) * len(frac_values)
    sorted_database_sequences = {}
    sorted_database_values = {}
    it2 = 0
    for it, dir in enumerate(directory_list):
        if os.path.exists(root + dir + "/sorted_evaluated_sequences_database.txt") and it < number_of_sims:
            # print(root + dir + "/sorted_evaluated_sequences_database.txt")
            sorted_database_sequences[it2] = np.loadtxt(root + dir + "/sorted_evaluated_sequences_database.txt", dtype=str,
                                             delimiter=",", usecols=0, skiprows=1)
            sorted_database_values[it2] = np.loadtxt(root + dir + "/sorted_evaluated_sequences_database.txt",
                                                    delimiter=",", usecols=(1, 2, 3), skiprows=1)
            it2 +=1

    # Make all mgaso run objects
    mgaso_runs = []
    max_value_per_frac = []
    min_value_per_frac = []
    it = 0
    for i in range(len(frac_values)):
        current_frac = frac_values[i]
        fitnesses = []
        for j in range(len(fitprop_values)):
            current_fitprop = fitprop_values[j]
            mgaso_runs.append(mgaso_run(frac=current_frac, 
                                        fitprop=current_fitprop,
                                        sequences=sorted_database_sequences[it],
                                        min_dvs=sorted_database_values[it][:, 0],
                                        mean_dvs=sorted_database_values[it][:, 1]))
            fitnesses.append(mgaso_runs[it].fitnesses)
            it += 1
        fitnesses = list(np.concatenate(fitnesses).flat)
        max_value_per_frac.append(max(fitnesses) + 2000)
        min_value_per_frac.append(min(fitnesses) - 2000)


    # Do plotting
    it = 0
    rot = [45, 90, 90, 90, 90, 90]
    for i in range(len(frac_values)):
        fig = plt.figure()
        fig.suptitle(f"EJ transfer - Fraction explored: {frac_values[i]}", fontweight='bold', y=0.95)

        for j in range(len(fitprop_values)):
            ax = fig.add_subplot(2,3,1+j)
            run = mgaso_runs[it]
            ax.scatter(run.sequences, run.fitnesses)
            index_list = []
            for it2, p in enumerate(run.sequences):
                if p in ['EJ', 'EMJ', 'EEMJ', 'EEEMJ']:
                    index_list.append(it2)

            it+=1
            plt.setp(ax.get_xticklabels(), rotation=rot[i], horizontalalignment='right')
            ax.set_ylim([min_value_per_frac[i], max_value_per_frac[i]])
            ax.title.set_text(f'Fitprop fraction: {fitprop_values[j]}')
            ax.grid()
            # ax.legend()
            trans = transforms.blended_transform_factory(ax.get_yticklabels()[0].get_transform(), ax.transData)
            ax.text(0,min_value_per_frac[i], "{:.0f}".format(min_value_per_frac[i]), color="red", transform=trans, 
                    ha="right", va="center")
            trans = transforms.blended_transform_factory(ax.get_yticklabels()[0].get_transform(), ax.transData)
            ax.text(0,max_value_per_frac[i], "{:.0f}".format(max_value_per_frac[i]), color="red", transform=trans, 
                    ha="right", va="center")
            for q in index_list:
                plt.gca().get_xticklabels()[q].set_color('red')
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        if save_fig:
            fig.savefig('figures/mgaso/' + f"{data_directory.split('/')[title]}_{frac_values[i]}.jpg", bbox_inches="tight")

    plt.show()

def mgaso_scatter(data_directory, fitprop_values=None, frac_values=None, save_fig=False, title=1):

    # Helper class
    class mgaso_run:
        def __init__(self,
                     frac=None, 
                     fitprop=None,
                     sequences=None,
                     fitnesses=None,
                     min_dvs=None,
                     mean_dvs=None) -> None:

            self.frac = frac
            self.fitprop = fitprop
            self.sequences = sequences
            self.min_dvs = min_dvs
            self.mean_dvs = mean_dvs
            self.fitnesses = fitnesses if fitnesses != None else self.fitprop*self.min_dvs + (1-self.fitprop)*self.mean_dvs

    #Get data
    for root, dirs, files in os.walk(data_directory):
        directory_list = sorted(dirs)
        break

    number_of_sims = len(fitprop_values) * len(frac_values)
    sorted_database_sequences = {}
    sorted_database_values = {}
    it2 = 0
    for it, dir in enumerate(directory_list):
        if os.path.exists(root + dir + "/sorted_evaluated_sequences_database.txt") and it < number_of_sims:
            # print(root + dir + "/sorted_evaluated_sequences_database.txt")
            sorted_database_sequences[it2] = np.loadtxt(root + dir + "/sorted_evaluated_sequences_database.txt", dtype=str,
                                             delimiter=",", usecols=0, skiprows=1)
            sorted_database_values[it2] = np.loadtxt(root + dir + "/sorted_evaluated_sequences_database.txt",
                                                    delimiter=",", usecols=(1, 2, 3), skiprows=1)
            it2 +=1

    # Make all mgaso run objects
    mgaso_runs = []
    max_value_per_frac = []
    min_value_per_frac = []
    it = 0
    for i in range(len(frac_values)):
        current_frac = frac_values[i]
        fitnesses = []
        for j in range(len(fitprop_values)):
            current_fitprop = fitprop_values[j]
            mgaso_runs.append(mgaso_run(frac=current_frac, 
                                        fitprop=current_fitprop,
                                        sequences=sorted_database_sequences[it],
                                        min_dvs=sorted_database_values[it][:, 0],
                                        mean_dvs=sorted_database_values[it][:, 1]))
            fitnesses.append(mgaso_runs[it].fitnesses)
            it += 1
        fitnesses = list(np.concatenate(fitnesses).flat)
        max_value_per_frac.append(max(fitnesses) + 2000)
        min_value_per_frac.append(min(fitnesses) - 2000)


    # Do plotting
    it = 0
    rot = [45, 90, 90, 90, 90, 90]

    c_list = ['b', 'r', 'y']
    m_list = ['o', 'x', 'd']
    # fig, ax = plt.subplots(1, 1)
    for i in range(len(frac_values)):
        fig = plt.figure()
        fig.suptitle(f"EJ transfer — Fitprop_itbs=1.0", fontweight='bold', y=0.95)

        for j in range(len(fitprop_values)):
            ax = fig.add_subplot(2,3,1+j)
            run = mgaso_runs[it]
            ax.scatter(run.sequences, run.fitnesses, c=c_list[i], marker=m_list[j])
            index_list = []
            for it2, p in enumerate(run.sequences):
                if p in ['EJ', 'EMJ', 'EEMJ', 'EEEMJ']:
                    index_list.append(it2)

            it+=1
            plt.setp(ax.get_xticklabels(), rotation=rot[i], horizontalalignment='right')
            ax.set_ylim([min_value_per_frac[i], max_value_per_frac[i]])
            ax.title.set_text(f'Fitprop fraction: {fitprop_values[j]}')
            ax.grid()
            # ax.legend()
            trans = transforms.blended_transform_factory(ax.get_yticklabels()[0].get_transform(), ax.transData)
            ax.text(0,min_value_per_frac[i], "{:.0f}".format(min_value_per_frac[i]), color="red", transform=trans, 
                    ha="right", va="center")
            trans = transforms.blended_transform_factory(ax.get_yticklabels()[0].get_transform(), ax.transData)
            ax.text(0,max_value_per_frac[i], "{:.0f}".format(max_value_per_frac[i]), color="red", transform=trans, 
                    ha="right", va="center")
            for q in index_list:
                plt.gca().get_xticklabels()[q].set_color('red')
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        if save_fig:
            fig.savefig('figures/mgaso/' + f"{data_directory.split('/')[title]}_{frac_values[i]}.jpg", bbox_inches="tight")

    plt.show()

def mgaso_scatter_multi(dir1, dir2, dir3, fitprop_values=None, fitprop_itbs_values=None, frac_values=None,
                        save_fig=False, title=1):

    # Helper class
    class mgaso_run:
        def __init__(self,
                     frac=None, 
                     fitprop=None,
                     fitprop_itbs=None,
                     sorted_sequences=None,
                     sequences=None,
                     fitnesses=None,
                     optim_chars=None,
                     min_dvs=None,
                     mean_dvs=None,
                     seed=None) -> None:

            self.frac = frac
            self.fitprop = fitprop
            self.fitprop_itbs = fitprop_itbs
            self.sorted_sequences = sorted_sequences
            self.sequences = sequences
            self.optim_chars = optim_chars
            optim_chars_dict = {}
            for i in range(len(optim_chars)):
                optim_chars_dict[optim_chars[i][0]] = optim_chars[i][1].replace('\t', '')
            self.layer_0 = int(optim_chars_dict[f'Number of sequences per planet - Layer 0']) * 4 + 1  # 4 possible planets
            self.layer_1 = int(optim_chars_dict[f'Number of sequences per planet - Layer 1']) * 4 + 1 
            # self.layer_2 = int(optim_chars_dict[f'Number of sequences per planet - Layer 2']) * 4 + 1 
            self.layer_0_seq = sequences[0:self.layer_0]
            self.layer_1_seq = sequences[self.layer_0:(self.layer_0+self.layer_1)]
            # self.layer_2_seq = sequences[(self.layer_0+self.layer_1):(self.layer_0+self.layer_1+self.layer_2)]
            self.min_dvs = min_dvs
            self.mean_dvs = mean_dvs
            self.fitnesses = fitnesses if fitnesses != None else self.fitprop*self.min_dvs + (1-self.fitprop)*self.mean_dvs
            self.layer_list = []
            for it, seq in enumerate(sorted_sequences):
                if seq in self.layer_0_seq:
                    self.layer_list.append(0)
                elif seq in self.layer_1_seq:
                    self.layer_list.append(1)
                # elif seq in self.layer_2_seq:
                #     self.layer_list.append(2)
                else:
                    raise RuntimeError("SOmething went wrong.")
            self.seed = seed



    #Get data
    dir_list = []
    if dir1 != None:
        dir_list.append(dir1)
    if dir2 != None:
        dir_list.append(dir2)
    if dir3 != None:
        dir_list.append(dir3)
    no_of_seeds = len(dir_list)

    directory_list = []
    for dir in dir_list:
        for root, dirs, files in os.walk(dir):
            dirs_2 = [root + dir for dir in sorted(dirs)]
            directory_list.extend(dirs_2)
            # print(directory_list)
            break

    # number_of_sims = len(fitprop_values) * len(frac_values)
    sorted_database_sequences = {}
    sorted_database_values = {}
    evaluated_sequences_database = {}
    optim_chars = {}
    it2 = 0
    for it, dir in enumerate(directory_list):
        if os.path.exists(dir + "/sorted_evaluated_sequences_database.txt"):# and it < number_of_sims:
            # print(root + dir + "/sorted_evaluated_sequences_database.txt")
            sorted_database_sequences[it2] = np.loadtxt(dir + "/sorted_evaluated_sequences_database.txt", dtype=str,
                                             delimiter=",", usecols=0, skiprows=1)
            sorted_database_values[it2] = np.loadtxt(dir + "/sorted_evaluated_sequences_database.txt",
                                                    delimiter=",", usecols=(1, 2, 3), skiprows=1)
            evaluated_sequences_database[it2] = np.loadtxt(dir +
                                                           "/topology_database/evaluated_sequences_database.txt",
                                                           dtype=str, delimiter=",", usecols=(0), skiprows=1)
            optim_chars[it2] = np.loadtxt(dir + '/optimisation_characteristics.dat', delimiter=',', dtype=str,
                                          usecols=(0, 1))
            it2 +=1

    # Values
    # frac_values = [0.1, 0.4, 0.7]
    # fitprop_values = [1.0, 0.5, 0.0]
    # fitprop_itbs_values = [1.0, 0.5, 0.0]

    # Make all mgaso run objects
    mgaso_runs = []
    max_value_per_frac = []
    min_value_per_frac = []
    it = 0

    # # Case for fitprop_itbs > frac > fitprop
    # for k, fitprop_itbs in enumerate(fitprop_itbs_values):
    #     for i, frac in enumerate(frac_values):
    #         # fitnesses = []
    #         for j, fitprop in enumerate(fitprop_values):
    #             mgaso_runs.append(mgaso_run(frac=frac, 
    #                                         fitprop=fitprop,
    #                                         fitprop_itbs=fitprop_itbs,
    #                                         sorted_sequences=sorted_database_sequences[it],
    #                                         sequences = evaluated_sequences_database[it],
    #                                         optim_chars = optim_chars[it],
    #                                         min_dvs=sorted_database_values[it][:, 0],
    #                                         mean_dvs=sorted_database_values[it][:, 1]))
    #             # fitnesses.append(mgaso_runs[it].fitnesses)
    #             it += 1

    # Case for frac > fitprop > fitprop_itbs
    for u in range(no_of_seeds):
        for i, frac in enumerate(frac_values):
            for j, fitprop in enumerate(fitprop_values):
                for k, fitprop_itbs in enumerate(fitprop_itbs_values):
                    mgaso_runs.append(mgaso_run(frac=frac, 
                                                fitprop=fitprop,
                                                fitprop_itbs=fitprop_itbs,
                                                sorted_sequences=sorted_database_sequences[it],
                                                sequences = evaluated_sequences_database[it],
                                                optim_chars = optim_chars[it],
                                                min_dvs=sorted_database_values[it][:, 0],
                                                mean_dvs=sorted_database_values[it][:, 1],
                                                seed=u))
                    it += 1

    assert it == len(directory_list)

    # Do plotting
    rot = [30, 80, 90, 90, 90, 90]

    c_list = [['lightgray', 'gray', 'black'], ['lightskyblue', 'blue', 'darkblue'], ['limegreen', 'forestgreen',
                                                                                     'darkgreen']]
    m_list = ['o', 'x', 'd']

    custom_lines = list(Line2D([0], [0], color=c_list[1][i], lw=8) for i in range(len(fitprop_itbs_values)))
    custom_lines.extend([ Line2D([], [], c='k',  marker=m_list[i], linestyle='None') for i in
                         range(len(fitprop_itbs_values))])

    custom_names = list(f'Fitprop : {fitprop_values[i]}' for i in range(len(fitprop_values)))
    custom_names.extend([ f'Fitprop ITBS : {fitprop_itbs_values[i]}' for i in range(len(fitprop_itbs_values))])

    for i in range(len(frac_values)):
        fig, ax = plt.subplots(1, 1) 
        current_mgaso_frac_runs = [run for run in mgaso_runs if run.frac==frac_values[i]]# and run.seed == u]

        it = 0
        for k in range(len(fitprop_values)):
            for j in range(len(fitprop_itbs_values)):
                run = current_mgaso_frac_runs[it]
                # current_m_list = [m_list[p] for p in run.layer_list]
                current_m_list = [m_list[0] if p == 0 else m_list[1] for p in run.layer_list ]
                ax.scatter(run.sorted_sequences, run.fitnesses / 1000, c=c_list[j][k], marker=current_m_list[it])

                plt.setp(ax.get_xticklabels(), rotation=rot[i], horizontalalignment='right')
                # ax.set_ylim([min_value_per_frac[i], max_value_per_frac[i]])
                ax.title.set_text(f'Fraction: {frac_values[i]}')
                ax.grid()
                # ax.legend()
                # trans = transforms.blended_transform_factory(ax.get_yticklabels()[0].get_transform(), ax.transData)
                # ax.text(0,min_value_per_frac[i], "{:.0f}".format(min_value_per_frac[i]), color="red", transform=trans, 
                #         ha="right", va="center")
                # trans = transforms.blended_transform_factory(ax.get_yticklabels()[0].get_transform(), ax.transData)
                # ax.text(0,max_value_per_frac[i], "{:.0f}".format(max_value_per_frac[i]), color="red", transform=trans, 
                #         ha="right", va="center")
                it+=1
                # break

                # plt.show()


        ax.set_ylabel('Fitness [km/s]')
        ax.set_xlabel('Sequence [-]')
        # ax.set_yscale('log')
        ax.legend(custom_lines, custom_names)#, loc='upper right')

        # index_list = []
        for it2, q in enumerate(plt.gca().get_xticklabels()):
            current_label = q.get_text()
            if current_label in ['EJ', 'EMJ', 'EEMJ', 'EEEMJ']:
                ax.get_xticklabels()[it2].set_color('red')

        # customize xticks
        # ticks = [r + 10 for r in ax.get_xticks() if i == 2]
        # ax.set_xticks(ticks=ticks)
        # values = []
        # for it2, q in enumerate(plt.gca().get_yticks()):
        #     # print(q)
        #     values.append(q)
        # plt.yticks(np.linspace(min(values), max(values), num=5))

        # plt.subplots_adjust(wspace=0.3, hspace=0.3)
        if save_fig:
            fig.savefig('figures/mgaso/' + f"{title}_{frac_values[i]}.jpg", bbox_inches="tight")

    plt.show()

