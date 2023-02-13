'''
Author: Sean Cowan
Purpose: MSc Thesis
Date Created: 31-01-2023

This module implements the multi-purpose function that performs an evolution for either ltto or mgaso.
'''
###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General 
import numpy as np
import pygmo as pg

# Tudatpy
from tudatpy.io import save2txt
from tudatpy.kernel import constants

# import sys
# sys.path.insert(0, "/Users/sean/Desktop/tudelft/tudat/tudat-bundle-test/build/tudatpy")

def perform_evolution(archi, 
                        number_of_islands, 
                        num_gen, 
                        objectives):
    list_of_f_dicts = []
    list_of_x_dicts = []
    if len(objectives) == 1:
        # with SO no weird work around champions have to be made
        # get_champions_x, ndf_x = lambda archi : (archi.get_champions_x(), None) # 2, no_of_islands 
        # get_champions_f, ndf_f = lambda archi : (archi.get_champions_f(), None)

        def get_champions_x(archi):
            return archi.get_champions_x(), None
        def get_champions_f(archi):
            return archi.get_champions_f(), None

    else:
        current_island_populations = lambda archi : [isl.get_population() for isl in archi]

        def get_champions_x(archi):
            pop_list= []
            pop_f_list= []
            ndf_x = []
            champs_x = []
            for j in range(number_of_islands):
                pop_list.append(current_island_populations(archi)[j].get_x())
                pop_f_list.append(current_island_populations(archi)[j].get_f())
                current_ndf_indices = pg.non_dominated_front_2d(pop_f_list[j]) # determine how to sort
                ndf_x.append([pop_list[j][i] for i in current_ndf_indices])
                champs_x.append(ndf_x[j][0]) # j for island, 0 for first (lowest dV) option
            return champs_x, ndf_x

        def get_champions_f(archi):
            pop_list= []
            pop_f_list= []
            ndf_f = []
            champs_f = []
            for j in range(number_of_islands):
                pop_list.append(current_island_populations(archi)[j].get_x())
                pop_f_list.append(current_island_populations(archi)[j].get_f())
                current_ndf_indices = pg.non_dominated_front_2d(pop_f_list[j]) # determine how to sort
                ndf_f.append([pop_f_list[j][i] for i in current_ndf_indices])
                champs_f.append(ndf_f[j][0]) # j for island, 0 for first (lowest dV) option
                # champs_f[j][1] /= 86400.0

            return champs_f, ndf_f

    for i in range(num_gen): # step between which topology steps are executed
        print('Evolving Gen : %i / %i' % (i+1, num_gen), end='\r')
        archi.evolve()
        champs_dict_current_gen = {}
        champ_f_dict_current_gen = {}
        champions_x, ndf_x = get_champions_x(archi)
        champions_f, ndf_f = get_champions_f(archi)
        for j in range(number_of_islands):
                champs_dict_current_gen[j] = champions_x[j] 
                champ_f_dict_current_gen[j] = champions_f[j] 
        list_of_x_dicts.append(champs_dict_current_gen)
        list_of_f_dicts.append(champ_f_dict_current_gen)
        # print('Topology', archi.get_topology())

        archi.wait_check()


    print("""
==================
Evolution finished
==================
          """)
    return list_of_x_dicts, list_of_f_dicts, champions_x, champions_f, ndf_x, ndf_f

