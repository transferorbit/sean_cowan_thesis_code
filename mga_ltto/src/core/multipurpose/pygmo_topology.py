'''

Author: Sean Cowan
Purpose: MSc Thesis
Date Created: 02-03-2023

This module creates a PyGMO compatible topology class that links the islands of the each sequence together
'''
###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General 
import numpy as np
import sys
import os
import pygmo as pg

# Tudatpy 
# from tudatpy.kernel import constants
# from tudatpy.util import result2array

current_dir = os.getcwd()
sys.path.append('/Users/sean/Desktop/tudelft/thesis/code/mga_ltto/src/') # this only works if you run ltto and mgso while in the directory that includes those files
# Local
# import core.multipurpose.mga_low_thrust_utilities as util

#######################################################################
# TOPOLOGY CLASS ######################################################
#######################################################################

class custom_vertex:

    def __init__(self, id, iteration_count) -> None:
        self.id = id
        self.iteration_count = iteration_count

class MGASOTopology(pg.free_form):
    """
    Class to connect islands of a specific sequence together
    """

    def __init__(self, number_of_sequences : int,
                 islands_per_sequence : int,
                 topology_weight : float=0.01) -> None:

        super().__init__()

        self.number_of_sequences = number_of_sequences
        self.islands_per_sequence = islands_per_sequence
        self.topology_weight = topology_weight
        self.list_of_custom_vertices = []
        self.vertex_names_list = MGASOTopology.get_vertex_names(self.number_of_sequences, self.islands_per_sequence)
        self.num_edges = 0

    @staticmethod
    def get_vertex_names(number_of_sequences, islands_per_sequence):
        """
        This function returns a list of all the required vertex names in the form (A1, A2, .., An, B1, B2, .. Bn, .., Z1,
                                                                                   Z2, Zn)
        """
        # number_of_islands = number_of_sequences * islands_per_sequence
        alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
                    'V', 'W', 'X', 'Y', 'Z']
        numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
                   '20', '21', '22', '23', '24']

        island_id_list = []
        for i in range(number_of_sequences):

            if i < len(alphabet):
                current_id = alphabet[i]
            elif i >= len(alphabet) and i < 2*len(alphabet):
                current_id = alphabet[0]
                current_id += alphabet[i%len(alphabet)]
            elif i >= 2*len(alphabet) and i < 3*len(alphabet):
                current_id = alphabet[1]
                current_id += alphabet[i%len(alphabet)]
            else:
                raise RuntimeError("The number of sequences is too large, expand this function to allow for that number.")

            for j in range(islands_per_sequence):
                assert islands_per_sequence <= len(numbers)
                island_id_list.append(current_id + numbers[j])

        return island_id_list


    def add_vertices_per_sequence(self, sequence_it):
        """
        This function adds vertices for a given sequence

        Parameters
        ----------
        sequence_it is the iteration of the sequence which starts at 0
        """
        for k in range(self.islands_per_sequence):
            iteration_count=k+sequence_it*self.islands_per_sequence
            id = self.vertex_names_list[iteration_count]
            self.list_of_custom_vertices.append(custom_vertex(id=id, iteration_count=iteration_count))
            self.add_vertex()

    def add_edges_per_sequence(self, sequence_it):
        """
        This function adds bi-directional edges between all islands for a given sequence

        Parameters
        ----------
        sequence_it is the iteration of the sequence which starts at 0
        """
        current_sequence_island_indices = [m + (sequence_it)*self.islands_per_sequence for m in
                                           range(self.islands_per_sequence)]
        for l in range(self.islands_per_sequence):
            first_index = current_sequence_island_indices[l]
            remaining_island_indices = [p for p in current_sequence_island_indices if p != first_index]
            for j, second_index in enumerate(remaining_island_indices):
                self.add_edge(first_index, second_index, w=self.topology_weight)
                self.num_edges +=1
        
    def add_connections(self):
        """
        This function adds all the vertices and edges for the number of sequences and islands
        """
        for i in range(self.number_of_sequences):
            self.add_vertices_per_sequence(i)
            self.add_edges_per_sequence(i)

# Testing
# topo = MGASOTopology(number_of_sequences=56, islands_per_sequence=14, topology_weight=0.01)
# topo.add_connections()
# print(topo.list_of_custom_vertices[56 * 14 - 120].id)



