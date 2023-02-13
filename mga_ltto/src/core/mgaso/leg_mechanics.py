'''
Author: Sean Cowan
Purpose: MSc Thesis
Date Created: 31-01-2023

This module includes classes that support the mgaso algorithm.
'''
###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General 
import numpy as np

# import sys
# sys.path.insert(0, "/Users/sean/Desktop/tudelft/tudat/tudat-bundle-test/build/tudatpy")

class separateLegMechanics:

    def __init__(self, mga_sequence_characters, champion_x, delta_v_per_leg) -> None:
        self.mga_sequence_characters = mga_sequence_characters
        self.champion_x = champion_x
        self.delta_v_per_leg = delta_v_per_leg

        chars = [i for i in self.mga_sequence_characters]
        self.number_of_legs = len(chars) - 1

        dict_of_legs = {}
        for i in range(self.number_of_legs):
            dict_of_legs[chars[i] + chars[i+1]] = i
        self.dict_of_legs = dict_of_legs

    def get_leg_specifics(self, leg_string) -> str:
        """
        This function returns the leg specifics of any given leg string

        Example
        -------
        Input : 'EM'
        Returns : [dV, ToF, #rev]
        """
        index = self.dict_of_legs[leg_string]

        current_dV = self.delta_v_per_leg[index]
        current_tof = self.champion_x[3+index] / 86400
        current_rev = self.champion_x[len(self.champion_x) - self.number_of_legs + index]
        current_leg_results = [leg_string, current_dV, current_tof, current_rev]
        return current_leg_results

    def get_sequence_leg_specifics(self):
        """
        This function returns a list of lists containing the information of whole sequence
        Example
        -------
        Input : 'EMMJ'
        Returns : [[EM, dV, ToF, #rev]
        [MM, dV, ToF, #rev]
        [MJ, dV, ToF, #rev]]
        """
        sequence_results = []
        for leg_string in self.dict_of_legs.keys():
            current_leg_specifics = self.get_leg_specifics(leg_string)
            sequence_results.append(current_leg_specifics)

        return sequence_results


class legDatabaseMechanics:

    def __init__(self):
        pass
        # self.leg_database = leg_data

    @staticmethod
    def get_leg_data_from_database(leg_to_compare, leg_database):
        """
        This function takes the leg_database and creates a list of results specific to that leg. 
        Returns list of leg specific design parameter vectors

        Parameters
        -----------

        leg_database : List[[str, float, float, int]]
        leg : str

        Returns
        --------

        List[np.array]
        """
        leg_data = []
        for leg_specific_data in leg_database:
            if leg_specific_data[0] == leg_to_compare:
                delta_v = leg_specific_data[1]
                tof = leg_specific_data[2]
                rev = leg_specific_data[3]
                leg_data.append(np.array([delta_v, tof, rev]))

        return leg_data

    @staticmethod
    def get_filtered_legs(leg_specific_database, no_of_predefined_individuals):
        """
        This function takes the dpv variables of a specific leg, and returns the individuals that are input
        into the island.
        Returns list of dpv variables that are to be input into the island design parameter vector

        Parameters
        -----------

        leg_specific_database : List[np.array]
        no_of_predefined_individuals : float

        Returns
        --------

        List[np.array]
        """

        leg_specific_database.sort(key=lambda dpv: dpv[0]) # sort in ascending order

        #limit to amount of predefined individuals
        leg_specific_database = leg_specific_database[:no_of_predefined_individuals] 
        # print(leg_specific_database)

        return leg_specific_database


    @staticmethod
    def get_dpv_from_leg_specifics(pre_dpv, filtered_legs, transfer_body_order, free_param_count,
            leg_count):
        """
        This function takes the filtered dpv variables and inputs them into otherwise random design
        parameter vectors.
        Filtered legs content based on get_sequence_leg_specifics function:
        0 - dV
        1 - ToF
        2 - #rev

        Parameters
        -----------

        filtered_legs : List[np.array]
        transfer_body_order : List[str]
        free_param_count : 2
        populaton : pg.population
        """
        no_of_legs = len(transfer_body_order) - 1
        no_of_gas = len(transfer_body_order) - 2
        # only loop over available leg information, if not legs exist, then just return the pre_dpv
        #the first x pre_dpv vectors are edited.
        for it, leg_info in enumerate(filtered_legs):
            pre_dpv[it][2 + leg_count] = leg_info[1]
            pre_dpv[it][2 + no_of_legs + 2 * no_of_gas + free_param_count * 3 * no_of_legs + leg_count] = leg_info[2]
        return pre_dpv

