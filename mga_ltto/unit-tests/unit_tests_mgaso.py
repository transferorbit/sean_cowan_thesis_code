'''
Author: Sean Cowan
Purpose: MSc Thesis
Date Created: 11-10-2022

This module performs unit tests for the mgso optimisation process.
'''

# General imports
import sys
import os

from tudatpy.kernel import constants

sys.path.append('../src/')
import manual_topology as topo
import unittest as unit

###########################################################################
# General parameters ######################################################
###########################################################################

write_results_to_file = False
julian_day = constants.JULIAN_DAY
seed = 421

####################################################################
# MGSO Unit tests ##################################################
####################################################################

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

class unitTestsMGSO(unit.TestCase):


    def test_011_EJ_2fp(self):
        """
        0 GA 1 SEQREC 1 SEQPERPLAN
        EJ
        2 FP
        gen, pop = 1, 100
        bounds = [[9000, 0, 200, 0, 2e2, -10**4, 0],
                [9200, 0, 1200, 7000, 2e11, 10**4, 4]]
        """
        max_no_of_gas = 0
        no_of_sequence_recursions = 1
        number_of_sequences_per_planet = [1 for _ in range(max_no_of_gas)]
        manual_base_functions = False
        leg_exchange = False
        
        ## Specific parameters
        Isp = 3200
        m0 = 1300
        departure_planet = "Earth"
        arrival_planet = "Jupiter"
        free_param_count = 2
        num_gen = 1
        pop_size = 100
        # assert pop_size > 62
        no_of_points = 1000
        bounds = [[9000, 0, 0, 200, 0, 2e2, -10**4, 0],
                [9200, 0, 0, 1200, 7000, 2e11, 10**4, 2]]
        
        topo.run_mgso_optimisation(departure_planet=departure_planet,
                                    arrival_planet=arrival_planet,
                                    free_param_count=free_param_count,
                                    Isp=Isp,
                                    m0=m0,
                                    num_gen=num_gen,
                                    pop_size=pop_size,
                                    no_of_points=no_of_points,
                                    bounds=bounds,
                                    max_no_of_gas=max_no_of_gas,
                                    no_of_sequence_recursions=no_of_sequence_recursions,
                                    number_of_sequences_per_planet=number_of_sequences_per_planet,
                                    seed=seed,
                                    manual_base_functions=manual_base_functions,
                                    leg_exchange=leg_exchange)
        self.assertEqual(True, True)

    def test_111_EJ_2fp(self):
        """
        1 GA 1 SEQREC 1 SEQPERPLAN
        EJ
        2 FP
        gen, pop = 1, 100
        bounds = [[9000, 0, 200, 0, 2e2, -10**4, 0],
                [9200, 0, 1200, 7000, 2e11, 10**4, 4]]
        """
        max_no_of_gas = 1
        no_of_sequence_recursions = max_no_of_gas
        number_of_sequences_per_planet = [1 for _ in range(max_no_of_gas)]
        manual_base_functions = False
        leg_exchange = False
        
        ## Specific parameters
        Isp = 3200
        m0 = 1300
        departure_planet = "Earth"
        arrival_planet = "Jupiter"
        free_param_count = 2
        num_gen = 1
        pop_size = 100
        assert pop_size > 62
        no_of_points = 1000
        bounds = [[9000, 0, 0, 200, 0, 2e2, -10**4, 0],
                [9200, 0, 0, 1200, 7000, 2e11, 10**4, 4]]
        
        topo.run_mgso_optimisation(departure_planet=departure_planet,
                                    arrival_planet=arrival_planet,
                                    free_param_count=free_param_count,
                                    Isp=Isp,
                                    m0=m0,
                                    num_gen=num_gen,
                                    pop_size=pop_size,
                                    no_of_points=no_of_points,
                                    bounds=bounds,
                                    max_no_of_gas=max_no_of_gas,
                                    no_of_sequence_recursions=no_of_sequence_recursions,
                                    number_of_sequences_per_planet=number_of_sequences_per_planet,
                                    seed=seed,
                                    manual_base_functions=manual_base_functions,
                                    leg_exchange=leg_exchange)
        self.assertEqual(True, True)



if __name__ == '__main__': 
    blockPrint()
    unit.main()
    enablePrint()
