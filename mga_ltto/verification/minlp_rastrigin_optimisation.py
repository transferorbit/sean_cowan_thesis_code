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
    output_directory = current_dir + '/test_minlp_batch'
    write_results_to_file = False
    
###########################################################################
# General parameters ######################################################
###########################################################################
    
    """
    Number of generations increases the amount of time spent in parallel computing the different islands
    Number of evolutions requires redefining the islands which requires more time on single thread.
    Population size; unknown
    """
    
    seed = 421
    
    my_problem = pg.minlp_rastrigin(5,5) 
    print(f'Best known solution : {my_problem.best_known()}')

    pop_size = 500
    prob = my_problem #optimisation verification
    mp.freeze_support()
    cpu_count = os.cpu_count() # not very relevant because differnent machines + asynchronous
    number_of_islands = 1
###################################################################
# LTTO Optimisation ###############################################
###################################################################

    pop = pg.population(prob, size=pop_size, seed=seed)
    algo = pg.algorithm(pg.gaco())
    algo.set_verbosity(1)
    # algo = pg.algorithm(pg.sga(gen=1))

    print('Evolution started')
    pop = algo.evolve(pop)
    print('Evolution finished')



###########################################################################
# Post processing #########################################################
###########################################################################

