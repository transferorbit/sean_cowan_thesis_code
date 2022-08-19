'''
Author: Sean Cowan
Purpose: MSc Thesis
Date Created: 02-08-2022

This module performs post processing actions to visualize the optimized results.
'''
###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import numpy as np
import os
import pygmo as pg
import matplotlib.pyplot as plt

# Tudatpy imports
#import tudatpy
#from tudatpy.io import save2txt
#from tudatpy.kernel import constants
#from tudatpy.kernel.numerical_simulation import propagation_setup
#from tudatpy.kernel.numerical_simulation import environment_setup
#from tudatpy.kernel.math import interpolators
#from tudatpy.kernel.trajectory_design import shape_based_thrust
#from tudatpy.kernel.trajectory_design import transfer_trajectory
from trajectory3d import trajectory_3d

import mga_low_thrust_utilities as mga_util
from pygmo_problem import MGALowThrustTrajectoryOptimizationProblem

state_history = np.loadtxt('test_optimization_results/state_history.dat')
state_history_dict = {}
for i in range(len(state_history)):
        state_history_dict[state_history[i, 0]] = state_history[i,1:]

node_times = np.loadtxt('test_optimization_results/node_times.dat')
print(node_times[0, 1])

fly_by_states = np.array([state_history_dict[node_times[i, 1]] for i in range(len(node_times))])

current_dir = os.getcwd()
fig, ax = trajectory_3d(
    state_history_dict,
    vehicles_names=["Spacecraft"],
    central_body_name="SSB",
    spice_bodies=["Earth", "Venus", "Jupiter", "Saturn"],
    frame_orientation= 'ECLIPJ2000',
)
# Change the size of the figure
ax.scatter(fly_by_states[0, 0] , fly_by_states[0, 1] , fly_by_states[0,
    2] , marker='o', color='blue', label='Earth departure')
ax.scatter(fly_by_states[1, 0] , fly_by_states[1, 1] , fly_by_states[1,
    2] , marker='o', color='brown', label='Mars fly-by')

fig.set_size_inches(8, 8)
# Show the plot
plt.show()

