###############################################################################
# IMPORT STATEMENTS ###########################################################
###############################################################################
# import build_directory

import numpy as np
import matplotlib.pyplot as plt
from tudatpy.kernel.trajectory_design import transfer_trajectory
from tudatpy.kernel.trajectory_design import shape_based_thrust
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.util import result2array
from tudatpy.kernel import constants
from tudatpy.kernel.math import root_finders

###########################################################################
# DEFINE TRANSFER SETTINGS ################################################
###########################################################################

# Simplified bodies
bodies = environment_setup.create_simplified_system_of_bodies()
central_body = 'Sun'

# Define order of bodies (nodes)
transfer_body_order = ['Earth', 'Earth', 'Venus', 'Venus',  'Mercury']

# Define departure and insertion orbit
departure_semi_major_axis = np.inf
departure_eccentricity = 0

arrival_semi_major_axis = np.inf
arrival_eccentricity = 0

###########################################################################
# CREATE TRANSFER SETTINGS AND OBJECT #####################################
###########################################################################

# Define type of leg between bodies
transfer_leg_settings, transfer_node_settings = transfer_trajectory.mga_settings_dsm_velocity_based_legs(
    transfer_body_order,
    departure_orbit=(departure_semi_major_axis, departure_eccentricity),
    arrival_orbit=(arrival_semi_major_axis, arrival_eccentricity))

# Create transfer calculation object
transfer_trajectory_object = transfer_trajectory.create_transfer_trajectory(
    bodies,
    transfer_leg_settings,
    transfer_node_settings,
    transfer_body_order,
    central_body)

###########################################################################
# DEFINE TRANSFER PARAMETERS ##############################################
###########################################################################

# Define times at each node
julian_day = constants.JULIAN_DAY
node_times = list()
node_times.append((1171.64503236 - 0.5) * julian_day)
node_times.append(node_times[0] + 399.999999715 * julian_day)
node_times.append(node_times[1] + 178.372255301 * julian_day)
node_times.append(node_times[2] + 299.223139512 * julian_day)
node_times.append(node_times[3] + 180.510754824 * julian_day)

# Define free parameters per leg
leg_free_parameters = list()
leg_free_parameters.append(np.array([0.234594654679]))
leg_free_parameters.append(np.array([0.0964769387134]))
leg_free_parameters.append(np.array([0.829948744508]))
leg_free_parameters.append(np.array([0.317174785637]))

# Define free parameters per node
node_free_parameters = list()
node_free_parameters.append(np.array([1408.99421278, 0.37992647165 * 2.0 * 3.14159265358979, np.arccos(2.0 * 0.498004040298 - 1.0) - 3.14159265358979 / 2.0]))
node_free_parameters.append(np.array([1.80629232251 * 6.378e6, 1.35077257078, 0.0]))
node_free_parameters.append(np.array([3.04129845698 * 6.052e6, 1.09554368115, 0.0]))
node_free_parameters.append(np.array([1.10000000891 * 6.052e6, 1.34317576594, 0.0]))
node_free_parameters.append(np.array([]))

###########################################################################
# EVALUATE TRANSFER #######################################################
###########################################################################

# Evaluate transfer with given parameters
transfer_trajectory_object.evaluate( node_times, leg_free_parameters, node_free_parameters)

# Extract and print computed Delta V and time of flight
print('Delta V [m/s]: ', transfer_trajectory_object.delta_v)
print('Time of flight [day]: ', transfer_trajectory_object.time_of_flight / julian_day)
print()
print('Delta V per leg [m/s] : ', transfer_trajectory_object.delta_v_per_leg)
print('Delta V per node [m/s] : ', transfer_trajectory_object.delta_v_per_node)

transfer_trajectory.print_parameter_definitions(transfer_leg_settings, transfer_node_settings)

# Extract state and thrust acceleration history
state_history = transfer_trajectory_object.states_along_trajectory(500)
thrust_acceleration_history = transfer_trajectory_object.inertial_thrust_accelerations_along_trajectory(500)

fly_by_states = np.array([state_history[node_times[i]] for i in range(len(node_times))])
state_history = result2array(state_history)
au = 1.5e11

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(state_history[:, 1] / au, state_history[:, 2] / au)
ax.scatter(fly_by_states[0, 0] / au, fly_by_states[0, 1] / au, color='blue', label='Earth departure')
ax.scatter(fly_by_states[1, 0] / au, fly_by_states[1, 1] / au, color='green', label='Earth fly-by')
ax.scatter(fly_by_states[2, 0] / au, fly_by_states[2, 1] / au, color='brown', label='Venus fly-by')
ax.scatter(fly_by_states[3, 0] / au, fly_by_states[3, 1] / au, color='brown')
ax.scatter(fly_by_states[4, 0] / au, fly_by_states[4, 1] / au, color='grey', label='Mercury arrival')
ax.scatter([0], [0], color='orange', label='Sun')
ax.set_xlabel('x wrt Sun [AU]')
ax.set_ylabel('y wrt Sun [AU]')
#ax.set_aspect('equal')
ax.legend(bbox_to_anchor=[1, 1])
plt.show()

