###############################################################################
# IMPORT STATEMENTS ###########################################################
###############################################################################

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

# Define order of bodies (nodes) for gravity assists
transfer_body_order = [
    'Earth', 'Venus', 'Venus', 'Earth',  'Jupiter',  'Saturn']

# Define departure and insertion orbit
departure_semi_major_axis = np.inf
departure_eccentricity = 0.

arrival_semi_major_axis = 1.0895e8 / 0.02
arrival_eccentricity = 0.98

###########################################################################
# CREATE TRANSFER SETTINGS AND OBJECT #####################################
###########################################################################

# Define trajectory settings
transfer_leg_settings, transfer_node_settings = transfer_trajectory.mga_settings_unpowered_unperturbed_legs(
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
node_times.append((-789.8117 - 0.5) * julian_day)
node_times.append(node_times[0] + 158.302027105278 * julian_day)
node_times.append(node_times[1] + 449.385873819743 * julian_day)
node_times.append(node_times[2] + 54.7489684339665 * julian_day)
node_times.append(node_times[3] + 1024.36205846918 * julian_day)
node_times.append(node_times[4] + 4552.30796805542 * julian_day)

# Define free parameters per leg (now: none)
leg_free_parameters = list()
for i in range(len(transfer_body_order)-1):
    leg_free_parameters.append(np.zeros(0))

# Define free parameters per node (now: none)
node_free_parameters = list()
for i in range(len(transfer_body_order)):
    node_free_parameters.append(np.zeros(0))

###########################################################################
# EVALUATE TRANSFER #######################################################
###########################################################################

# Evaluate transfer with given parameters
transfer_trajectory_object.evaluate(node_times, leg_free_parameters, node_free_parameters)

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

# Plot state history
fly_by_states = np.array([state_history[node_times[i]] for i in range(len(node_times))])
state_history = result2array(state_history)
au = 1.5e11

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(state_history[:, 1] / au, state_history[:, 2] / au, state_history[:, 3] / au)
ax.scatter(fly_by_states[0, 0] / au, fly_by_states[0, 1] / au, fly_by_states[0, 2] / au, color='blue', label='Earth departure')
ax.scatter(fly_by_states[1, 0] / au, fly_by_states[1, 1] / au, fly_by_states[1, 2] / au, color='brown', label='Venus fly-by')
ax.scatter(fly_by_states[2, 0] / au, fly_by_states[2, 1] / au, fly_by_states[2, 2] / au, color='brown', label='Venus fly-by')
ax.scatter(fly_by_states[3, 0] / au, fly_by_states[3, 1] / au, fly_by_states[3, 2] / au, color='green', label='Earth fly-by')
ax.scatter(fly_by_states[4, 0] / au, fly_by_states[4, 1] / au, fly_by_states[4, 2] / au, color='peru', label='Jupiter fly-by')
ax.scatter(fly_by_states[5, 0] / au, fly_by_states[5, 1] / au, fly_by_states[5, 2] / au, color='red', label='Saturn arrival')
ax.scatter([0], [0], [0], color='orange', label='Sun')
ax.set_xlabel('x wrt Sun [AU]')
ax.set_ylabel('y wrt Sun [AU]')
ax.set_zlabel('z wrt Sun [AU]')
ax.set_xlim([-10.5, 2.5])
ax.set_ylim([-8.5, 4.5])
ax.set_zlim([-6.5, 6.5])
ax.legend(bbox_to_anchor=[1.15, 1])
plt.show()
