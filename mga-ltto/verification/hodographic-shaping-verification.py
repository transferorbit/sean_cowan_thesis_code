# Load standard modules
import numpy as np
from matplotlib import pyplot as plt
import os

import sys
sys.path.insert(0, "/Users/sean/Desktop/tudelft/tudat/tudat-bundle/build/tudatpy")

# Load tudatpy modules
from tudatpy.io import save2txt
from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel import constants
from tudatpy.util import result2array
from itertools import islice

parent = os.getcwd()
print(parent)
sys.path.append(parent)
from trajectory3d import trajectory_3d
# import mga_low_thrust_utilities as mga_util

# Load spice kernels
spice.load_standard_kernels()

data_directory = "./verification/tdep9200_vdep160_tof1185_rev3/island_0/"
thrust_array = np.loadtxt(data_directory + "thrust_acceleration.dat")

current_dir = os.getcwd()
write_results_to_file = True
plotting = True

subdirectory = '/verification/numerical_results/'

julian_day = constants.JULIAN_DAY
dep_vel = 150

thrust_magnitude = {}
specific_impulse = {}
for i in range(10):
    thrust_magnitude[thrust_array[i, 0]] = np.linalg.norm(thrust_array[i, 1:])
    specific_impulse[thrust_array[i, 0]] = 4000

# def take(n, iterable):
#     "Return first n items of the iterable as a list"
#     return list(islice(iterable, n))
# n_items = take(0, thrust_magnitude.items())
#
# print(n_items)
# def thrust_magnitude(time, thrust_direction=thrust_direction):
#     thrust_magnitude = np.array([np.linalg.norm(thrust_direction[i, 1:]) for i in
#         range(len(thrust_direction))])
#     return thrust_magnitude[time]
#
# def specific_impulse(time, thrust_direction=thrust_direction):
#     specific_impulse = np.array([4000 for _ in range(len(thrust_direction))])
#     return specific_impulse[time]
#

# print(thrust_magnitude)
thrust_magnitude_lambda = lambda t: thrust_magnitude[t] #this t should be the index, not epoch dictionary?
specific_impulse_lambda = lambda t: specific_impulse[t]

# Set simulation start and end epochs (total simulation time of 30 days)
simulation_start_epoch = 9200*julian_day
simulation_end_epoch = simulation_start_epoch + 1185*julian_day

print(simulation_start_epoch, simulation_end_epoch)

# Define bodies in simulation
bodies_to_create = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune',
        'Sun'] 

# Create bodies in simulation
body_settings = environment_setup.get_default_body_settings(bodies_to_create)
system_of_bodies = environment_setup.create_system_of_bodies(body_settings)


# Create the Spacecraft body in the environment
system_of_bodies.create_empty_body("Spacecraft")
system_of_bodies.get_body("Spacecraft").set_constant_mass(1e3)

# Define bodies that are propagated
bodies_to_propagate = ["Spacecraft"]

# Define central bodies of propagation
central_bodies = ["SSB"]

# Define the direction of the thrust as colinear with the velocity of the orbiting Spacecraft, pushing it from behind
thrust_direction_settings = (
    propagation_setup.thrust.thrust_direction_from_state_guidance(
        central_body="SSB",
        is_colinear_with_velocity=True,
        direction_is_opposite_to_vector=False ) )

# Define the thrust magnitude as constant
# thrust_magnitude_settings = (
#     propagation_setup.thrust.constant_thrust_magnitude(
#         thrust_magnitude=10.0, specific_impulse=5e3 ) )

# thrust_magnitude_settings = \
#         propagation_setup.thrust.custom_thrust_magnitude(thrust_magnitude_function=thrust_magnitude_lambda,
                # specific_impulse_function=specific_impulse_lambda)
thrust_magnitude_settings = \
        propagation_setup.thrust.constant_thrust_magnitude(thrust_magnitude=1e-5,
                specific_impulse=4e3)

# Define the accelerations acting on the Spacecraft
acceleration_on_spacecraft = dict(
    Spacecraft=[
        # Define the thrust acceleration from its direction and magnitude
        propagation_setup.acceleration.thrust_from_direction_and_magnitude(
            thrust_direction_settings=thrust_direction_settings,
            thrust_magnitude_settings=thrust_magnitude_settings,
        )
    ],
    # Define the acceleration due to the Earth, mars, and Sun as Point Mass
    Earth=[propagation_setup.acceleration.point_mass_gravity()],
    Mars=[propagation_setup.acceleration.point_mass_gravity()],
    Sun=[propagation_setup.acceleration.point_mass_gravity()]
)

# Compile the accelerations acting on the Spacecraft
acceleration_dict = dict(Spacecraft=acceleration_on_spacecraft)

# Create the acceleration models from the acceleration mapping dictionary
acceleration_models = propagation_setup.create_acceleration_models(
    body_system=system_of_bodies,
    selected_acceleration_per_body=acceleration_dict,
    bodies_to_propagate=bodies_to_propagate,
    central_bodies=central_bodies
)

planetary_radii = {}
for i in bodies_to_create:
    planetary_radii[i] = spice.get_average_radius(i)
planetary_radii = planetary_radii
# Get system initial state (in cartesian coordinates)
system_initial_state = np.array([planetary_radii["Earth"]+2e5, 0, 0, (7.8e3+dep_vel), 0, 0])

# Create a dependent variable to save the altitude of the Spacecraft w.r.t. Earth over time
spacecraft_altitude_dep_var = propagation_setup.dependent_variable.altitude( "Spacecraft", "Earth" )

# Create a dependent variable to save the mass of the spacecraft over time
spacecraft_mass_dep_var = propagation_setup.dependent_variable.body_mass( "Spacecraft" )

# Define list of dependent variables to save
dependent_variables_to_save = [spacecraft_altitude_dep_var, spacecraft_mass_dep_var]

# Create a termination setting to stop when altitude of the Spacecraft is above 100e3 km
# termination_distance_settings = propagation_setup.propagator.dependent_variable_termination(
#         dependent_variable_settings = spacecraft_altitude_dep_var,
#         limit_value = 100e6,
#         use_as_lower_limit = False)

# Create a termination setting to stop when the Spacecraft has a mass below 4e3 kg
termination_mass_settings = propagation_setup.propagator.dependent_variable_termination(
        dependent_variable_settings = spacecraft_mass_dep_var,
        limit_value = 800.0,
        use_as_lower_limit = True)

# Create a termination setting to stop at the specified simulation end epoch
termination_time_settings = propagation_setup.propagator.time_termination(simulation_end_epoch)

# Setup a hybrid termination setting to stop the simulation when one of the aforementionned termination setting is reached
termination_condition = propagation_setup.propagator.hybrid_termination(
    [termination_mass_settings, termination_time_settings],
    fulfill_single_condition = True)

# Create the translational propagation settings (use a Cowell propagator)
translational_propagator_settings = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models,
    bodies_to_propagate,
    system_initial_state,
    termination_condition,
    propagation_setup.propagator.cowell,
    output_variables=[spacecraft_altitude_dep_var, spacecraft_mass_dep_var]
)

# Create a mass rate model so that the Spacecraft looses mass according to how much thrust acts on it
mass_rate_settings = dict(Spacecraft=[propagation_setup.mass_rate.from_thrust()])
mass_rate_models = propagation_setup.create_mass_rate_models(
    system_of_bodies,
    mass_rate_settings,
    acceleration_models
)
# Create the mass propagation settings
mass_propagator_settings = propagation_setup.propagator.mass(
    bodies_to_propagate,
    mass_rate_models,
    [1e3], # initial Spacecraft mass
    termination_condition )

# Combine the translational and mass propagator settings
propagator_settings = propagation_setup.propagator.multitype(
    [translational_propagator_settings, mass_propagator_settings],
    termination_condition,
    [spacecraft_altitude_dep_var, spacecraft_mass_dep_var])

# Setup the variable step integrator time step sizes
initial_time_step = 10.0
minimum_time_step = 0.001
maximum_time_step = 86400
# Setup the tolerance of the variable step integrator
tolerance = 1e-10

# Create numerical integrator settings (using a RKF7(8) coefficient set)
# integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(
#     simulation_start_epoch,
#     initial_time_step,
#     propagation_setup.integrator.rkf_78,
#     minimum_time_step,
#     maximum_time_step,
#     relative_error_tolerance=tolerance,
#     absolute_error_tolerance=tolerance)

integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(
        simulation_start_epoch,
        1000,
        propagation_setup.integrator.rk_4)



# Instantiate the dynamics simulator and run the simulation
dynamics_simulator = numerical_simulation.SingleArcSimulator(
    system_of_bodies, integrator_settings, propagator_settings, print_dependent_variable_data=True
)

# Extract the state and dependent variable history
state_history = dynamics_simulator.state_history
dependent_variable_history = dynamics_simulator.dependent_variable_history



if write_results_to_file:
    output_path = current_dir + subdirectory 
else:
    output_path = None

if write_results_to_file:
    save2txt(state_history, 'state_history.dat', output_path)
    save2txt(dependent_variable_history, 'dependent_variable_history.dat', output_path)

if plotting:
    # Convert the dictionaries to multi-dimensional arrays
    spacecraft_array = result2array(state_history)
    dep_var_array = result2array(dependent_variable_history)

    # Retrieve the mars trajectory over Spacecraft propagation epochs from spice
    mars_states_from_spice = {
        epoch:spice.get_body_cartesian_state_at_epoch("Mars", "Earth", "J2000", "None", epoch)
        for epoch in list(state_history.keys())
    }
    # Convert the dictionary to a mutli-dimensional array
    mars_array = result2array(mars_states_from_spice)

    # Convert the time to days
    time_days = (spacecraft_array[:,0] - spacecraft_array[0,0])/constants.JULIAN_DAY

#     # Create a figure for the altitude of the Spacecraft above Earth
#     fig1 = plt.figure(figsize=(9, 5))
#     ax1 = fig1.add_subplot(111)
#     ax1.set_title(f"Spacecraft altitude above Earth")
# 
#     # Plot the altitude of the Spacecraft over time
#     ax1.plot(time_days, dep_var_array[:,1]/1e3)
# 
#     # Add a grid and axis labels to the plot
#     ax1.grid(), ax1.set_xlabel("Simulation time [day]"), ax1.set_ylabel("Spacecraft altitude [km]")
# 
#     # Use a tight layout for the figure (do last to avoid trimming axis)
#     fig1.tight_layout()
# 
#     # Create a figure for the altitude of the Spacecraft above Earth
#     fig2 = plt.figure(figsize=(9, 5))
#     ax2 = fig2.add_subplot(111)
#     ax2.set_title(f"Spacecraft mass over time")
# 
#     # Plot the mass of the Spacecraft over time
#     ax2.plot(time_days, dep_var_array[:,2])
# 
#     # Add a grid and axis labels to the plot
#     ax2.grid(), ax2.set_xlabel("Simulation time [day]"), ax2.set_ylabel("Spacecraft mass [kg]")
# 
#     # Use a tight layout for the figure (do last to avoid trimming axis)
#     fig2.tight_layout()
# 
#     # Create a figure with a 3D projection for the mars and Spacecraft trajectory around Earth
#     fig3 = plt.figure(figsize=(8, 8))
#     ax3 = fig3.add_subplot(111, projection="3d")
#     ax3.set_title(f"System state evolution in 3D")
# 
#     # Plot the Spacecraft and mars positions as curve, and the Earth as a marker
#     ax3.plot(spacecraft_array[:,1], spacecraft_array[:,2], spacecraft_array[:,3], label="Spacecraft", linestyle="-.", color="green")
#     ax3.plot(mars_array[:,1], mars_array[:,2], mars_array[:,3], label="Mars", linestyle="-", color="grey")
#     ax3.scatter(0.0, 0.0, 0.0, label="Earth", marker="o", color="blue")
# 
#     # Add a legend, set the plot limits, and add axis labels
#     ax3.legend()
#     ax3.set_xlim([-3E8, 3E8]), ax3.set_ylim([-3E8, 3E8]), ax3.set_zlim([-3E8, 3E8])
#     ax3.set_xlabel("x [m]"), ax3.set_ylabel("y [m]"), ax3.set_zlabel("z [m]")

    # data_directory = "verification/numerical_results/"
    # mga_util.hodographic_shaping_visualisation(dir=data_directory, trajectory_function=mga_util.trajectory_3d)

    trajectory_3d(state_history,
            vehicles_names=["Spacecraft"],
            central_body_name="SSB",
            spice_bodies=["Sun", "Mercury", "Venus", "Earth", "Mars"],
            frame_orientation= 'ECLIPJ2000')


    # Use a tight layout for the figure (do last to avoid trimming axis)
    # fig3.tight_layout()

    plt.show()

