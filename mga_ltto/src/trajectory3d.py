import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Union
import tudatpy
from tudatpy.kernel.numerical_simulation import environment_setup
# from tudatpy.kernel.interface import spice_interface
from tudatpy.util import result2array

def trajectory_3d(
    vehicles_states: Dict[float, np.ndarray],
    vehicles_names: List[str],
    central_body_name:str,
    bodies: List[str] = ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter",
        "Saturn", "Uranus", "Neptune"],
    frame_orientation:str = "J2000",
    center_plot:bool = False,
    colors:List[str] = [],
    linestyles:List[str] = [] ):

    # Save color and linestyle index
    i_c, i_ls = 0, 0

    # Set color and linestyles if nothing specified
    if len(colors) == 0:
        colors = ["C%i" % i for i in range(10)]
    if len(linestyles) == 0:
        linestyles = ["-"]*(len(vehicles_names)+len(bodies))

    # If the SPICE kernels are not loaded, load THEM
    # if spice_interface.get_total_count_of_kernels_loaded() < 4:
    #     spice_interface.load_standard_kernels()

    # Make sure inputs are the correct format
    if type(vehicles_names) == str:
        vehicles_names = [vehicles_names]
    # if type(spice_bodies) == str:
    #     spice_bodies = [spice_bodies]

    # Convert the states to a ndarray
    vehicles_states_array = result2array(vehicles_states)
    sim_epochs = vehicles_states_array[:,0]
    # print(sim_epochs[-1],sim_epochs[0])

    # Make a list of positions per vehicle
    vehicles_positions = []
    for i in range(len(vehicles_names)):
        vehicles_positions.append(vehicles_states_array[:, 1+i*6:4+i*6])
    # print(vehicles_positions[0][-10:-1,:])

    # Create a figure with a 3D projection for the Moon and vehicle trajectory around Earth
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Save the minimum and maximum positions
    min_pos, max_pos = 0, 0

    # Plot the trajectory of each vehicle
    for i, vehicle_name in enumerate(vehicles_names):
        if len(vehicle_name) != 0:
            # Update the minimum and maximum positions
            min_pos, max_pos = min(min_pos, np.min(vehicles_positions[i])), max(max_pos,
                    np.max(vehicles_positions[i]))
            # Select appropriate color and linestyle
            _color = colors[i_c]
            _linestyle = linestyles[i_ls]
            # Increment color and linestyle indexes
            i_c = i_c + 1 if i_c != len(colors) else 0
            i_ls = i_ls + 1 if i_ls != len(linestyles) else 0
            # Plot the trajectory of the vehicle
            ax.plot(vehicles_positions[i][:,0], vehicles_positions[i][:,1],
                    vehicles_positions[i][:,2], label=vehicle_name, color='k',
                    linestyle='-', linewidth=0.5)
            ax.scatter(vehicles_positions[i][0, 0] , vehicles_positions[i][0, 1] ,
                    vehicles_positions[i][0, 2] , marker='D', s=40, color='k')
            ax.scatter(vehicles_positions[i][-1, 0] , vehicles_positions[i][-1, 1] ,
                    vehicles_positions[i][-1, 2] , marker='D', s=40, color='k')

    body_list_settings = lambda : \
        environment_setup.get_default_body_settings(bodies=bodies,
                base_frame_origin='SSB', base_frame_orientation="ECLIPJ2000")


    for i in bodies:
        current_body_list_settings = body_list_settings()
        current_body_list_settings.add_empty_settings(i)            
        current_body_list_settings.get(i).ephemeris_settings = \
        environment_setup.ephemeris.approximate_jpl_model(i)        

    system_of_bodies = environment_setup.create_system_of_bodies(current_body_list_settings)

    for body in bodies:
        body_object = system_of_bodies.get(body)
        body_state_array = np.array([body_object.state_in_base_frame_from_ephemeris(epoch) for epoch
            in sim_epochs])
        body_state_array = body_state_array[:, 0:3]


# Old GTOP ephemeris with simplified_system_of_bodies
    # system_of_bodies = environment_setup.create_simplified_system_of_bodies()
    # for spice_body in spice_bodies:
    #     body_object = system_of_bodies.get(spice_body)
    #     body_state_array = np.array([body_object.state_in_base_frame_from_ephemeris(epoch) for epoch
    #         in sim_epochs])
    #     body_state_array = body_state_array[:, 0:3]

#Old Spice direct ephemerides
        # print(body_state_array)

        # body_state_array = np.array([
        #     spice_interface.get_body_cartesian_position_at_epoch(spice_body, central_body_name,
        #         frame_orientation, "None", epoch) for epoch in sim_epochs ])
        # print(body_state_array)

        # print(body_state_array[0, 0] - body_state_array[-1, 0])
        # Update the minimum and maximum positions
        min_pos, max_pos = min(min_pos, np.min(body_state_array)), max(max_pos, np.max(body_state_array))
        # Select appropriate color and linestyle
        _color = colors[i_c]
        _linestyle = linestyles[i_ls]
        # Increment color and linestyle indexes
        i_c = i_c + 1 if i_c != len(colors) else 0
        i_ls = i_ls + 1 if i_ls != len(linestyles) else 0
        # Plot the trajectory of the body
        ax.plot(body_state_array[:,0], body_state_array[:,1], body_state_array[:,2],
                label=body, color=_color, linestyle=_linestyle)
        ax.scatter(body_state_array[0, 0], body_state_array[0, 1],
                body_state_array[0, 2], marker="o", color=_color)
        ax.scatter(body_state_array[-1, 0], body_state_array[-1, 1],
                body_state_array[-1, 2], marker="o", color=_color)

    # Plot the central body position
    ax.scatter(0.0, 0.0, 0.0, label=central_body_name, marker="o", color="k")

    # Make the plot centered if needed
    if center_plot:
        min_pos, max_pos = -max(np.fabs(min_pos), max_pos), max(np.fabs(min_pos), max_pos)

    # Add a legend, set the plot limits, and add axis labels
    ax.legend()
    ax.set_xlim([min_pos*1.2, max_pos*1.2]), ax.set_ylim([min_pos*1.2, max_pos*1.2]), ax.set_zlim([min_pos*1.2, max_pos*1.2])
    ax.set_xlabel("x [m]"), ax.set_ylabel("y [m]"), ax.set_zlabel("z [m]")

    # Use a tight layout for the figure (do last to avoid trimming axis)
    fig.tight_layout()

    # Return the figure and the axis system
    return fig, ax