import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Union

def solar_system_propagation(spice_bodies : List[str]=[],
                                ):
    pass

def trajectory_3d(
    vehicles_states: Dict[float, np.ndarray],
    vehicles_names: List[str],
    central_body_name:str,
    spice_bodies: List[str] = [],
    frame_orientation:str = "J2000",
    center_plot:bool = False,
    colors:List[str] = [],
    linestyles:List[str] = [] ):
    """Plot the trajectory specified bodies in 3D.

    This function allows to plot the 3D trajectory of vehicles of which the state has been propagated, as well as
    the trajectory of bodies taken directly from the SPICE interface.

    Parameters
    ----------
    vehicles_states : Dict[float, numpy.ndarray]
        Dictionary mapping the simulation time steps to the propagated state time series of the simulation bodies.
    vehicles_names : List[str]
        List of the name of the simulated body for which the trajectory must be plotted. Use an empty string in the list to skip a specific body.
    central_body_name : str
        Name of the central body in the simulation
    spice_bodies : List[str], optional
        List of the name of bodies for which the trajectory has to be taken from SPICE and plotted.
    frame_orientation : str, optional, default="J2000"
        Orientation of the reference frame used in the simulation.
    center_plot : bool, optional, default=False
        If True, the central body will be centered on the plot.
    colors : List[str], optional
        List of colors to use for the trajectories. In order, the colors will first be used for the vehicles and then for the SPICE bodies.
    linestyles : List[str], optional
        List of linestyles to use for the trajectories. In order, the linestyles will first be used for the vehicles and then for the SPICE bodies.
    

    Examples
    --------
    After the propagation of two CubeSats on which thrust is applied, we can for instance plot both of their trajectories, as well as the trajectory of the Moon,
    using the following code snippet:

    .. code-block:: python

        # Plot the trajectory of two satellite and the Moon around the Earth
        fig, ax = plotting.trajectory_3d(
            vehicles_states=dynamics_simulator.state_history,
            vehicles_names=["Lunar CubeSat A", "Lunar CubeSat B"],
            central_body_name="Earth",
            spice_bodies=["Moon"],
            linestyles=["dotted", "dashed", "solid"],
            colors=["blue", "green", "grey"]
        )
        # Change the size of the figure
        fig.set_size_inches(8, 8)
        # Show the plot
        plt.show()

    .. image:: _static/trajectory_3D.png
       :width: 450
       :align: center

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the 3D plot.
    ax : matplotlib.axes.Axes
        3D axis system used to plot the trajectory.

    """
    # Import SPICE
    from tudatpy.kernel.interface import spice_interface
    from tudatpy.util import result2array

    # Save color and linestyle index
    i_c, i_ls = 0, 0

    # Set color and linestyles if nothing specified
    if len(colors) == 0:
        colors = ["C%i" % i for i in range(10)]
    if len(linestyles) == 0:
        linestyles = ["-"]*(len(vehicles_names)+len(spice_bodies))

    # If the SPICE kernels are not loaded, load THEM
    if spice_interface.get_total_count_of_kernels_loaded() < 4:
        spice_interface.load_standard_kernels()

    # Make sure inputs are the correct format
    if type(vehicles_names) == str:
        vehicles_names = [vehicles_names]
    if type(spice_bodies) == str:
        spice_bodies = [spice_bodies]

    # Convert the states to a ndarray
    vehicles_states_array = result2array(vehicles_states)
    sim_epochs = vehicles_states_array[:,0]
    # print(sim_epochs[-1]-sim_epochs[0])

    # Make a list of positions per vehicle
    vehicles_positions = []
    for i in range(len(vehicles_names)):
        vehicles_positions.append(vehicles_states_array[:, 1+i*6:4+i*6])

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
                    linestyle='-.')
            ax.scatter(vehicles_positions[i][0, 0] , vehicles_positions[i][0, 1] ,
                    vehicles_positions[i][0, 2] , marker='o', color=_color)
            ax.scatter(vehicles_positions[i][-1, 0] , vehicles_positions[i][-1, 1] ,
                    vehicles_positions[i][-1, 2] , marker='o', color=_color)

    for spice_body in spice_bodies:
    # spice_body = "Jupiter"
    # Get the position of the body from SPICE
        body_state_array = np.array([
            spice_interface.get_body_cartesian_position_at_epoch(spice_body, central_body_name,
                frame_orientation, "None", epoch) for epoch in sim_epochs ])
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
                label=spice_body, color=_color, linestyle=_linestyle)
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
