
# Tudatpy imports
import tudatpy
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.trajectory_design import shape_based_thrust
from tudatpy.kernel.trajectory_design import transfer_trajectory
from tudatpy.kernel.astro import time_conversion
from tudatpy.kernel.interface import spice

spice.load_standard_kernels()

bodies_to_create = ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn"]
planetary_radii = {}
for i in bodies_to_create:
    planetary_radii[i] = spice.get_average_radius(i)
print(planetary_radii)
