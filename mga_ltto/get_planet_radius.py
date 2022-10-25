import sys

# sys.path.insert(0, "/Users/sean/Desktop/tudelft/tudat/tudat-bundle/build/tudatpy")

# Tudatpy imports
import tudatpy
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.trajectory_design import shape_based_thrust
from tudatpy.kernel.trajectory_design import transfer_trajectory
from tudatpy.kernel.astro import time_conversion
# from tudatpy.kernel.interface import spice
# spice.load_standard_kernels()

############################################################
### THE WAY I INITIALISE SYSTEMOFBODIES ####################
############################################################

bodies = ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn"]
# 
# frame_origin = 'Sun'
# frame_orientation = 'ECLIPJ2000'
# body_list_settings = lambda : \
#     environment_setup.get_default_body_settings(bodies=bodies,
#             base_frame_origin=frame_origin, base_frame_orientation=frame_orientation)
# 
# # current_body_list_settings = environment_setup.BodyListSettings
# for i in bodies:
#     current_body_list_settings = body_list_settings()
#     current_body_list_settings.add_empty_settings(i)            
#     current_body_list_settings.get(i).ephemeris_settings = \
#     environment_setup.ephemeris.approximate_jpl_model('Earth')        
#     print(current_body_list_settings.get(i).ephemeris_settings)
# 
# system_of_bodies = environment_setup.create_system_of_bodies(current_body_list_settings)
# # system_of_bodies = environment_setup.create_simplified_system_of_bodies()
# 
# print(system_of_bodies.get('Mars').ephemeris)

############################################################
### THIS VERSION DOESNT WORK ###############################
############################################################

bodies = ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn"]

frame_origin = 'Sun'
frame_orientation = 'ECLIPJ2000'

current_body_list_settings = environment_setup.BodyListSettings(frame_origin, frame_orientation)
for i in bodies:
    current_body_list_settings.add_empty_settings(i)            
    if i == "Sun":
        central_body_mu = 1.3271244e20 # m^3 / s^2
        current_body_list_settings.get(i).ephemeris_settings = \
        environment_setup.ephemeris.keplerian([0, 0, 0, 0, 0, 0], 1000*constants.JULIAN_DAY,
                central_body_mu, 'SSB', frame_orientation)
    else:
        current_body_list_settings.get(i).ephemeris_settings = \
        environment_setup.ephemeris.approximate_jpl_model(i)        
    print(current_body_list_settings.get(i).ephemeris_settings)

print(current_body_list_settings)

system_of_bodies = environment_setup.create_system_of_bodies(current_body_list_settings)
# system_of_bodies = environment_setup.create_simplified_system_of_bodies()

print(system_of_bodies.get('Mars').ephemeris)