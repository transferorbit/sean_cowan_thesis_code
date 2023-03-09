'''
Author: Sean Cowan
Purpose: MSc Thesis
Date Created: 14-02-2023

This module is supposed to import the relevant ephemeris data of the Bepicolombo spacecraft from
https://naif.jpl.nasa.gov/pub/naif/BEPICOLOMBO/kernels/spk/. The output of this file can then be compared with the
hodographic shaping leg meant as validation.
'''

import numpy as np
import sys
import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import argparse

from tudatpy.kernel.interface import spice
import tudatpy.kernel.io as io
from tudatpy.util import result2array
from tudatpy.io import save2txt
from tudatpy.kernel import constants

sys.path.append('/Users/sean/Desktop/tudelft/thesis/code/mga_ltto/src/') # this only works if you run ltto and mgso while in the directory that includes those files
from misc.date_conversion import dateConversion
import misc.trajectory3d as traj

#plt parameters setup
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = "Arial"
plt.rcParams['figure.figsize'] = (12, 8)

# Setup directories
current_dir = os.getcwd()
output_directory = current_dir + '/pp_validation'

bepi_spk_files = current_dir + "/validation/BEPICOLOMBO/kernels/spk"
bepi_ck_files = current_dir + "/validation/BEPICOLOMBO/kernels/ck"

# Parse ID for file reproduction
parser = argparse.ArgumentParser(description='This file runs an LTTO process')
parser.add_argument('--id', default='0', dest='id', action='store', required=False)
args = parser.parse_args()
id = args.id
subdirectory = f'/bepi_data_test{id}'

# io.get_spice_kernel_path()
spice.load_standard_kernels()

# Load files
onlyfiles = [f for f in listdir(bepi_spk_files) if isfile(join(bepi_spk_files, f))]
for file in onlyfiles:
    spice.load_kernel(current_dir + f"/validation/BEPICOLOMBO/kernels/spk/{file}")
onlyfiles = [f for f in listdir(bepi_ck_files) if isfile(join(bepi_ck_files, f))]
for file in onlyfiles:
    spice.load_kernel(current_dir + f"/validation/BEPICOLOMBO/kernels/ck/{file}")


#Setup dates and times to get data
no_of_points = 100
write_results_to_file = True
# it 0
# begin_date = dateConversion(calendar_date="2018, 10, 20")
# begin_epoch = begin_date.date_to_mjd() * 86400
# end_date = dateConversion(calendar_date="2023, 2, 16")
# end_epoch = end_date.date_to_mjd() * 86400
#it 1
# begin_epoch = 6.4*10**8 - 1e6
# end_epoch = 6.4*10**8 +1e6

#it 2
# begin_epoch = 6.3955*10**8
# end_epoch = 6.4004*10**8

# it 3 (based on esa article)
begin_date = dateConversion(calendar_date="2018, 11, 20")
begin_epoch = begin_date.date_to_mjd(time=[11, 33, 00]) * 86400 
end_date = dateConversion(calendar_date="2018, 11, 20")
end_epoch = begin_date.date_to_mjd(time=[16, 33, 00]) * 86400 

# print(end_epoch / 86400 - begin_epoch / 86400)
times = np.linspace(begin_epoch, end_epoch, no_of_points)


# print(pos)
# BEPICOLOMBO did not work, it said insufficient data loaded. There was also a bp
# naif_id = spice.convert_body_name_to_naif_id("BEPICOLOMBO MTM")
# print(naif_id)

bepi_state = {}
bepi_pos = {}
bepi_vel = {}
velocity_norm = {}
cylin_pos = {}
cylin_vel = {}
cylin_acc = {}
cylin_acc_thrust = {}
bepi_acc= {}
bepi_acc_thrust= {}
acceleration_norm = {}
velocity_diff = {}
velocity_diff_sum = 0

def cartesian_to_cylindrical(pos, vel):
    rho = np.sqrt(pos[0]**2 + pos[1]**2)
    theta = np.arctan2(pos[1],pos[0])
    rho_v = np.sqrt(vel[0]**2 + vel[1]**2)
    theta_v = np.arctan2(vel[1],vel[0])
    return np.array([rho, theta, pos[2]]), np.array([rho_v, theta_v, vel[2]])

# Fill arrays
for it, i in enumerate(times):
    bepi_state[i] = spice.get_body_cartesian_state_at_epoch("-121", "SSB", "ECLIPJ2000", "NONE", i)
    bepi_pos[i] = bepi_state[i][0:3]
    bepi_vel[i] = bepi_state[i][3:6]
    velocity_norm[i] = np.linalg.norm(bepi_vel[i])
    cylin_pos[i], cylin_vel[i] = cartesian_to_cylindrical(bepi_pos[i], bepi_vel[i])
    if it == 0:
        pass
    elif it == 1:
        bepi_acc_inertial = np.array([(bepi_vel[times[it]][k] - bepi_vel[times[it-1]][k])/(times[it]-times[it-1]) for k
                                      in range(3)])
        acceleration_norm[i] = np.linalg.norm(bepi_vel[i])

        cylin_acc_inertial = np.array([(cylin_vel[times[it]][k] - cylin_vel[times[it-1]][k])/(times[it]-times[it-1]) for
                                       k in range(3)])

        velocity_diff[i] = np.abs(velocity_norm[times[it]] - velocity_norm[times[it-1]])
        velocity_diff_sum +=velocity_diff[i]

    else:
        bepi_acc[i] = np.array([(bepi_vel[times[it]][k] - bepi_vel[times[it-1]][k])/(times[it]-times[it-1]) for k in
                                range(3)])
        bepi_acc_thrust[i] = np.subtract(np.array([(bepi_vel[times[it]][k] -
                                                    bepi_vel[times[it-1]][k])/(times[it]-times[it-1]) for k in
                                                   range(3)]), bepi_acc_inertial)
        acceleration_norm[i] = np.linalg.norm(bepi_vel[i])

        cylin_acc[i] = np.array([(cylin_vel[times[it]][k] - cylin_vel[times[it-1]][k])/(times[it]-times[it-1]) for k in
                                 range(3)])
        cylin_acc_thrust[i] = np.subtract(np.array([(cylin_vel[times[it]][k] -
                                                    cylin_vel[times[it-1]][k])/(times[it]-times[it-1]) for k in
                                                   range(3)]), cylin_acc_inertial)

        velocity_diff[i] = np.abs(velocity_norm[times[it]] - velocity_norm[times[it-1]])
        velocity_diff_sum +=velocity_diff[i]

# bepi_acc[times[0]] = np.array([bepi_acc[times[2]][k] for k in range(3)]) 
# bepi_acc[times[1]] = np.array([bepi_acc[times[2]][k] for k in range(3)]) 
# print(bepi_states)
velocity_array = result2array(velocity_norm)
acceleration_array = result2array(acceleration_norm)
cylin_acc_array = result2array(cylin_acc)
cylin_acc_thrust_array = result2array(cylin_acc_thrust)
# print(f'Delta V sum : {velocity_diff_sum}')

# Save files
if write_results_to_file:
    save2txt(bepi_state, 'bepi_state.dat', output_directory + subdirectory)
    save2txt(bepi_pos, 'bepi_position.dat', output_directory + subdirectory)
    save2txt(bepi_vel, 'bepi_velocity.dat', output_directory + subdirectory)
    save2txt(bepi_acc, 'bepi_acceleration.dat', output_directory + subdirectory)


# bepi_array = result2array(bepi_states)
# plt.plot(bepi_array[:, 0], bepi_array[:, 1])
# plt.plot(bepi_array[:, 0], bepi_array[:, 2])
# plt.plot(bepi_array[:, 0], bepi_array[:, 3])


# Plot bepi trajectory
# traj.trajectory_3d(bepi_pos,
#               "Bepicolombo",
#               "SSB",
#               bodies=["Sun", "Mercury", "Venus", "Earth"],
#               save_fig=True,
#               projection='xy') 

# traj.trajectory_2d(bepi_pos,
#               "Bepicolombo",
#               "SSB",
#               bodies=["Sun", "Mercury", "Venus", "Earth"],
#               save_fig=True) 
#
# plt.show()

# plt.plot(velocity_array[:, 0], velocity_array[:, 1])
# print(acceleration_r_array)
# fig, ax = plt.subplots(1, 1)
# ax.plot(times[2:] - begin_epoch, cylin_acc_thrust_array[:, 1] * 1000, label='radial')
# ax.plot(times[2:] - begin_epoch, cylin_acc_thrust_array[:, 2] * 1000, label='normal')
# ax.plot(times[2:] - begin_epoch, cylin_acc_thrust_array[:, 3] * 1000, label='axial')
# ax.grid()
# ax.set_xlabel("Time [s]")
# ax.set_ylabel("Acceleration [mm/s^2]")
# # ax.yscale('log')
# ax.legend()
# # ax.show()
#
# fig.savefig(current_dir + '/figures/bepi_thrust_full.jpg', bbox_inches="tight")
# print(pos)
# BEPICOLOMBO did not work, it said insufficient data loaded. There was also a bp
# naif_id = spice.convert_body_name_to_naif_id("bepi")
# print(naif_id)
# print(spice.check_body_property_in_kernel_pool("-203", "RADII"))


# def see_available_kernels():

# def add_kernel(str):
#     # use str to get url
#
#     # use wget and save to correct place
#
#     # load kernel
#     spice.load_kernel()
