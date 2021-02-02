# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation


import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from matplotlib.animation import FuncAnimation
import numpy as np
import floris.tools as wfct
#import ffmpeg

# **************************************** Parameters **************************************** #

# whether or not a visualization should be generated
visualize = True

# whether or not the visualization animation should be saved as an mp4
save = False

# if save is True, what the file should be called
file_name = "test_animation.mp4"

# x and y resolution
x_res = 200
y_res = 200

# total simulation time
total_time = 400

# **************************************** Initialization **************************************** #
# Initialize the FLORIS interface fi
# For basic usage, the florice interface provides a simplified interface to
# the underlying classes
fi = wfct.floris_interface.FlorisInterface("./example_input.json")

# Calculate initial wake
fi.calculate_wake()

# Get horizontal plane at default height (hub-height)
# NOTE: this is currently commented out, wind conditions 
# at sim_time 0 below will be assumed to be initial conditions
#hor_plane = fi.get_hor_plane()

# lists that will be needed for visualizations
yaw_angles = [0 for turbine in fi.floris.farm.turbines]
powers = []
true_powers = []
turbine_velocities = []
hor_planes = []
wind_vectors = []
iterations = []
turbine_wind_speeds = [ [] for turbine in fi.floris.farm.turbines]
turbine_powers = [ [] for turbine in fi.floris.farm.turbines]

# **************************************** Simulation **************************************** #

TI = 5

mean = 8
dev = (TI/100)*mean

wind_speed_profile_high_ti = {}

for i in range(total_time):
    wind_speed_profile_high_ti[i] = np.random.normal(loc=mean, scale=dev)#np.random.uniform(low=8, high=8.3)
#wind_speed_profile_high_ti[total_time] = np.nan

if visualize:
        [x1_bounds, x2_bounds] = fi.get_plane_of_points(x1_resolution=x_res, x2_resolution=y_res, return_bounds=True)
        fi.first_x = x1_bounds[0]
fi.floris.farm.flow_field.mean_wind_speed = 8
for sim_time in range(total_time):
    iterations.append(sim_time)
    if sim_time % 100 == 0:
        print("Iteration:", sim_time)
    # if sim_time in wind_speed_profile_high_ti:
    #     fi.reinitialize_flow_field(wind_speed=wind_speed_profile_high_ti[sim_time], sim_time=sim_time)
    # if sim_time == 1:
    #     fi.floris.farm.flow_field.mean_wind_speed = 10
    #     fi.reinitialize_flow_field(wind_speed=10, sim_time=sim_time)
    #     if visualize:
    #         fi.vis_flow_field.mean_wind_speed = 10
    # if sim_time == 15:
    #     fi.reinitialize_flow_field(wind_speed=15, sim_time=sim_time)
    #     fi.floris.farm.flow_field.mean_wind_speed = 15
    #     if visualize:
    #         fi.vis_flow_field.mean_wind_speed = 15
    # if sim_time == 20:
    #     fi.reinitialize_flow_field(wind_speed=8, sim_time=sim_time)
    # if sim_time == 11:
    #     fi.reinitialize_flow_field(wind_speed=7, sim_time=sim_time)
    if sim_time == 10:
        fi.reinitialize_flow_field(wind_speed=10, sim_time=sim_time)
    # if sim_time == 2:
    #     fi.reinitialize_flow_field(wind_speed=7, sim_time=sim_time)
    # if sim_time == 10:
    #     yaw_angles = [45, 45, 0]

    #     fi.reinitialize_flow_field(wind_speed=8, sim_time=sim_time)
    #     if visualize:
    #         fi.vis_flow_field.mean_wind_speed = 8

    # calculate dynamic wake computationally
    fi.calculate_wake(sim_time=sim_time, yaw_angles=yaw_angles)

    for i,turbine in enumerate(fi.floris.farm.turbines):
        turbine_wind_speeds[i].append(fi.floris.farm.wind_map.turbine_wind_speed[i])

    for i,turbine in enumerate(fi.floris.farm.turbines):
        turbine_powers[i].append(turbine.power/1e6)

    # NOTE: at this point, can also use measure other quantities like average velocity at a turbine, etc.
    powers.append(sum([turbine.power for turbine in fi.floris.farm.turbines])/1e6)

    # calculate horizontal cut plane if necessary
    if visualize:
        hor_plane = fi.get_hor_plane(x_resolution=x_res, y_resolution=y_res, sim_time=sim_time)
        hor_planes.append(hor_plane)
        wind_vector = hor_plane.df.u.to_numpy()
        wind_vectors.append(wind_vector)
    
    # calculate steady-state wake computationally
    fi.calculate_wake()

    # NOTE: at this point, can also use measure other quantities like average velocity at a turbine, etc.
    true_powers.append(sum([turbine.power for turbine in fi.floris.farm.turbines])/1e6)

# **************************************** Plots/Animations **************************************** #
# Plot and show
plt.figure()

for i,turbine_power in enumerate(turbine_powers):
    label = "Turbine " + str(i)
    plt.plot(list(range(total_time)), turbine_power, label=label)
plt.ylabel("Power (MW)")
plt.xlabel("Time (s)")
plt.legend()

plt.figure()

for i,turbine_wind_speed in enumerate(turbine_wind_speeds):
    label = "Turbine " + str(i)
    plt.plot(list(range(total_time)), turbine_wind_speed, label=label)
plt.ylabel("Wind Speed (m/s)")
plt.xlabel("Time (s)")
plt.legend()

plt.figure()

plt.plot(list(range(total_time)), powers, label="Dynamic")
plt.plot(list(range(total_time)), true_powers, label="Steady-State")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Power (MW)")

fig, (ax1, ax2) = plt.subplots(2,1)

ax1.set_xlim([0, total_time])
ax1.set_ylim([0, max(max(true_powers), max(powers))*1.1])

line1, = ax1.plot([], [], label="Dynamic")
line2, = ax1.plot([], [], label="Steady-State")

def animate(frame, ax1, ax2):
    global hor_planes

    line1.set_data(iterations[:frame], powers[:frame])
    line2.set_data(iterations[:frame], true_powers[:frame])
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Power (MW)")
    ax1.legend(loc="lower right")

    wfct.visualization.visualize_cut_plane(hor_planes[frame], ax=ax2)

# frame rate
x = 10

if visualize:
    animation = FuncAnimation(fig, animate, np.arange(total_time), fargs=[ax1,ax2])

    writer = manimation.FFMpegWriter(fps=x)

    if save:
        animation.save(file_name, writer=writer)

plt.show()
