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
import ffmpeg

angle_changes = {250:[10,0,0], 500:[10,10,0], 750:[20,10,0], 1000:[20,20,0]}

# Initialize the FLORIS interface fi
# For basic usage, the florice interface provides a simplified interface to
# the underlying classes
fi = wfct.floris_interface.FlorisInterface("./example_input.json")

# D = fi.floris.farm.turbines[0].rotor_diameter
# layout_x = [0, 7 * D, 14 * D]
# layout_y = [0, 0, 0]
# fi.reinitialize_flow_field(layout_array=(layout_x, layout_y))

# Calculate wake
fi.calculate_wake()

# Get horizontal plane at default height (hub-height)
#hor_plane = fi.get_hor_plane()

powers = []
true_powers = []
total_time = 75

# turb_0_yaw = 20

# fi.reinitialize_flow_field(wind_speed=8)
# fi.calculate_wake(yaw_angles=[turb_0_yaw,0])
# true_power = fi.get_farm_power()/1e6

# fi.calculate_wake()

yaw_angles = [0 for turbine in fi.floris.farm.turbines]

turbine_velocities = []

hor_planes = []
iterations = []

for sim_time in range(total_time):
    iterations.append(sim_time)
    print("Iteration:", sim_time)
    if sim_time == 1:
        fi.reinitialize_flow_field(wind_speed=10, sim_time=sim_time)
    if sim_time == 15:
        fi.reinitialize_flow_field(wind_speed=15, sim_time=sim_time)
    if sim_time == 20:
        fi.reinitialize_flow_field(wind_speed=8, sim_time=sim_time)
    # if sim_time == 11:
    #     fi.reinitialize_flow_field(wind_speed=7, sim_time=sim_time)

    hor_plane = fi.get_hor_plane(sim_time=sim_time)


    hor_planes.append(hor_plane)

    fi.calculate_wake(sim_time=sim_time)
    #powers.append(fi.get_farm_power()/1e6)
    powers.append(sum([turbine.power for turbine in fi.floris.farm.turbines])/1e6)
    
    velocity = fi.floris.farm.turbines[1].average_velocity
    turbine_velocities.append(velocity)
    fi.calculate_wake()
    #true_powers.append(fi.get_farm_power()/1e6)
    true_powers.append(sum([turbine.power for turbine in fi.floris.farm.turbines])/1e6)

# Plot and show
#fig, ax = plt.subplots()
#wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)

plt.figure()

plt.plot(list(range(total_time)), powers, label="Dynamic")
#plt.plot(list(range(total_time)), turbine_velocities)
plt.plot(list(range(total_time)), true_powers, label="Steady-State")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Power (MW)")

# for hor_plane in hor_planes:
#     # Plot and show
#     fig, ax = plt.subplots()
#     wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
#     #plt.

fig, (ax1, ax2) = plt.subplots(2,1)

ax1.set_xlim([0, 75])
ax1.set_ylim([0, 20])

line1, = ax1.plot([], [], label="Dynamic")
line2, = ax1.plot([], [], label="Steady-State")

def animate(frame, ax1, ax2):
    global hor_planes

    line1.set_data(iterations[:frame], powers[:frame])
    line2.set_data(iterations[:frame], true_powers[:frame])

    wfct.visualization.visualize_cut_plane(hor_planes[frame], ax=ax2)

x = 100

animation = FuncAnimation(fig, animate, np.arange(total_time), fargs=[ax1,ax2], interval=1000/x)

#writer = manimation.PillowWriter(fps=x)

#animation.save("test_animation.mp4", writer=writer)
plt.show()
