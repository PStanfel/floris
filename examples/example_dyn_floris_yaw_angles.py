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

import floris.tools as wfct

angle_changes = {250:[10,0,0], 500:[10,10,0], 750:[20,10,0], 1000:[20,20,0]}

# Initialize the FLORIS interface fi
# For basic usage, the florice interface provides a simplified interface to
# the underlying classes
fi = wfct.floris_interface.FlorisInterface("./example_input.json")

# D = fi.floris.farm.turbines[0].rotor_diameter
# print(D)
# layout_x = [0, 7 * D]
# layout_y = [0, 0]
# fi.reinitialize_flow_field(layout_array=(layout_x, layout_y))

# Calculate wake
fi.calculate_wake()

# Get horizontal plane at default height (hub-height)
hor_plane = fi.get_hor_plane()

powers = []
true_powers = []
total_time = 1200

# turb_0_yaw = 20

# fi.reinitialize_flow_field(wind_speed=8)
# fi.calculate_wake(yaw_angles=[turb_0_yaw,0])
# true_power = fi.get_farm_power()/1e6

# fi.calculate_wake()

yaw_angles = [0 for turbine in fi.floris.farm.turbines]

for sim_time in range(total_time):

    if sim_time in angle_changes:
        yaw_angles = angle_changes[sim_time]
        #fi.reinitialize_flow_field(wind_speed=10, sim_time=sim_time)

    fi.calculate_wake(sim_time=sim_time, yaw_angles=yaw_angles)

    powers.append(fi.get_farm_power()/1e6)
    
    
    fi.calculate_wake(yaw_angles=yaw_angles)
    true_powers.append(fi.get_farm_power()/1e6)

# Plot and show
fig, ax = plt.subplots()
wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)

plt.figure()

plt.plot(list(range(total_time)), powers, label="Dynamic")
plt.plot(list(range(total_time)), true_powers, label="Steady-State")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Power (MW)")

plt.show()
