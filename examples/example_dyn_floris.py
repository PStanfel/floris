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


# Initialize the FLORIS interface fi
# For basic usage, the florice interface provides a simplified interface to
# the underlying classes
fi = wfct.floris_interface.FlorisInterface("./example_input.json")

# Calculate wake
fi.calculate_wake()

# Get horizontal plane at default height (hub-height)
hor_plane = fi.get_hor_plane()

powers = []
true_powers = []
total_time = 15

fi.reinitialize_flow_field(wind_speed=10)
fi.calculate_wake()
true_power = fi.get_farm_power()/1e6

fi.reinitialize_flow_field(wind_speed=8)
fi.calculate_wake()

for sim_time in range(total_time):
    print("Iteration:", sim_time)
    if sim_time == 0:
        fi.reinitialize_flow_field(wind_speed=10, sim_time=sim_time)

    fi.calculate_wake(sim_time=sim_time)

    powers.append(fi.get_farm_power()/1e6)
    true_powers.append(true_power)

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
