import sys
import floris.tools as wfct
from floris.tools.trainer import Trainer, LUT, TrainingMethod
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import pickle
import math
from datetime import datetime
import floris.tools.train_run as tr
from floris.simulation.simulator import Simulator

mean = 8
TI = 5
dev = (TI/100)*mean

total_time = 2000

wind_speeds = [7,8,9,10,11,12,13,14]
wind_dirs = [270]

wind_speed_profile = {}

for i in range(total_time):
    wind_speed_profile[i] = np.random.normal(loc=mean, scale=dev)#np.random.uniform(low=8, high=8.3)
wind_speed_profile[total_time] = np.nan
wind_dir_profile = tr.create_constant_wind_profile(wind_dirs, total_time)
#wind_speed_profile = tr.create_constant_wind_profile(wind_speeds, total_time)

wind_profiles = [wind_speed_profile, wind_dir_profile]

# comment out wind_speeds entry if performing the stochastic sweep tests
parameters = {"wind_profiles": wind_profiles, "wind_speeds": wind_speeds, "wind_directions": [270]}

# NOTE: this is hardcoded right now
num_turbines = 2

#Static method
static_tm = TrainingMethod(static=True, num_turbines=num_turbines, name="static")

# Boltzmann method
b_c_n_tm = TrainingMethod(static=False, num_turbines=num_turbines, coord=None, action_selection="boltzmann", \
    reward_signal="constant", iterations=total_time, name="b_c_n")

g_c_u_tm = TrainingMethod(static=False, num_turbines=num_turbines, coord="up_first", action_selection="gradient", \
    reward_signal="constant", iterations=total_time, opt_window = math.floor(total_time/num_turbines))

training_methods = [g_c_u_tm]#, b_c_n_tm]
lut_sets = []
wind_speed_lists = []

num_sims = 10
#training_methods = []
# for i in range(num_sims):
#     name = "stoch_test_var_" + str(i)

#     tm = TrainingMethod(static=False, num_turbines=3, coord=None, action_selection="boltzmann", \
#         reward_signal="variable", iterations=1000, name=name)

#     training_methods.append(tm)

for training_method in training_methods:
    fi = wfct.floris_interface.FlorisInterface("example_input.json")
    D = fi.floris.farm.turbines[0].rotor_diameter

    # x and y coordinates of turbines in meters
    #layout_x = [0, 7*D, 14*D]
    #layout_y = [0, 0, 0]

    #fi.reinitialize_flow_field(layout_array=[layout_x, layout_y])
    #fi.calculate_wake()

    trainer = Trainer(fi, parameters, training_method, dyn_train=False)

    #prefix = "./paper_data/" + training_method.name
    #print(prefix)
    tic = datetime.now()
    lut_sets.append( trainer.train() )
    toc = datetime.now()

    #time_file_path = prefix + "_time.npy"
    #np.save(time_file_path, np.array([toc.timestamp()-tic.timestamp()]))

    wind_speed_lists.append( trainer.wind_speeds() )

    plt.rcParams.update({'font.size':'25'})


for lut_set in lut_sets:
    for lut in lut_set:
        print(lut.read(all_states=True))

for lut_set in lut_sets:
    simulator = Simulator(fi, lut_set)

    (true_powers, powers, turbine_yaw_angles) = simulator.simulate( wind_profiles )

plt.figure()

plt.plot(powers)
plt.title("Power vs. Time")
plt.xlabel("Time (s)")
plt.ylabel("Power (MW)")

#plt.plot(true_powers, 'r--')

plt.figure()

for i,yaw_angles in enumerate(turbine_yaw_angles):
    label = "Turbine" + str(i)
    plt.plot(yaw_angles, label=label)

plt.title("Yaw Angles vs. Time")
plt.xlabel("Time (s)")
plt.ylabel("Yaw Angle (degrees)")
plt.legend()
plt.show()