import sys
import floris.tools as wfct
from floris.tools.agent_server_coord import Server, TurbineAgent
from floris.tools.optimization.scipy.yaw import YawOptimization
import floris.tools.floris_agent as fa
import floris.tools.train_run as tr
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import pickle
#import floris.tools.cc_blade_utilities as ccb
#from ccblade import CCAirfoil, CCBlade
import math
import random
from datetime import datetime
import itertools
from mpl_toolkits.mplot3d import Axes3D

start = datetime.now()

# *************************************** SIMULATION PARAMETERS ***************************************
# define yaw and wind direction limits

# same as defaults in trainer.py
yaw_min = -45
yaw_max = 45
yaw_step = 0.3

wind_speed_min = 0
wind_speed_max = 20

# relative to 270 deg
wind_dir_min = -180#-90
wind_dir_max = 180#90

N = 100

#yaw = np.linspace(yaw_min, yaw_max, N)
yaw = np.arange(yaw_min, yaw_max, yaw_step)#np.linspace(yaw_min, yaw_max, round((yaw_max-yaw_min+1)/yaw_step))
wind_speed = np.linspace(wind_speed_min, wind_speed_max, N)
wind_dir = np.linspace(wind_dir_min, wind_dir_max, N)

# create discrete state space
discrete_states = [wind_speed, yaw]

# specify dimensions that constitute a turbine's neighborhood
downwind = 2000
crosswind = 100
neighborhood_dims = [downwind, crosswind]

use_ccblade=False
fi = wfct.floris_interface.FlorisInterface("./example_input.json")
D = fi.floris.farm.turbines[0].rotor_diameter

# x and y coordinates of turbines in meters
layout_x = [0, 7*D, 14*D]
layout_y = [0, 0, 0]

# *************************************** SIMULATION INITIALIZATION ***************************************

#fi.reinitialize_flow_field(layout_array=[layout_x, layout_y])
#fi.calculate_wake()

aliases = ["turbine_" + str(i) for i in range(len(fi.floris.farm.turbines))]

farm_turbines = {alias: (x,y) for alias,x,y in zip(aliases,layout_x,layout_y)}

observation = fa.observe_turbine_state_wind_speed#yaw
#observation = fa.observe_turbine_state_sp_dir_yaw
modify_behavior = fa.modify_behavior_yaw
#modify_behavior = fa.modify_behavior_sp_dir_yaw

# this loop is to initialize a yaw angle state associate with each turbine. 
# this allows one turbine to have the yaw angles of multiple other turbines in its own state space
yaw_states = []
for index,turbine in enumerate(fi.floris.farm.turbines):
    model = fa.FlorisModel(fi, turbine, index)

    state = fa.State(name="yaw_angle", method=model.yaw_angle, state_type="discrete", discrete_values=yaw, action_type="step")

    yaw_states.append(state)

turbine_agents = []
for index,turbine in enumerate(fi.floris.farm.turbines):
        model = fa.FlorisModel(fi, turbine, index)

        wind_speed_state = fa.State(name="wind_speed", method=model.wind_speed, state_type="discrete", discrete_values=wind_speed)
        wind_direction_state = fa.State(name="wind_direction", method=model.wind_direction, state_type="discrete", discrete_values=wind_dir)
        # set observed back to True to return to normal behavior
        yaw_angle_state = fa.State(name="yaw_angle", method=model.yaw_angle, state_type="discrete", discrete_values=yaw, controlled=True, observed=True, action_type="step")

        # yaw angle state should only be controllable if it is for the current turbine, the other yaw angle states cannot be controled (decentralized)
        # NOTE: currently unused
        turb_states = yaw_states
        for i,turb_state in enumerate(turb_states):
            turb_state.controlled = (i == index)

        sim_context = fa.SimContext([wind_direction_state, yaw_angle_state])

        agent = TurbineAgent(aliases[index], discrete_states, 
                            farm_turbines=farm_turbines, 
                            observe_turbine_state=sim_context.observe_state, 
                            modify_behavior=sim_context.modify_behavior,
                            num_actions=fa.num_actions_yaw, 
                            value_function=fa.value_function_power_opt, 
                            find_neighbors=fa.find_neighbors, 
                            neighborhood_dims=neighborhood_dims,
                            model=model,
                            yaw_prop=0.2,
                            yaw_offset=5,
                            error_type=0,
                            sim_context=sim_context)
        agent.verbose = True
        turbine_agents.append(agent)
server = Server(turbine_agents)

min_yaw = 0
max_yaw = 45.0

# Instantiate the Optimization object
yaw_opt = YawOptimization(fi, minimum_yaw_angle=min_yaw, maximum_yaw_angle=max_yaw)

# Perform optimization
best_yaw_angles = yaw_opt.optimize()

fi.calculate_wake(yaw_angles=best_yaw_angles)
power_old_opt = sum([turbine.power for turbine in fi.floris.farm.turbines])
fi.calculate_wake(yaw_angles=[0,0,0])

# *********************************************************************
# DIRECTIONS FOR USE
# wind_speed_profile_s and wind_speed_profile_d are dicts with key:value pairs
# time:wind_speed. These dicts map a simulation time to the wind speed that the simulation
# should be set to at that time. Both dicts must begin with a 0 entry which is the intial
# wind speed. The largest time in the dict keys will be interpreted as the stop time, so the
# wind speed associated with the maximum simulation time can be set to np.nan. It should be 
# noted that if this maximum time is not added, it could result in a wind speed not being
# simulated because the simulation could end when a new wind speed is supposed to start.
# *********************************************************************

# *************************************** STEADY-STATE TRAINING ***************************************

action_selection = "boltzmann"
reward_signal = "constant"
coord = None
file_prefix = "test"
opt_window = 500

wind_dirs = [270]
wind_speeds = [8]
num_iterations = 0#1000
wind_dir_profile_s = tr.create_constant_wind_profile(wind_dirs, num_iterations*len(wind_speeds))
wind_speed_profile_s = tr.create_constant_wind_profile(wind_speeds, num_iterations)

[powers_s, turbine_yaw_angles_s, turbine_error_yaw_angles_s, turbine_values_s, reward_s, prob_lists_s] = \
    tr.train_farm(fi, turbine_agents, server, wind_speed_profile=wind_speed_profile_s, \
        wind_direction_profile=wind_dir_profile_s, action_selection=action_selection, \
        reward_signal=reward_signal, coord=coord, opt_window=opt_window)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

iterations = range(num_iterations)
yaw_angles = turbine_yaw_angles_s
powers = powers_s

file_path = ".\\paper_data\\"
yaw_name = file_path + file_prefix + "_yaw.npy"
power_name = file_path + file_prefix + "_power.npy"
iteration_name = file_path + file_prefix + "_iterations.npy"


best_angles= []
for agent in turbine_agents:
    best_angles.append(agent.utilize_q_table(state_name="yaw_angle"))

print("Best angles: ")
print(best_angles)
#fi.calculate_wake(yaw_angles=[25,25,0])

# *************************************** QUASI-DYNAMIC TESTING ***************************************
yaw_offset = 5
#fi.calculate_wake(yaw_angles=[agent.model.turbine.yaw_angle + yaw_offset for agent in turbine_agents])

print("New yaw angles:")
print([turbine.yaw_angle for turbine in fi.floris.farm.turbines])

action_selection = "boltzmann"
reward_signal = "constant"

# calamity that adds a yaw offset error
def add_error(fi, turbine_agents, server):
    for agent in turbine_agents:
        # reset learning rate to be 0.9
        agent.configure_dynamic(error_type=2,yaw_offset=yaw_offset)
        #agent.sim_context.change_error("yaw_angle", "offset", 10)

# calamity that shuts down turbine_1
def turbine_failure(fi, turbine_agents, server):
    """
    Simple method to test algorithm response to loss of a turbine.
    """
    
    turbine_agents[1].turn_off(server)
    for agent in turbine_agents:
        #agent.n = np.zeros_like(agent.n)
        #agent.Q = np.zeros_like(agent.Q)
        agent.find_neighbors(agent)
        agent.configure_dynamic()#error_type=2, yaw_offset=offset)
    #turbine_agents[1].Q = np.zeros_like(turbine_agents[1].Q)

fail_time = 1000
calamities = {}#{fail_time:turbine_failure}#{0:add_error}#, fail_time:turbine_failure}

for agent in turbine_agents:
        #agent._modify_behavior = fa.modify_behavior_delay#sp_dir_yaw_delay
        agent.num_actions=fa.num_actions_delay
        #agent.n = np.zeros_like(agent.n) # resets n values, typically used for instances with lots of error
        #agent.error_type = 2 # adds yaw error
        #agent.configure_dynamic()

# for agent in turbine_agents:
#     plt.matshow(agent.Q[agent.state_indices[0]])
#     title = "Initial Dynamic Q Table for " + agent.alias + "\nState is: " + str(agent.state)
#     plt.title(title)

wind_speeds_d = [8]
num_iterations_d = 10
wind_speed_profile_d = {}
for i, wind_speed in enumerate(wind_speeds_d):
    wind_speed_profile_d[i*num_iterations_d] = wind_speed
wind_speed_profile_d[len(wind_speeds_d)*num_iterations_d] = np.nan
wind_speed_profile_d = {len(wind_speeds_d)*num_iterations_d: np.nan}

num_iterations_d = 1000

wind_dirs_d = [270, 265, 265, 265, 265, 265, 265]
wind_speeds_d = [8]*len(wind_dirs_d)


# uncomment for stepwise constant profiles
wind_speed_profile_d = tr.create_constant_wind_profile(wind_speeds_d, num_iterations_d)
wind_dir_profile_d = tr.create_constant_wind_profile(wind_dirs_d, num_iterations_d)

action_selection = "boltzmann"

[powers_d, turbine_yaw_angles_d, turbine_error_yaw_angles_d, turbine_values_d, rewards_d] = \
    tr.run_farm(fi, turbine_agents, server, wind_speed_profile_d, wind_direction_profile=wind_dir_profile_d, calamities=calamities, \
        action_selection=action_selection, reward_signal=reward_signal)

#np.save('rl_training_d.npy', powers_d)

# *************************************** RESULTS ***************************************
turbine_yaw_angles = np.concatenate((turbine_yaw_angles_s, turbine_yaw_angles_d), axis=1)
turbine_error_yaw_angles = np.concatenate((turbine_error_yaw_angles_s, turbine_error_yaw_angles_d), axis=1)
turbine_values = np.concatenate((turbine_values_s, turbine_values_d), axis=1)

#ax = plt.gca()
plt.rcParams.update({'font.size':'25'})
#ax.get_yaxis().get_major_formatter().set_scientific(False)

fig, ax1 = plt.subplots(1,1,sharex=True, sharey=True)

#""" for i,yaw_angles in enumerate(turbine_error_yaw_angles):
#    label = "Turbine " + str(i) + " Error Reading"
#     ax1.plot(list(range(num_iterations_s*len(wind_speeds) + num_iterations_d)), yaw_angles, label=label) """

#total_time = list(range(num_iterations_s*len(wind_speeds_s) + num_iterations_d*len(wind_speeds_d)))

total_time_steady = list(range(max(wind_speed_profile_s.keys()) ))
total_time = list(range(max(wind_speed_profile_s.keys()) + max(wind_speed_profile_d.keys()) ))
total_time_dynamic= list(range(max(wind_speed_profile_d.keys()) ))

for i,yaw_angles in enumerate(turbine_yaw_angles):
    label = "Turbine " + str(i)
    ax1.plot(total_time, yaw_angles, label=label)
    ax1.axhline(best_angles[i], color='k')
ax1.legend() 
plt.figure()
for i,yaw_angles in enumerate(turbine_error_yaw_angles):
    label = "Turbine " + str(i)
    plt.plot(total_time, yaw_angles, label=label)
    plt.axhline(best_angles[i], color='k')
plt.title("Error Yaw Angles")

for time in wind_speed_profile_s.keys():
    ax1.axvline(x=time, color='red', linestyle='--')

offset = max(wind_speed_profile_s.keys()) - 1
for time in wind_speed_profile_d.keys():
    ax1.axvline(x=time + offset, color='magenta', linestyle='--')

#plt.title("Turbine Yaw Angles")

ax1.set_ylabel("Yaw Angle (degrees)")

#ax1.legend(prop={'size': 10})

# fill a list of the simulation wind speeds for plotting purposes
farm_wind_speeds_s = np.zeros(max(wind_speed_profile_s.keys()))
farm_wind_speeds_d = np.zeros(max(wind_speed_profile_d.keys()))

# set the element of the farm wind speed list that is the closest behind each simulation time
for i in range(len(farm_wind_speeds_s)):
    # NOTE: CHANGE THESE BACK TO WIND_SPEED_PROFILE_S
    diffs = {(i - time):time for time in wind_speed_profile_s.keys()}
    
    farm_wind_speeds_s[i] = wind_speed_profile_s[diffs[min([diff for diff in diffs.keys() if diff >= 0])]]

for i in range(len(farm_wind_speeds_d)):
    diffs = {(i - time):time for time in wind_speed_profile_d.keys()}
    if len(wind_speed_profile_d) > 1:
        farm_wind_speeds_d[i] = wind_speed_profile_d[diffs[min([diff for diff in diffs.keys() if diff >= 0])]]
    else:
        farm_wind_speeds_d[i] = wind_speeds[-1]#farm_wind_speeds_s[-1]

farm_wind_speeds = np.concatenate([farm_wind_speeds_s, farm_wind_speeds_d])

# ax2.plot(total_time, farm_wind_speeds)
# ax2.set_xlabel("Simulation Iteration")
# #ax2.set_ylabel("Wind Speed (m/s)")
# ax2.set_ylabel("Wind Dir. degrees)")

fig, (ax1, ax2) = plt.subplots(1,2, sharey=True)#figure()

layout_x = [0, 7*D, 14*D]
layout_y = [0, 0, 0]

#fi.reinitialize_flow_field(layout_array=[layout_x, layout_y])
# for agent in turbine_agents:
#     agent.configure_dynamic(yaw_offset=10)
# min_yaw = -55.0
# max_yaw = 55.0
#yaw_angles = optimize_yaw(fi, yaw_min, yaw_max)
#print(yaw_angles)
#fi.calculate_wake(yaw_angles=yaw_angles)

power_opt = fi.get_farm_power()#sum([turbine.power for turbine in fi.floris.farm.turbines])
#power_two_turbines = fi.floris.farm.turbines[0].power + fi.floris.farm.turbines[2].power
print("Observed yaw angles for optimization: ")
print([turbine.yaw_angle for turbine in fi.floris.farm.turbines])
print("Actual yaw angles for optimization: ")
print([turbine._yaw_angle for turbine in fi.floris.farm.turbines])
#print(best_yaw_angles)


fi.calculate_wake(yaw_angles=best_yaw_angles)
power_opt_two_turbines = fi.get_farm_power()#sum([turbine.power for turbine in fi.floris.farm.turbines])


print(power_opt_two_turbines)
#print(power_two_turbines)






fi.calculate_wake(yaw_angles=[angle + yaw_offset for angle in best_yaw_angles])

#power_old_opt = sum([turbine.power for turbine in fi.floris.farm.turbines])

print("power_opt: " + str(power_opt))
print("power_old_opt: " + str(power_old_opt))

steady_time = list(range(max(wind_speed_profile_s.keys())))
dynamic_time = list(range(max(wind_speed_profile_d.keys())))

file_path = "./paper_data/"

ax1.plot(steady_time, powers_s)
ax1.set_xlabel("Simulation Iteration")
ax1.set_ylabel("Power (MW)")
ax1.set_title("Training")

ax2.plot(dynamic_time, powers_d, label='RL Yaw Schedule')
#ax2.axhline(y=power_opt/1e6, color='red', linestyle='--', label='FLORIS Optimum (Shut Down)')
#ax2.axhline(y=power_opt_two_turbines/1e6, color='black', linestyle='--', label='FLORIS Optimum (Not Shut Down)')
#ax2.axvline(x=fail_time, color='magenta', linestyle='--')
ax2.set_xlabel("Time(s)")
ax2.set_title("Implementation")
#ax2.legend()
#ax2.axhline(y=power_two_turbines/1e6, color='red', linestyle='--', label='Waked Two Turbines')
#ax2.axhline(y=power_opt_two_turbines/1e6, color='black', linestyle='--', label='Non-Waked Two Turbines')
fig.suptitle("Power vs. Time for a Three Turbine Wind Farm - Combined \n")


ax = fig.get_axes()

num_ticks = 6
tick_interval = round(num_iterations_d / (num_ticks-1))

fig = plt.figure()
for i,turbine_value in enumerate(turbine_values):
    label = "Turbine " + str(i)
    plt.plot(total_time, turbine_values[i], label=label)
plt.legend()

plt.title("Value vs. Time for a Three Turbine Wind Farm - Combined \n")
plt.ylabel("Value Function Output")
plt.xlabel("Simulation Iteration")

fig = plt.figure()
for i,reward in enumerate(reward_s):
    label = "Turbine " + str(i)
    plt.plot(total_time_steady, reward, label=label)
plt.legend

plt.title("Reward vs. Time for a Three Turbine Wind Farm - Combined \n")
plt.ylabel("Reward")
plt.xlabel("Simulation Iteration")


print("Elapsed time: ", datetime.now() - start)

    
plt.show()