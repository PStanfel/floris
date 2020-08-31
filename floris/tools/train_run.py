from floris.tools import iterate, agent_server_coord
import itertools
import numpy as np
import matplotlib.pyplot as plt
from floris.tools.optimization.scipy.yaw import YawOptimization

# File created by Paul Stanfel for CSM-Envision Energy Research in Wind Farm Control; alpha version not yet validated.

def create_constant_wind_profile(wind_values, num_iterations):
    """
    Creates stepwise constant wind profile with equal numbers of simulation iterations at each 
    wind speed.

    Args:
        wind_values: A list of wind speeds or wind directions that the farm should be trained at.
        num_iterations: The number of simulation iterations that each wind speed should be run at.
    """
    wind_profile = {} #time: wind_speed
    for i, wind_value in enumerate(wind_values):
        wind_profile[i*num_iterations] = wind_value
    wind_profile[len(wind_values)*num_iterations] = np.nan

    return wind_profile

def _reintialize_turbine_value(turbine_agents, server):
    """
    Iterates through list of TurbineAgents and initializes the value function of each. This 
    method is intended to be called at the beginning of a simulation and potentially whenever
    the wind conditions change.
    Args:
        turbine_agents: A list of TurbineAgents.
        server: A Server object that enables inter-turbine communication.
    """
    # Initialize the value function value that each turbine broadcasts to the server.
    for agent in turbine_agents:
        agent.push_data_to_server(server)
        agent.find_neighbors(agent)

    for agent in turbine_agents:
        # Get initial value function of turbine and its neighbors
        agent.calculate_total_value_function(server, time=0)
        agent.calculate_total_value_function(server, time=1)

def train_farm(fi, turbine_agents, server, wind_speed_profile, sim_factor=1, \
    wind_direction_profile=None, action_selection="boltzmann", reward_signal="constant", \
    coord=None, opt_window=100, print_iter=True, num_episodes=1, num_iterations=None):
    """
    This method is intended to train a set of agents using a given wind profile and user 
    determined parameters.

    Args:
        fi: A FlorisUtilities object.
        turbine_agents: A list of TurbineAgents.
        server: A Server object that enables inter-turbine communication.
        wind_speed_profile: A dictionary mapping a simulation time to a change in wind speed.
        sim_factor: Float intended to account for different units (for example, wind speeds
        other than m/s), but currently unused.
        wind_direction_profile: A dictionary mapping a simulation time to a change in wind
        direction. Currently not implemented.
        action_selection: A string specifiying which action selection method should be used. 
        Current options are:
            - boltzmann: Boltzmann action selection
            - epsilon: Epsilon-greedy action selection
            - gradient: First-order backward-differencing gradient approximation.
        reward_signal: A string specifying what kind of reward signal can be used. For a 
        variable reward signal, reward will be capped to avoid overflow errors. Current options
        are:
            - constant: -1 if value decreased significantly, +1 if value increased significantly, 
            0 if value does not change significantly. Essentially implements reward clipping.
            - variable: Reward returned from the environment is scaled and used to directly 
            update the Bellman equation.
        coord: String specifying how coordination should be accomplished. If None, no
        coordination will be used, and execution will be simultaneous. Current options are:
            - up_first: Optimize from upstream to downstream
            - down_first: Optimize from downstream to upstream
        opt_window: Number of simulation iterations that each turbine or group of turbines is
        given to optimize, if coord is not None.
        print_iter: Boolean, whether or not to print the simulation iteration every 1000 iterations
    """
    # # Initialize the value function value that each turbine broadcasts to the server.
    # for agent in turbine_agents:
    #     agent.push_value_to_server(server)
    #     agent.find_neighbors(agent)

    # for agent in turbine_agents:
    #     # Get initial value function of turbine and its neighbors
    #     agent.calculate_total_value_function(server, time=0)

    # calculate initial wakes of the wind farm
    fi.calculate_wake()

    _reintialize_turbine_value(turbine_agents, server)

    turbine_yaw_angles = [[] for turbine in fi.floris.farm.turbines]
    turbine_error_yaw_angles = [[] for turbine in fi.floris.farm.turbines]
    turbine_values = [[] for turbine in fi.floris.farm.turbines]
    rewards = [[] for turbine in fi.floris.farm.turbines]

    powers = []
    prob_lists = []
    if print_iter:
        print("Beginning iteration of steady-state simulation...")

    for e in range(num_episodes):

        if num_iterations is None:
            iter_list = range(max( max(wind_speed_profile.keys())+1, max(wind_direction_profile.keys())+1))
        else:
            iter_list = range(num_iterations)

        # count up indefinitely, since simulation end time is not known 
        for i in iter_list:#itertools.count():
            # if i % 1000 == 0:
            #     fi.calculate_wake(yaw_angles=[0,0,0])
            # NOTE: this was moved to the end of the episode iteration
            if i == max( max(wind_speed_profile.keys()), max(wind_direction_profile.keys()) ):
                break
            # # end if stop conditions are met
            # if i == max(wind_speed_profile.keys()) and len(powers) > 0:
            #     return [powers, turbine_yaw_angles, turbine_error_yaw_angles, turbine_values, rewards, prob_lists]
            # elif i == max(wind_speed_profile.keys()):
            #     return [[], turbine_yaw_angles, turbine_error_yaw_angles, turbine_values, rewards, prob_lists]

            # reinitialize farm wind direction to next wind direction in the profile 
            if wind_direction_profile is not None and i in wind_direction_profile: 
                #diff = (wind_direction_profile[i] - 270) - fi.floris.farm.flow_field.wind_direction
                fi.reinitialize_flow_field(wind_direction=wind_direction_profile[i])
                #print("Wind direction reset to ", wind_direction_profile[i])
                # TODO: figure out if this is best place to put this
                server.reset_coordination_windows()

                # NOTE: I am currently trying to implement this in floris_agent, so that I can use absolute yaw angles
                #server.change_wind_direction(diff)
                # print("Wind direction changed by", diff, "degrees")
                # yaw_angles = [turbine.yaw_angle - diff for turbine in fi.floris.farm.turbines]
                # fi.calculate_wake(yaw_angles=yaw_angles)

            # reinitialize farm wind speed to next wind speed in the profile 
            if wind_speed_profile is not None and i in wind_speed_profile: 
                fi.reinitialize_flow_field(wind_speed=wind_speed_profile[i])

                # TODO: figure out if this is best place to put this
                server.reset_coordination_windows()

                # TODO: determine if _reinitialize_turbine_value should be called here or not
                #_reintialize_turbine_value(turbine_agents, server)

            if i%1000== 0 and print_iter:
                print("Iteration:", str(i))

            output = iterate.iterate_floris_steady(fi, turbine_agents, server, action_selection, reward_signal,\
                coord=coord, opt_window=opt_window)

            for j,value in enumerate([agent.total_value_present for agent in turbine_agents]):
                turbine_values[j].append(value)

            for j,yaw_angle in enumerate(output[0]):
                turbine_yaw_angles[j].append(yaw_angle)
                turbine_error_yaw_angles[j].append(yaw_angle)

            for j,reward in enumerate(output[1]):
                rewards[j].append(reward)

            power = np.sum([turbine.power for turbine in fi.floris.farm.turbines])
            powers.append(power / 1e6)
            prob_lists.append(output[2])

        
        print("Optimizing yaw for episode ", str(e))
        #fi.reinitialize_flow_field(layout_array=[layout_x, layout_y])
        min_yaw = -45.0
        max_yaw = 45.0
        # Instantiate the Optimization object
        yaw_opt = YawOptimization(fi, minimum_yaw_angle=min_yaw, maximum_yaw_angle=max_yaw)

        # Perform optimization
        best_yaw_angles = yaw_opt.optimize()
        # print(best_yaw_angles)
        # print([turbine_yaw_angles[i][-1] for i in range(len(turbine_agents))])

        punishment_factor = 0
        if [item for sublist in turbine_yaw_angles for item in sublist]:
            # skip this if turbine_yaw_angles has not been filled yet
            for i,agent in enumerate(turbine_agents):
                #error = abs(best_yaw_angles[i] - agent.sim_context.return_state("yaw_angle").get_state())
                error = abs(best_yaw_angles[i] - turbine_yaw_angles[i][-1])
                agent.update_Q(threshold=None, set_reward=error*punishment_factor)
                # print("Punishment value for", agent.alias, ":", error*punishment_factor)
    return [powers, turbine_yaw_angles, turbine_error_yaw_angles, turbine_values, rewards, prob_lists]

def run_farm(fi, turbine_agents, server, wind_speed_profile, \
    wind_direction_profile=None, action_selection="boltzmann", reward_signal="constant", calamities=None, print_iter=True):
    """
    This method is intended to implement a set of trained agents in a quasi-dynamic environment
    using a given wind profile and user determined parameters.

    Args:
        fi: A FlorisUtilities object.
        turbine_agents: A list of trained TurbineAgents.
        server: A Server object that enables inter-turbine communication.
        wind_speed_profile: A dictionary mapping a simulation time to a change in wind speed.
        wind_direction_profile: A dictionary mapping a simulation time to a change in wind
        direction. Currently not implemented.
        action_selection: A string specifiying which action selection method should be used. 
        Current options are:
            - boltzmann: Boltzmann action selection
            - epsilon: Epsilon-greedy action selection
        reward_signal: A string specifying what kind of reward signal can be used. For a 
        variable reward signal, reward will be capped to avoid overflow errors. Current options
        are:
            - constant: -1 if value decreased significantly, +1 if value increased significantly, 
            0 if value does not change significantly. Essentially implements reward clipping.
            - variable: Reward returned from the environment is scaled and used to directly 
            update the Bellman equation.
        calamities: A dictionary that maps a simulation time to a method that takes a list of
        TurbineAgents as an argument. This method will be called at a given simulation time
        to simulate events such as a loss of a turbine.
    """
    
    # calculate initial wakes of the wind farm
    fi.calculate_wake()

    _reintialize_turbine_value(turbine_agents, server)

    for i,turbine in enumerate(fi.floris.farm.turbines):
        turbine.number = i

    turbine_yaw_angles = [[] for turbine in fi.floris.farm.turbines]
    turbine_error_yaw_angles = [[] for turbine in fi.floris.farm.turbines]
    turbine_values = [[] for turbine in fi.floris.farm.turbines]
    rewards = [[] for turbine in fi.floris.farm.turbines]
    
    fi.wind_speed_change = (False, np.nan)

    verbose = False
    for agent in turbine_agents:
        agent.verbose = verbose

    # currently unused, intended to allow wind farm wakes to completely propagate before the simulation ends
    num_iterations_stabilize = 0#round(fi.floris.farm.flow_field.find_largest_distance() / fi.floris.farm.flow_field.wind_speed)

    powers = []
    if print_iter: print("Beginning quasi-dynamic test...")

    # count up indefinitely, since simulation end time is not known 
    for i in itertools.count():

        # activate calamity if the correct simulation time is met
        if calamities is not None and i in calamities:
            calamities[i](fi, turbine_agents, server)

        # end if stop conditions are met
        if i == max(wind_speed_profile.keys()) and len(powers) > 0:
            return [powers, turbine_yaw_angles, turbine_error_yaw_angles, turbine_values, rewards]
        elif i == max(wind_speed_profile.keys()):
            return [[], turbine_yaw_angles, turbine_error_yaw_angles, turbine_values, rewards]

        if wind_speed_profile is not None and i in wind_speed_profile:
            fi.reinitialize_flow_field(wind_speed=wind_speed_profile[i], sim_time=i)
            #print("Wind speed inside run_farm() is registered as", wind_speed_profile[i])

        if wind_direction_profile is not None and i in wind_direction_profile:
            fi.reinitialize_flow_field(wind_direction=wind_direction_profile[i], sim_time=i)

        if verbose:
            print("Iteration:", str(i))
        else:
            if i%1000== 0 and print_iter:
                print("Iteration:", str(i))

        train_action_selection = action_selection

        if i == num_iterations_stabilize and i != 0: train_action_selection = "hold"

        output = iterate.iterate_floris_delay(fi, turbine_agents, server, \
                                                sim_time=i, verbose=verbose, \
                                                action_selection=train_action_selection, \
                                                reward_signal=reward_signal)

        """ for i,value in enumerate([agent.total_value_present for agent in turbine_agents]):
            turbine_values[i].append(value) """
        for i,value in enumerate(output[3]):
            turbine_values[i].append(value)

        for i,yaw_angle in enumerate(output[1]):
            turbine_yaw_angles[i].append(yaw_angle)

        for i,yaw_angle in enumerate(output[2]):
            turbine_error_yaw_angles[i].append(yaw_angle)

        for i, reward in enumerate(output[4]):
            rewards[i].append(reward)

        """ for i,value in enumerate(output[3]):
            print("Iteration:", i)
            print(value)
            turbine_values[i].append(value) """

        power = np.sum([turbine.power for turbine in fi.floris.farm.turbines])
        powers.append(power / 1e6)
    print("Returning data...")
    return [powers, turbine_yaw_angles, turbine_error_yaw_angles, turbine_values, rewards]