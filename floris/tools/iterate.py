from floris.tools import agent_server_coord
import math
import random
import numpy as np
import itertools

# File created by Paul Stanfel for CSM-Envision Energy Research in Wind Farm Control; alpha version not yet validated.

'''
This file contains functions that are intended to be iterated continuously, constituting the
heart of a simulation.
'''

def iterate_floris(fi, turbine_agents, server, action_selection="boltzmann", reward_signal="constant", sim_time=None):
    """
    Method that is intended to be called repeatedly to simulate each time step of a simulation. Note that this method is
    deprectated and should be replaced by iterate_floris_steady.

    Args:
        fi: A FlorisUtilities object.
        turbine_agents: A list of TurbineAgents.
        server: A Server object that enables inter-turbine communication.
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
        sim_time: Current simulation time. If this value is an integer, the wake-delay version of FLORIS will be used. If not,
        FLORIS will behave as a normal steady-state simulator.
    """

    #print("NOTE: iterate_floris is deprecated. It is recommended to use iterate_floris_steady instead.")
    
    yaw_angles = []

    for agent in turbine_agents:
        # Determine current turbine state. NOTE: this overwrites the previous state and state_indices values.
        agent.observe_state()

        # Determine which yaw angle to set the turbine to.
        # Output of agent.take_action() depends on how modify_behavior is defined
        yaw_angle = agent.take_action(action_selection=action_selection)

        # Update, but don't run, the FLORIS model with the new yaw angle. All turbines must update yaw angle before FLORIS runs.
        yaw_angles.append(yaw_angle)

    # Run FLORIS with the new yaw angle settings
    yaw_angles = fi.calculate_wake(yaw_angles=yaw_angles, sim_time=sim_time)

    # update the server value function values
    for agent in turbine_agents:
        agent.push_data_to_server(server)

    for agent in turbine_agents:
        # Calculate "future" value function value
        agent.calculate_total_value_function(server, time=1)

        # turbines with more neighbors have to meet a higher threshold
        threshold = 5 * (len(agent.neighbors) + 1)

        # Update Q-table
        agent.update_Q(threshold, reward_signal=reward_signal)    

        agent.reset_value()
        
    return yaw_angles

def iterate_floris_steady(fi, turbine_agents, server, action_selection="boltzmann", 
                          reward_signal="constant", sim_time=None, coord=None, opt_window=100):  
    """
    Method that is intended to be called repeatedly to simulate each time step of a steady-state training simulation.

    Args:
        fi: A FlorisUtilities object.
        turbine_agents: A list of TurbineAgents.
        server: A Server object that enables inter-turbine communication.
        action_selection: A string or list of strings specifiying which 
            action selection method should be used. If a list, there must 
            be one entry per turbine. Current options for each turbine are:
                - boltzmann: Boltzmann action selection
                - epsilon: Epsilon-greedy action selection
                - gradient: First-order gradient approximation
                - hold: The turbine does not move
        reward_signal (string): Specifies what kind of reward signal can be 
            used. For a variable reward signal, reward will be capped to avoid 
            overflow errors. Current options are:
                - constant: -1 if value decreased significantly, +1 if value 
                    increased significantly, 0 if value does not change 
                    significantly. Essentially implements reward clipping.
                - variable: Reward returned from the environment is scaled and 
                    used to directly update the Bellman equation.
        sim_time (int): Current simulation time. If this value is an integer, 
            the wake-delay version of FLORIS will be used. If not, FLORIS will 
            behave as a normal steady-state simulator.
        coord (string): String specifying how coordination should be accomplished. 
            If None, no coordination will be used, and execution will be 
            simultaneous. Current options are:
                - up_first: Optimize from upstream to downstream
                - down_first: Optimize from downstream to upstream
                - None: No coordination.
        opt_window (int): Number of simulation iterations that each turbine or 
            group of turbines is given to optimize, if coord is not None.
    """
    yaw_angles = [None for agent in turbine_agents]

    rewards = []

    # determine which agents are ready to move based on the coordination scheme and the coordination_check method
    if coord:
        ready_agents = [agent for agent in turbine_agents if server.coordination_check(agent, coord)]
    else:
        ready_agents = turbine_agents

    if isinstance(action_selection, list):
        if len(action_selection) != len(turbine_agents):
            raise ValueError("action_selection must be same size as turbine_agents.")

    for i,agent in enumerate(turbine_agents):
        if agent in ready_agents:
            # Determine current turbine state. NOTE: this overwrites the previous state and state_indices values.
            agent.observe_state()

            # iterate through action selections if it is a list
            if isinstance(action_selection, list):
                turb_action_selection = action_selection[i]
            else:
                turb_action_selection = action_selection

            # Determine which yaw angle to set the turbine to.
            # Output of agent.take_action() depends on how modify_behavior is defined
            if agent.state_change:
                yaw_angle = 0
            else:
                yaw_angle = agent.take_action(action_selection=turb_action_selection, server=server)[0]

            # Update, but don't run, the FLORIS model with the new yaw angle. All turbines must update yaw angle 
            # before FLORIS runs.
            yaw_angles[i] = yaw_angle

    fi.steady_yaw_angles = yaw_angles

    # Run FLORIS with the new yaw angle settings
    yaw_angles = fi.calculate_wake(yaw_angles=yaw_angles)

    # update the server value function values
    for agent in turbine_agents:
        agent.push_data_to_server(server)

    for agent in turbine_agents:
        if agent in ready_agents:
            # The value function must be reset before update_Q so that gradient-based algorithms can have access
            # to the value function delta
            agent.reset_value()

            # Calculate "future" value function value
            agent.calculate_total_value_function(server, time=1)

            # turbines with more neighbors have to meet a higher threshold
            threshold = 5 * (len(agent.neighbors) + 1)

            if agent.state_change:
                set_reward = 0
            else:
                set_reward = None

            # Update Q-table
            reward = agent.update_Q(threshold, reward_signal=reward_signal, scaling_factor=1000, set_reward=set_reward)#250000))#1e6*0.005))    

            rewards.append(reward)

            if coord is not None: 
                agent.inc_opt_counter(opt_window=opt_window)
            # NOTE: called above, this is most likely not needed here

        else:
            rewards.append(0)
    
    prob_lists = []
    # NOTE: don't need these right now
    # for agent in turbine_agents:
    #     # these parameters must be changed based on how many states there are
    #     # [0,1] and [0:2] for three states
    #     # [0] and [0:1] for two states
    #     fixed_state_indices = [0]
    #     prob_lists.append(agent.prob_sweep(fixed_state_indices, agent.state[0:1]))
    #print(prob_lists)
    return [yaw_angles, rewards, prob_lists]

def iterate_floris_delay(fi, turbine_agents, server, sim_time=0, verbose=False, 
                         action_selection="boltzmann", reward_signal="constant", 
                         watch_state=["wind_speed", "wind_direction"], coord=None, opt_window=100, last_iter=None):
    """
    Method that is intended to be called repeatedly to simulate 
    each time step of a quasi-dynamic FLORIS simulation.

    Args:
        fi: A FlorisUtilities object.
        turbine_agents: A list of TurbineAgents.
        server: A Server object that enables inter-turbine 
            communication.
        sim_time (int): Integer specifying simulation time. 
            This is used in conjunction with modifications 
            to FLORIS to simulate a wake delay, so it 
            must be iterated upwards on each call of 
            iterate_floris_delay.
        verbose (bool): Prints out extra information when 
            True.
        action_selection (string): Specifies which action 
            selection method should be used. 
            Current options are:
                - boltzmann: Boltzmann action selection
                - epsilon: Epsilon-greedy action selection
        reward_signal: A string specifying what kind of 
            reward signal can be used. For a variable reward 
            signal, reward will be capped to avoid overflow 
            errors. Current options are:
                - constant: -1 if value decreased significantly, 
                    +1 if value increased significantly, 0 if 
                    value does not change significantly. Essentially 
                    implements reward clipping.
                - variable: Reward returned from the environment is 
                    scaled and used to directly update the Bellman 
                    equation.
        target_state (int): Specifies which index of the 
            discrete state space should be "looked at" to watch 
            for changes. It is currently used to activate a ramp 
            up or down to a given yaw angle if a wind speed 
            change is detected.
        coord (string): What kind of coordination to use. Current 
            options are:
                - up_first: begins coordination at most upstream 
                    turbine
                - down_first: begins coordination at most 
                    downstream turbine
                - None: no coordination
        opt_window (int): If coordinated, the amount of time each 
            turbine should be given to optimize.
    """
    locking_turbine = []
    yaw_angles = [None for turbine in fi.floris.farm.turbines]
    values = [None for turbine in fi.floris.farm.turbines]
    fi.calculate_wake()
    if coord:
        ready_agents = [agent for agent in turbine_agents if server.coordination_check(agent, coord)]
    else:
        ready_agents = turbine_agents

    state_changes = [False for turbine in fi.floris.farm.turbines]

    active_agents = [agent for agent in ready_agents if not agent.shut_down]
    indices = list(range(len(active_agents)))
    random.shuffle(indices)

    # BUG: this causes the quasi-dynamic phase to not work properly
    # for i in indices:
    #     active_agents[i].observe_state(target_state=target_state, peek=True)
        #, fi=fi)

    #     if active_agents[i].state_change or active_agents[i].target_state is not None:
    #         if active_agents[i].verbose: print(active_agents[i].alias, "ramping...")
    #         active_agents[i].observe_state(target_state=target_state, peek=False)
    #         yaw_angles[i] = active_agents[i].ramp(state_name="yaw_angle")

    #         # ramping still causes a wake delay, so turbines must be locked
    #         # if there is no ramping, delay_map should be empty
    #         #print(active_agents[i].alias, "calls first server.lock()")
    #         server.lock(active_agents[i]) 

    for i,agent in enumerate(turbine_agents):
        agent.observe_state(peek=True)
    
        yaw_set = agent.ramp(state_name="yaw_angle")
        if yaw_set is not None:
            yaw_angles[i] = yaw_set

    for i in indices:
        agent = active_agents[i]

        if server.check_neighbors(agent):

            agent.observe_state(watch_state=watch_state)

            if verbose: print(indices)

            agent.calculate_total_value_function(server, time=0)

            # uses data member set by TurbineAgent.observe_state() to adjust yaw angles for a wind speed change
            yaw_angle = agent.take_action(action_selection=action_selection)[0]

            if agent.verbose:
                print(agent.alias, "chooses yaw angle", yaw_angle)


            # Update, but don't run, the FLORIS model with the new yaw angle. All turbines must update yaw angle before FLORIS runs.
            yaw_angles[i] = yaw_angle

            # set flags properly
            agent.completed_propagation = False
            agent.completed_action = False
            agent.start_power_state_filters()

            # lock downstream turbines
            server.lock(agent)
            
            locking_turbine.append(i)

            break

    fi.steady_yaw_angles=yaw_angles

    # Run FLORIS with the new yaw angle settings
    yaw_angles = fi.calculate_wake(yaw_angles=yaw_angles, sim_time=sim_time)
    # NOTE: agents don't register this change in yaw angles until the next time agent.observe_state is called
    # I don't think this is a significant issue, however, at least at this point, but it does explain some inconsistencies
    # between the error yaw angle plots and the actual yaw angle plots

    # update the server value function values
    for i,agent in enumerate(turbine_agents):
        agent.push_data_to_server(server)

    for i,agent in enumerate(turbine_agents):
        #if agent.verbose: print(agent.alias, " delay is ", agent.delay)
        reached_zero = False
        if agent.wake_delay > 0:
            agent.wake_delay -= 1
            if agent.wake_delay == 0: reached_zero = True

        if agent.state_delay > 0:
            agent.state_delay -= 1
        if agent.power_delay > 0:
            agent.power_delay -= 1

        if agent.wake_delay == 0 and agent.completed_propagation == False:
            agent.completed_propagation = True
            if agent.verbose:
                print(agent.alias, "sets completed_propagation to True")
        elif agent.wake_delay == 0 and reached_zero and agent.completed_propagation is None:
            agent.completed_action = False
            agent.start_power_state_filters()
            if agent.verbose:
                print(agent.alias, "sets filters and sets completed_action to False")
        
        if agent.power_delay == 0 and agent.state_delay == 0 and agent.completed_action == False:
            agent.completed_action = True
            if agent.verbose:
                print(agent.alias, "sets completed_action to True")

        values[i] = 0

    if not sim_time == 0:
        random.shuffle(indices)
        for i in indices:
            agent = turbine_agents[i]
            if agent.completed_propagation and server.check_neighbors(agent):
                # The value function must be reset before update_Q so that gradient-based algorithms can have access
                # to the value function delta
                #agent.reset_value()

                # Calculate "future" value function value
                agent.calculate_total_value_function(server, time=1)

                # turbines with more neighbors have to meet a higher threshold
                threshold = 5 * (len(agent.neighbors) + 1)

                # Update Q-table
                agent.update_Q(threshold, reward_signal=reward_signal, scaling_factor=1e3)   

                agent.completed_action = None
                agent.completed_propagation = None

                agent.reset_value()

                if coord is not None: 
                    agent.inc_opt_counter(opt_window=opt_window)

                if verbose: print(agent.alias, "completes Q update")


        rewards = [agent.total_value_future - agent.total_value_present for agent in turbine_agents]
    
    rewards = [agent.reward for agent in turbine_agents]

    return [locking_turbine, yaw_angles, [agent.state[1] for agent in turbine_agents], values, rewards]
