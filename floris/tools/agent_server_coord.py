from floris.tools import q_learn 
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import sys
from scipy.ndimage.filters import gaussian_filter

# File created by Paul Stanfel for CSM-Envision Energy Research in Wind Farm Control; alpha version not yet validated.

class TurbineAgent():
    """
    TurbineAgent is a class that facilitates agent-based modeling of a wind turbine using Q-learning.

    TurbineAgent uses an externally defined model to encapsulate information about a system and 
    defines methods that can be used to implement a reinforcement learning approach to 
    controlling a user-defined parameter of the system.

    Args:
        alias: The name that the turbine will be referred to as
        discrete_states: A list of lists containing the discrete states that the system can be in
        farm_turbines: A map of every turbine in the farm from alias to a (x,y) coordinate tuple
        observe_turbine_state: A function that returns the turbine state in the state space
            defined by discrete_states. The output of this function must have the same number
            of dimensions as discrete_states.
        modify_behavior: A function that returns a desired system parameter that can be used to 
            modify the external system. This function uses the action selected by an exploration/
            explotation algorithm, so the function must map an integer to a system parameter like
            yaw angles.
        num_actions: The number of possible actions that the agent can take and which 
            modify_behavior uses to map to change in system behavior.
        value_function: A function that returns system value. The algorithm attempts to maximize
            the output of this function in conjunction with all turbines in the neighborhood.
        find_neighbors: A function that populates self.neighbors with the aliases of turbines 
            that can be communicated with.
        neighborhood_dims: A list [downwind, crosswind] of dimensions that correspond to the 
            definition of a turbine's neighborhood with respect to the wind direction.
        model: An implementation-specific variable that encapsulates any information about the
            model of the environment that is needed to implement the user-defined method 
            observe_turbine_state, modify_behavior, value_function, and find_neighbors.
        power_reference: A power reference value that the individual turbine is trying to achieve. 
        yaw_prop: Float, proportional yaw error
        verbose: Boolean, when True prints out information about which methods are called and
        their arguments.
        yaw_offset: Float, flat turbine offset error
        error_type: Int, specifies what kind of error should be added to a measurement. This
        value can be used in the observe_turbine_state method. If an error is selected, the 
        appropriate value (yaw_prop, yaw_offset, etc.) can be used. Current options are:
            - 0: no error
            - 1: proportional error
            - 2: offset error
        sim_factor: 

    Returns:
        TurbineAgent: An instantiated TurbineAgent object.
    """
    def __init__(self, alias, 
                    discrete_states, 
                    farm_turbines, 
                    observe_turbine_state, 
                    modify_behavior, num_actions, 
                    value_function, 
                    find_neighbors, 
                    neighborhood_dims,
                    leader=False, 
                    model=None,
                    power_reference=None,
                    yaw_prop=0,
                    verbose=False,
                    yaw_offset=0,
                    error_type=0,
                    sim_factor=1,
                    tau=0.5,
                    epsilon=0.1,
                    discount=0.5,
                    sim_context=None,
                    value_baseline=0):
    
        self.sim_context = sim_context

        self.leader = leader # True if this turbine leads the consensus movement

        self.yaw_prop = yaw_prop
        self.yaw_offset = yaw_offset
        self.error_type = error_type #0: no error, 1: proportional error, 2: offset error

        # variable that tracks wind direction differences
        self.wind_dir_diff = 0

        self.verbose = verbose
        self.farm_turbines = farm_turbines # dict mapping all aliases in the farm to (x,y) coordinates
        self.power = 0 # output power of this turbine
        self.alias = alias
        self.position = self.farm_turbines[self.alias] # (x,y) coordinates
        
        self.model = model

        self._value_function = value_function # must have TurbineAgent as only argument and return int or float representing the turbine value function
        self.value_baseline = value_baseline

        self._observe_turbine_state = observe_turbine_state # must have TurbineAgent as only argument and return tuple representing state
        self.state = self._observe_turbine_state(self)

        self.discrete_states = discrete_states

        # if len(self.state) != len(self.discrete_states):
        #     raise ValueError("Tuple returned by observe_turbine_state does not match the dimensions of the discrete state space.")

        self.state_indices = self.sim_context.get_state_indices()#self._get_state_indices(self.state)

        self.neighbors = [] # list of agent neighbors
        self.reverse_neighbors = [] # list of turbines that have the agent as neighbors

        self.downwind = neighborhood_dims[0]
        self.crosswind = neighborhood_dims[1]

        self.find_neighbors = find_neighbors

        self.discrete_states = discrete_states

        # TODO: in the future, self.Q will be replaced by self.Q_obj, left in here to avoid errors elsewhere
        [self.n, self.Q, self.Q_obj] = self.sim_context.blank_tables()

        # initialize eligibility trace table
        self.E = np.zeros_like(self.Q)

        # TD(lambda) parameter (lambda is a reserved python keyword)
        self.lamb = 0

        # dim = [len(state_space) for state_space in discrete_states]
        # self.n = np.zeros(tuple(dim)) # table that keeps track of how many times a state has been visited

        # # NOTE: changed to new Q order
        # dim.insert(0, num_actions)
        # dim.append(num_actions)
    
        # self.Q = np.zeros(tuple(dim)) # internal Q-table
        
        # set simulation parameters, as they will cause errors if set to None
        if tau is not None:
            self.tau = tau
        else:
            self.tau = 0.5

        if epsilon is not None:
            self.epsilon = epsilon
        else:
            self.epsilon = 0.1

        if discount is not None:
            self.discount = discount
        else:
            self.discount = 0.5

        self.action = 0
        self.total_value_present = 0
        self.total_value_future = 0

        self.self_value_present = 0
        self.self_value_future = 0

        self.control_action_present = 0
        self.control_action_future = 0

        self.reward = 0 # keep track of reward signals

        self._modify_behavior = modify_behavior # must have TurbineAgent as only argument and return a parameter that will be updated in the overall system

        self.k = [1, 1, 1] # these values are used to determine the learning rate

        self.power_reference = power_reference

        self.completed_action = None

        self.delay = 0
        self.delay_map = {} # maps a turbine alias to how long it will be delayed for

        self.comm_failure = False
        self.state_change = False # changes to true if a significant state change occurs
        self.target_state = None
        self.sim_factor = sim_factor # scaling factor to change simulation resolution NOTE: unused
        #self.z = 1

        self.done = False # if True, the agent has finished optimizing in its coordination window
        self.opt_counter = 0 # this counts how many iterations the agent has completed in its optimization window

        self.shut_down = False

        self.locked_by = "" # string variable that keeps track of which turbine alias locked the agent

        
    def turn_off(self, server):
        # delay of 0 means the turbine is no longer locked
        self.delay = 0
        self.model.turbine.turbine_shut_down = True
        self.shut_down = True
        server.shut_down(self)

    def turn_on(self):
        self.model.turbine.turbine_shut_down = False
        self.shut_down = False

    def set_tau(self, tau):
        self.tau = tau

    def _get_state_indices(self, state):
        state_indices = q_learn.find_state_indices(self.discrete_states, state)
        return state_indices

    def push_data_to_server(self, server):
        """
        This method pushes a turbine's value function value to the server.

        Args:
            server: A Server object that all the turbines communicate on.
        """
        # NOTE: normalize power
        if self.shut_down:
            data = 0
        else:
            data = self.model.turbine.power
            
        self.self_value_present = self.self_value_future
        self.self_value_future = data

        server.update_channel(self.alias, data)#self._value_function(self, server))

    def pull_from_server(self, server, target_alias):
        """
        This method reads the value function value posted to the server by a given alias.

        Args:
            server: A Server object that all the turbines communicate on.

            target_alias: The turbine to pull information from.
        """
        self.neighbor_value_values[target_alias] =  server.read_channel(target_alias)

    def observe_state(self, target_state=0, peek=False):
        """
        Determines the tuple state_indices that correspond to the current system state (which can be used 
        to index Q) and the actual system state values. This method also updates the self.state_change
        variable, which is True if a change in the target_state occurred.

        Args:
            target_state: An int representing what state(s) to use to determine whether a significant state
                change has occurred. 

            peek: A Boolean that represents whether or not the internal state should be changed. When peek
                is True, this method will only check to see if the target state has changed. When it is 
                False, self.state and self.state_indices will be modified.
        """
        old_state = self.state

        # observe new state
        new_state = self._observe_turbine_state(self)

        # this conditional should set the internal state variables if the agent is ramping (ie self.target_state is not None)
        # or if peek is False and the agent is not ramping
        # setting peek to True will bypass this, unless the agent is ramping
        if not peek:#if self.target_state is not None or (not peek and self.target_state is None):
            # if peek is False, set new state and calculate corresponding indices in the state space        
            self.state = new_state
            self.state_indices = self.sim_context.get_state_indices()#self._get_state_indices(self.state)

            # Update n to reflect how many times this state has been visited
            self.n[self.state_indices] += 1

        # change state_change member variable if a state change occurred in the target state
        if old_state[target_state] != new_state[target_state]:
            self.state_change = True
            if self.verbose:
                print(self.alias, "state_change set to True")
                print(self.state)
                print(self.state_indices)
                #plt.matshow(self.Q[self.state_indices[0]])

    # NOTE: This method hard codes in state values: need to do this a different way
    def ramp(self, state_name):

        if self.state_change:
            self.control_to_value(self.utilize_q_table(state_name), state_name)
            self.state_change = False

        if self.target_state is not None:
            yaw_angles = []

            yaw_angle = self.model.turbine.yaw_angle

            yaw_rate = self.model.turbine.yaw_rate

            diff = self.target_state[0] - yaw_angle

            # NOTE: this code is redundant because of how modfy_behavior_delay is written, however it is left in this
            # form for flexiblity, should a different modify behavior function be used
            # if abs(diff) < yaw_rate:
            #     yaw_angle = self.target_state[0]
            #     self.target_state = None
            # else:
            yaw_angle = self._modify_behavior(self)[0]
            self.completed_action = False

            #print(self.alias, "yaw angle:", yaw_angle)

            return yaw_angle
        else:
            self.delay_map = {}
            return None

    def control_to_value(self, target, control_state):
        """
        Sets an internal target parameter based on a designated control_state

        Args:
            target: The value that a given control variable should reach.
            control_state: Which state is controllable.
        """
        #discrete_target_index = q_learn.find_state_indices([self.discrete_states[control_state]], [target])
        discrete_target = self.sim_context.return_state(control_state).get_state(target=target) #self.discrete_states[control_state][discrete_target_index]
        self.target_state = (discrete_target, control_state)
        if self.verbose: print(self.alias, "sets target as", discrete_target)

    # def _evaluate_value_function(self):
    #     return self._value_function(self, server)

    def calculate_total_value_function(self, server, time=None):
        """
        Determines the value function of the turbine and its neighbors.

        Args:
            server: A Server object that all the turbines communicate on.

            time: An integer variable indicatin whether this is the "current" time (0) or the "future" 
                time (1). The corresponding variable is updated accordingly. This is needed to make 
                sure that the Q-learning algorithm can be properly executed.

            total: Boolean, indicates whether or not the 
        """

        return self._value_function(self, server, time)

    def _select_action(self, action_selection="boltzmann"):
        """
        Chooses action based on Boltzmann search. This action must be mapped to a corresponding 
        change in system parameters by the function modify_behavior.

        NOTE: This method is not necessary and will soon be removed. Use take_action instead.

        Args:
            action_selection: The algorithm that will be used to select a control action.
        """
        print("_select_action is deprecated and will soon be removed. Use take_action")
        if action_selection == "boltzmann":
            self.action = q_learn.boltzmann(self.Q, self.state_indices, self.tau)
        elif action_selection == "epsilon":
            self.action = q_learn.epsilon_greedy(self.Q, self.state_indices, self.epsilon)

    def _calculate_deltas(self, server):
        """
        Calculates change in value function and change in control input between iterations for use in, for
        example, a gradient-based action selection algorithm.

        Args:
            server: Server object so that the agent can learn information about neighbors.

        Returns:
            deltas: An iterable that has the difference in value function as its first element and the difference
            in control input as its second element.
        """
        deltas = []
        delta_control = self.control_action_future - self.control_action_present
        for alias in self.neighbors:
            deltas.append((server.read_delta(alias), delta_control))

        deltas.append((server.read_delta(self.alias), delta_control))

        # # difference in the value function between the "future" and the "present"
        # delta_V = self.total_value_future - self.total_value_present

        # # difference in the control input between the "future" and the "present"
        # delta_control = self.control_action_future - self.control_action_present

        # deltas = [delta_V, delta_control]
        #print(self.alias, "delta_V:", delta_V)
        #print(self.alias, "delta_control:", delta_control)
        return deltas

    def take_action(self, action_selection="boltzmann", server=None, state_indices=None, return_action=False):
        """
        Chooses an action and returns a system parameter mapped via the function modify_behavior

        Args:
            action_selection: The algorithm that will be used to select a control action.
            state_indices: A tuple that, if specified, will be used as the state indices to find
            the Q entry at, not the current state_indices stored in self.state_indices.
            return_action: Boolean specifiying whether or not the action that was selected should 
            be returned.

        Returns:
            A system parameter that must be interpreted by the external code based on how modify_behavior is defined
        """
        # skip this method if the turbine is shut down
        if self.shut_down:
            return None

        if state_indices is not None:
            indices = state_indices
        else:
            indices = self.state_indices
        #self._select_action(action_selection=action_selection)
        if action_selection == "boltzmann":
            # this is the only method currently reconfigured for Q_obj
            action = q_learn.boltzmann(self.Q_obj, self.state, self.tau)
        elif action_selection == "epsilon":
            action = q_learn.epsilon_greedy(self.Q, indices, self.epsilon)
        elif action_selection == "gradient":
            action = q_learn.gradient(self._calculate_deltas(server))
            #print(self.alias, "takes action", self.action)
            #print(self.alias, self._calculate_deltas())
        elif action_selection == "hold":
            action = None

        """ if self.alias == "turbine_0" or self.alias == "turbine_2":
            self.action = 1 """
        
        if return_action:
            return action
        
        self.action = action
        return self._modify_behavior(self)

    def update_Q(self, threshold, reward_signal="constant", scaling_factor=100, set_reward=None):
        """
        This function assumes that a simulation has been run and total_value_future has been updated, and updates
        the internal Q-table based on which action is currently selected by the turbine agent.

        Args:
            threshold: If a constant reward signal, the threshold in the value function that is used to 
            determine if a significant change took place.

            reward_signal: What algorithm is used to allocate reward to the Q-Learning algorithm.

            scaling_factor: What value to scale the value function differential down by to prevent overflowing
            in the action selection routine.
        """
        # skip this method if the turbine is currently ramping or is shut down
        if self.target_state is not None or self.shut_down:
            return

        # Determine learning rate.
        l = self.k[0] / (self.k[1] + self.k[2]*self.n[self.state_indices])
        #l = 0.5
        
        #diff = (self.total_value_future - self.total_value_present) + (self.total_value_future - self.value_baseline) / self.value_baseline 
        #NOTE: testing out different diff calculation
        
        #print(diff)
        #print(diff)
        if reward_signal == "variable" and set_reward is None:
            # Calculate difference between "future" value and "present" value.
            diff = (self.total_value_future - self.total_value_present)

            # NOTE: only remove scaling_factor if power is scaled already in push_data_to_server
            reward = diff/scaling_factor

        elif reward_signal == "constant" and set_reward is None:
            # Calculate difference between "future" value and "present" value.
            diff = (self.total_value_future - self.total_value_present)

            # Assign reward based on change in system performance.
            if diff > threshold:
                reward = 1
                #if self.alias == "turbine_1": print("r+")
            elif abs(diff) < threshold:
                reward = 0
                #if self.alias == "turbine_1": print("r0")
            elif diff < -threshold:
                reward = -1
                #if self.alias == "turbine_1": print("r-")
            else:
                # intended to mitigate reward not being defined if diff is NaN
                reward = 0

        elif reward_signal == "absolute" and set_reward is None:
            diff = self.total_value_future - self.value_baseline/2

            reward = diff / self.value_baseline * 2

        else:
            reward = set_reward

        # set self.reward so reward signals can be visualized over the course of the simulation
        self.reward = reward

        # NOTE testing different reward assignment scheme
        new_gap = self.total_value_future - self.value_baseline
        old_gap = self.total_value_present - self.value_baseline

        # if old_gap < 0:
        #     if new_gap >= old_gap:
        #         reward = 0
        #     else:
        #         reward = -1
        # else:
        #     if new_gap >= old_gap:
        #         reward = 1
        #     else:
        #         reward = 0

        # The "current" Q value, obtained using the chosen action and the previous state_indices.
        #NOTE: changed to new Q order
        #Q_t = self.Q[self.action][self.state_indices]
        #Q_t = self.Q[self.state_indices][self.action]
        
        # accumulating traces
        #self.E[self.state_indices][self.action] = self.E[self.state_indices][self.action] + 1

        # Observe new state, using the internal function that doesn't overwrite any variables.
        future_state = self._observe_turbine_state(self)
        future_state_indices = self.sim_context.get_state_indices(targets=future_state)#self._get_state_indices(future_state)

        # Maximum "future" Q value.
        #max_Q_t_1 = q_learn.max_Q(self.Q, future_state_indices)

        future_action = self.take_action(state_indices=future_state_indices, return_action=True)
        #next_Q = self.Q[future_state_indices][future_action]

        # The "future" Q value.
        #Q_t_1 = Q_t + l*(reward + self.discount*max_Q_t_1 - Q_t)

        # Update the Q table
        #NOTE: changed to new Q order
        #self.Q[self.action][self.state_indices] = Q_t_1

        # print statement to see if table is being updated
        if set_reward is not None:
            print("Q table entry for state", self.state, "and action", self.action, "updated with reward", reward, "for agent", self.alias)

        #delta = reward + self.discount*max_Q_t_1 - Q_t

        #self.Q = self.Q + l*delta*self.E
        #self.E = self.discount*self.lamb*self.E
        #print(self.alias, "updating Q-table for state", self.state)
        self.Q_obj.update(self.state, self.action, reward, future_state, n=self.n)
        self.Q = self.Q_obj.return_q_table()

        # commented out temporarily to test eligibility traces
        #self.Q[self.state_indices][self.action] = Q_t_1
        #print(self.alias, "completes Q update with reward", reward)
        return reward

    def prob_sweep(self, fixed_state_indices, fixed_states):

        # must have one fixed state value for each fixed state index
        if len(fixed_state_indices) != len(fixed_states):
            raise ValueError("fixed_state_indices size must match fixed_states size.")

        # determine the state indices of the fixed states
        state_indices = [None for state in self.discrete_states]
        for index in fixed_state_indices:
            state_indices[index] = q_learn.find_state_indices([self.discrete_states[index]], [fixed_states[index]])[0]
        
        # must be one and only one state that is not specified
        if sum(1 for index in state_indices if index is None) != 1:
            raise ValueError("Specify fixed_state_indices to include all but one state.")
        
        # determine which state is not specified and is meant to be swept through
        for i,index in enumerate(state_indices):
            if index is None:
                sweep_index = i

        # sweep through the correct discrete state space and determine the probability values for each action
        probs_list = []
        for state in self.discrete_states[sweep_index]:
            state_indices[sweep_index] = q_learn.find_state_indices([self.discrete_states[sweep_index]], [state])[0]

            probs = q_learn.boltzmann(self.Q, tuple(state_indices), self.tau, return_probs=True)
            #probs = [1,2,3]
            probs_list.append(probs)

        return probs_list

    def reset_neighbors(self, neighbors):
        # NOTE: probably don't need this method.
        self.neighbors = neighbors
        self.neighbor_value_values = {neighbor: 0 for neighbor in neighbors}

    def reset_value(self):
        """
        Resets the value (or value) function by overwriting the "present" value using the "future" value.
        """
        self.total_value_present = self.total_value_future

    def configure_dynamic(self, error_type=None, yaw_offset=None, yaw_prop=None, tau=None, epsilon=None):
        """
        Configures a turbine agent to execute a dynamic simulation.

        Args:
            error_type: Int, no error (0), proportional error (1), or constant error (2)
            yaw_offset: Float, the yaw angle offset, if constant error
            yaw_prop: Float, the yaw angle percent error, if proportional error.
            tau: Float, the "temperature" value if using Boltzmann action selection
            epsilon: Float, the probability value if using epsilon-greedy action selection
        """
        if error_type is not None:
            self.error_type = error_type
        if yaw_offset is not None:
            self.yaw_offset = yaw_offset
        if yaw_prop is not None:
            self.yaw_prop = yaw_prop
        if tau is not None:
            self.tau = tau
        if epsilon is not None: 
            self.epsilon = epsilon

        #self.Q = np.zeros_like(self.Q)
        self.k = [0.9, 1, 0]
        #self.n = np.zeros_like(self.n)

    def utilize_q_table(self, state_name="yaw_angle", state_map={"wind_speed":None, "wind_direction":None}):
        """
        Uses a filled Q-table to deterministically determine which index in the state space maximizes
        expected reward. NOTE: this method is still in development and does not yet have complete 
        functionality.

        Args:
            axis: which dimension of the state-space to optimize over. Can be int or array_like, if more than one
            elements of the discrete state space are desired.

            target_action: index of the action in the action space that the function should find the 
            highest corresponding expected reward. For example, in a 3 action space, a typical value
            would be 1 (no change in yaw angle), and the function finds the yaw angle that corresponds
            to the state in which the highest expected reward is obtained with action 1.

            state: Optional state tuple that can be used to determine the appropriate LUT entry for a 
            given state.

            state_map: Dictionary mapping state names to their setpoint. Setting state names to None means that the value returned from the state space will be used.
        """
        

        state_values = self.sim_context.return_state(state_name)
        #states = self.sim_context.index_split(self.n, state_name, state_map)
        states_Q = self.sim_context.index_split(self.Q, state_name, state_map)
        blurred_Q = gaussian_filter(states_Q, sigma=[7,0])
        #plt.matshow(np.reshape(blurred_Q, (1, len(blurred_Q))))
        #max_index = np.argmax(states)
        #max_index_Q = np.argmax(states_Q[:,1])
        #max_index_Q = np.argmin(blurred_Q[:,0] + blurred_Q[:,2])
        #max_index_Q = np.shape(states_Q)[0]-1
        # for i in list(range(np.shape(states_Q)[0])):
        #     if blurred_Q[i,1] >= blurred_Q[i,0] and blurred_Q[i,1] >= blurred_Q[i,2]:
        #         max_index_Q = i
        #         break
        if state_values.observed:
            max_index_Q = 0
            diffs = blurred_Q[:,2] - blurred_Q[:,0]
            for i,diff in enumerate(diffs):
                if diff < 0:
                    max_index_Q = i
                    break
                if blurred_Q[i,0] == blurred_Q[i,1] and blurred_Q[i,1] == blurred_Q[i,2]:
                    # if there are no angles found for which the first condition is true, choose
                    # the first instance in which all actions have the same value
                    # NOTE: this assumes only three actions
                    max_index_Q = i
                    break
        else:
            max_index_Q = np.argmax(blurred_Q)

        # zero_crossings_inc = np.where(np.diff(np.sign(blurred_Q[:,2])))[0]
        # zero_crossings_dec = np.where(np.diff(np.sign(blurred_Q[:,0])))[0]

        #max_index_Q = np.argmax(states_Q[:,1] - (states_Q[:,0] + states_Q[:,2]))
        #max_index_Q = np.argmin( ( abs(states_Q[:,1] - states_Q[:,0]) + abs(states_Q[:,1] - states_Q[:,2]) ) / 2 )

        #plt.matshow(states_Q)
        state_values = self.sim_context.return_state(state_name)

        return state_values.discrete_values[max_index_Q]


    def inc_opt_counter(self, opt_window=100):
        """
        Increments an agent's obtimization counter and sets "done" to be true if the counter reaches opt_window.

        Args:
            opt_window: Int, specifies how long of an optimization window an agent gets
        """

        self.opt_counter += 1

        if self.opt_counter == opt_window - 1:
            self.done = True

    def reset_opt_counter(self):
        """
        Resets the agent's optimizatin counter to be 0 to allow for a new coordination window to begin.
        """
        
        self.opt_counter = 0
        self.done = False

    def process_shut_down(self):
        """
        This method defines how a turbine behaves when a turbine in its neighborhood shuts off.
        """
        # NOTE: if locking conditions are relaxed, this method will need to become more complex
        self.delay = 0
        return

class Server():
    """
    Server is a class handles communication tasks between TurbineAgent objects.

    Server contains a map for each turbine alias representing a channel of communication. A 
    TurbineAgent object can post information to its channel which can be read by other turbines.
    Server does not enforce communication or geographic constraints and will allow any turbine
    to read any channel, so these real-world constraints must be imposed by the find_neighbors
    function in TurbineAgent.

    Args:
        agents: A list of every TurbineAgent in the wind farm.

    Returns:
        Server: An instantiated Server object.
    """
    def __init__(self, agents):
        self.channels = {agent.alias: None for agent in agents}
        self.agents = agents

    def update_channel(self, alias, value):
        """
        Post a new value to the server.

        Args:
            alias: The alias that corresponds to the correct communication channel.
            value: The data that will be posted to the communication channel.
        """
        self.channels[alias] = value

    def read_channel(self, alias):
        """
        Read a value from the server.

        Args:
            alias: The alias that corresponds to the correct communication channel.
        
        Returns:
            The value posted to the alias' communication channel.
        """
        return self.channels[alias]

    def read_delta(self, alias):

        agent = self._find_agent(alias)

        delta_P = agent.self_value_future - agent.self_value_present
        #delta_control = agent.control_action_future - agent.control_action_present

        return delta_P

    def reset_channels(self, value):
        """
        Reset all channels to the same value

        Args:
            value: The value to set all of the channels to.
        """
        for alias in self.channels:
            self.channels[alias] = value

    def _find_agent(self, alias):
        """
        Helper function to determine if an alias is registered with the server.

        Args:
            alias: The alias that is being searched for.
        """
        for agent in self.agents:
            if agent.alias == alias: 
                return agent
        return None

    def lock(self, agent):
        """
        Locks turbines in a given turbines neighborhood (including the turbine itself) using the
        parameter delay_map.

        Args:
            agent: Agent that is initiating action and needs to lock other turbines.
        """
        # locking phase
        if agent.verbose: print(agent.alias, "calls server.lock()")

        # remove aliases that are shut down from the delay map
        new_delay_map = {}
        for alias in agent.delay_map:
            if not self._find_agent(alias).shut_down:
                new_delay_map[alias] = agent.delay_map[alias]

        agent.delay_map = new_delay_map

        #print(agent.delay_map)
        for alias in agent.delay_map:
            # NOTE: testing what happens when the turbine only locks itself
            # locked_turbine = self._find_agent(alias)
            # print(agent.alias, "locks", locked_turbine.alias, "for", agent.delay_map[alias])
            # locked_turbine.delay = agent.delay_map[alias]
            # break
            # END NOTE: remove preceding section to return to normal behavior

            # if there is only one alias in the delay map, it is the turbine itself, which means
            # that one of the downstream turbines is shut down, so the turbine does not need to lock
            # itself
            if alias == agent.alias and len(agent.delay_map) == 1:
                continue
            locked_turbine = self._find_agent(alias)
            # NOTE: this if statement might not be necessary
            if locked_turbine.delay == 0 and not locked_turbine.shut_down:
                # set locked_turbine delay according to the delay map
                locked_turbine.delay = agent.delay_map[alias]
                # set locked_by string so locked_turbine knows which agent initiated the lock
                locked_turbine.locked_by = agent.alias
                #if agent.verbose: print(locked_turbine.alias, "locked for", locked_turbine.delay, "seconds")
                #print(locked_turbine.alias, "locked for", locked_turbine.delay, "seconds")
            elif locked_turbine.delay != 0:
                # if the turbine is already delayed, then the turbines are ramping and turbines are moving simultaneously
                # in this case, the larger of the delay values should be assigned to the agent
                #print("In server.lock(), new value for", locked_turbine.alias, "is", str(agent.delay_map[alias]), "and old value is", str(locked_turbine.delay))
                locked_turbine.delay = max(agent.delay_map[alias], locked_turbine.delay)
                #print(locked_turbine.alias, "delay is set to", locked_turbine.delay)

    def unlock_all(self):
        """
        Clears the delay value for every turbine in the farm
        """
        for agent in self.agents:
            agent.delay = 0

    def check_neighbors(self, agent):
        """
        Examines neighboring turbines to see if a given turbine can move. This is used for instances in which a 
        wake delay is present and must be accounted for.

        Args:
            agent: The agent that is checking its neighbors.
        """
        if agent.delay > 0 or agent.target_state is not None:
            #if agent.alias == "turbine_0": print("Turbine 0 has delay > 0.")
            return False

        # NOTE: commenting out this block to see if less strict locking can be used
        for alias in agent.neighbors:
            if self._find_agent(alias).delay > 0 or self._find_agent(alias).target_state is not None:
                #if agent.alias == "turbine_0": print("Turbine 0 has neighbor with delay > 0.")
                return False
        # END NOTE

        return True

    def coordination_check(self, agent, coord):
        """
        Examines neighboring agents to determine if a given turbine can be given the "go-ahead" to begin its
        optimization cycle within a hierarchical coordination architecture.

        Args:
            agent: The agent that is checking for permission to begin optimization.
            coord: Which type of coordination to use. Current options are:
                - up_first: optimize from upstream to downstream
                - down_first: optimize from downstream to upstream
        """
        if agent.done:
            return False

        if coord == "up_first":
            other_turbines = agent.reverse_neighbors
        elif coord == "down_first":
            other_turbines = agent.neighbors
        else:
            raise ValueError("Invalid coordination algorithm choice.")

        if len(other_turbines) == 0:
            return True

        for alias in other_turbines:
            if not self._find_agent(alias).done:
                return False
        
        return True

    def reset_coordination_windows(self):
        """
        Resets all agents to allow coordination to be performed again at a new wind speed.
        """
        for agent in self.agents:
            agent.reset_opt_counter()

    def shut_down(self, agent):
        # this sets the delay of a turbine that 
        locked_by = self._find_agent(agent.locked_by)
        locked_by.process_shut_down()
        return

    def change_wind_direction(self, diff):
        # this method updates the wind direction differential entry for each agent
        for agent in self.agents:
            agent.wind_dir_diff += diff