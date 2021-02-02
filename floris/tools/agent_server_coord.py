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
        alias (string): The name that the turbine will be referred to
            as.

        discrete_states: A list of lists containing the discrete
            states that the system can be in.

        farm_turbines: A map of every turbine in the farm from alias
            to a (x,y) coordinate tuple.

        observe_turbine_state: A function that returns the turbine
            state in the state space defined by discrete_states. The
            output of this function must have the same number
            of dimensions as discrete_states.

        modify_behavior: A function that returns a desired system
            parameter that can be used to modify the external system.
            This function uses the action selected by an exploration/
            explotation algorithm, so the function must map an
            integer to a system parameter like yaw angles.

        num_actions (int): The number of possible actions that the
            agent can take and which modify_behavior uses to map to
            change in system behavior.

        value_function: A function that returns system value. The
            algorithm attempts to maximize the output of this
            function in conjunction with all turbines in the
            neighborhood.

        find_neighbors: A function that populates self.neighbors with
            the aliases of turbines that can be communicated with.

        neighborhood_dims: A list [downwind, crosswind] of dimensions
            that correspond to the definition of a turbine's
            neighborhood with respect to the wind direction.

        leader (bool): NOTE: currently unused.

        model: An implementation-specific variable that encapsulates
            any information about the model of the environment that
            is needed to implement the user-defined method 
            observe_turbine_state, modify_behavior, value_function,
            and find_neighbors.

        power_reference: A power reference value that the individual
            turbine is trying to achieve. 

        yaw_prop (double): proportional yaw error.

        verbose (bool): when True prints out information about which
            methods are called and their arguments.

        yaw_offset (double): flat turbine offset error

        error_type (int): specifies what kind of error should be
            added to a measurement. This value can be used in the
            observe_turbine_state method. If an error is selected,
            the appropriate value (yaw_prop, yaw_offset, etc.) can be
            used. Current options are:
            - 0: no error
            - 1: proportional error
            - 2: offset error

        sim_factor (double): adjustment factor for simulation time
            steps greater than or less then one second. 
            NOTE: not fully implemented. 

        tau (double): Boltzmann tau parameter.

        epsilon (double): Epsilon-greedy epsilon parameter.

        discount (double): Q-learning discount factor parameter.

        sim_context: Input for SimContext object. Inputting this
            object will allow for other inputs to be left unspecified,
            although the other inputs to the TurbineAgent object are
            still implemented to avoid bugs.

        value_baseline (double): optional input for reward assignment
            algorithms that are compared to a given baseline.

        filtered (bool): when True agent will implement filtering of
            any noisy states.

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
                    value_baseline=0,
                    filtered=False):
    
        self.wake_delay = 0
        self.state_delay = 0
        self.power_delay = 0
        self.filter_window = 100

        self.sim_context = sim_context

        self.leader = leader # True if this turbine leads the consensus movement

        self.filtered = filtered # Boolean specifying whether states should be filtered or not

        self.yaw_prop = yaw_prop
        self.yaw_offset = yaw_offset
        self.error_type = error_type #0: no error, 1: proportional error, 2: offset error

        # variable that tracks wind direction differences
        self.wind_dir_diff = 0

        self.verbose = verbose
        self.farm_turbines = farm_turbines # dict mapping all aliases in the farm to (x,y) coordinates
        self.power = 0 # output power of this turbine
        self.alias = alias

        self.opt_yaw = self.sim_context.return_state(state_name="yaw_angle").discrete_values[0]

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
        self.completed_propagation = None

        self.delay = 0
        self.filter_delay = 0

        self.average_power = False
         # Boolean specifiying whether or not power should be filtered

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
            data = self.filter_power(self.model.turbine.power, append=True)#self.model.turbine.power

        #NOTE: this means that the gradient won't work in noisy wind conditions, will need to change where this is updated    
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

    def observe_state(self, watch_state=["wind_speed"], peek=False, fi=None):
        """
        Determines the tuple state_indices that correspond to the current system state (which can be used 
        to index Q) and the actual system state values. This method also updates the self.state_change
        variable, which is True if a change in the target_state occurred.

        Args:
            watch_state: An int representing what state(s) to use to
                determine whether a significant state
                change has occurred. 

            peek (bool): Specifies whether or not the internal
                state should be changed. When peek is True, this
                method will only check to see if the target state 
                has changed. When it is False, self.state and 
                self.state_indices will be modified.

            fi: FlorisInterface object that can optionally be 
                specified to pass in an object to read states
                from.
        """
        old_state = self.state

        use_filter = peek

        def method():
            return fi.floris.farm.wind_map.turbine_wind_speed[0]

        if fi is not None:
            observe_method = method
        else:
            observe_method = None
        #print("Calling _observe_turbine_state")
        # observe new state
        new_state = self._observe_turbine_state(self, use_filter=use_filter, method=observe_method)
     
        if self.verbose:
            print(self.alias, "setting state to", new_state)  
        #self.state = new_state
        #self.state_indices = self.sim_context.get_state_indices()#self._get_state_indices(self.state)
        if not peek:
            # Update n to reflect how many times this state has been visited

            self.state = new_state
            self.state_indices = self.sim_context.get_state_indices(self.state)
            self.n[self.state_indices] += 1
        else:
            # if peek is False, only update the noisy states
            temp_state = list(self.state)
            for i,state in enumerate(self.sim_context.obs_states):
                if state.noisy: temp_state[i] = new_state[i]
            self.state = tuple(temp_state)
            self.state_indices = self.sim_context.get_state_indices(self.state)

        # change state_change member variable if a state change occurred in a watched state
        self.state_change = False
        for state_name in watch_state:
            i = self.sim_context.find(state_name, return_index=True)
            if i is not None:
                if old_state[i] != new_state[i]:
                    # NOTE: temporarily commenting this out
                    self.state_change = True

    def ramp(self, state_name):
        """
        This method is intended to be called multiple times in order
        to facilitate a turbine's deterministic ramping capability.

        Args:
        state_name (string): The name of the state that will undergo
            ramping (this is typically "yaw_angle").
        """
        if self.state_change:

            state_map = {}
            for i,state in enumerate(self.sim_context.obs_states):
                state_map[state.name] = self.state[i]

            # this method will set the self.target_state variable
            # using utilize_q_table
            self.control_to_value(self.utilize_q_table(state_name,state_map=state_map, print_q_table=False, blur=True, sigma=1, method="first_swap", func=self.func), state_name)
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

            #print(self.alias, "returning yaw angle:", yaw_angle)

            return yaw_angle
        else:
            #print(self.alias, "returning None")
            self.delay_map = {}
            return None

    def control_to_value(self, target, control_state):
        """
        Sets an internal target parameter based on a designated control_state

        Args:
            target (double): The value that a given control variable
                should reach.

            control_state (string): Which state is controllable.
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
            server: A Server object that all the turbines communicate
                on.

            time (int): Indicates whether this is the
                "current" time (0) or the "future" 
                time (1). The corresponding variable is updated
                accordingly. This is needed to make 
                sure that the Q-learning algorithm can be properly
                executed.

            total: Boolean, indicates whether or not the 
        """

        return self._value_function(self, server, time)

    def handle_filters(self):
        """
        This method is designed to handle all filtering-related tasks.
        """

        # state_delay_done = False
        # power_delay_done = False
        # wake_delay_done = False

        # if self.state_delay > 0:
        #     self.state_delay -= 1

        #     if self.state_delay == 0:
        #         state_delay_done = True
        
        # if self.power_delay > 0:
        #     self.power_delay -= 1

        #     if self.power_delay == 0:
        #         power_delay_done = True

        # if self.wake_delay > 0:
        #     # wake_delay is only meant to prevent turbine from moving until it reaches 0, nothing needs to be done other than subtract 1
        #     self.wake_delay -= 1

        #     if self.wake_delay == 0:
        #         wake_delay_done = True

        # if wake_delay_done and (not state_delay_done and self.state_delay == 0) and (not power_delay_done and self.power_delay == 0):
        #     self.state_delay = self.filter_window
        #     self.power_delay = self.filter_window

        # this is in charge of iteratively accumulating bin counts for the "score-keeping" state estimation technique
        self.observe_state(peek=True)

    def check_delays(self):
        """
        Check if delays are at non-zero values.
        """
        return (self.wake_delay > 0) or (self.state_delay > 0) or (self.power_delay > 0)

    def filter_power(self, power, append=True):
        """
        Filters power readings in order to find the average power during a given interval (self.filter_window)

        Args:
            power: A noisy power reading

            append (bool): determines whether new power readings
                should be added to the list
        """
        if not hasattr(self, "_power_list"):
            self._power_list = []

            self.power_reading = power

            self._power_list.append(power)

            return power

        if self.average_power and self.power_delay > 0 and (len(self._power_list) < self.filter_window) and append:
            self._power_list.append(power)

        if not self.average_power:
            return power
        else:

            self.power_reading = np.average(self._power_list)

            return self.power_reading

    def start_power_state_filters(self):
        """
        Initializes state and power filters.
        """
        self._power_list = []
        self.sim_context.clear_bin_counts()

        self.power_delay = self.filter_window
        self.state_delay = self.filter_window

    def _select_action(self, action_selection="boltzmann"):
        """
        Chooses action based on Boltzmann search. This action must be mapped to a corresponding 
        change in system parameters by the function modify_behavior.

        NOTE: This method is not necessary and will soon be removed. Use take_action instead.

        Args:
            action_selection: The algorithm that will be used to select a control action.
        """
        print("_select_action is deprecated. Use take_action")
        if action_selection == "boltzmann":
            self.action = q_learn.boltzmann(self.Q, self.state_indices, self.tau)
        elif action_selection == "epsilon":
            self.action = q_learn.epsilon_greedy(self.Q, self.state_indices, self.epsilon)

    def _calculate_deltas(self, server):
        """
        Calculates change in value function and change in control input between iterations for use in, for
        example, a gradient-based action selection algorithm.

        Args:
            server: Server object so that the agent can learn
                information about neighbors.

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
        return deltas

    def take_action(self, action_selection="boltzmann", server=None, state_indices=None, return_action=False):
        """
        Chooses an action and returns a system parameter mapped via the function modify_behavior

        Args:
            action_selection (string): The algorithm that will be
                used to select a control action.
            server: NOTE: this input is not currently used
            state_indices: A tuple that, if specified, will be used
                as the state indices to find
                the Q entry at, not the current state_indices stored
                in self.state_indices.
            return_action (bool): specifiying whether or not the
                action that was selected should be returned.

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
        
        if action_selection == "boltzmann":
            # NOTE: this is the only method currently reconfigured for Q_obj
            action = q_learn.boltzmann(self.Q_obj, self.state, self.tau)
            if self.verbose:
                print(self.alias, "selects action", action)
        elif action_selection == "epsilon":
            action = q_learn.epsilon_greedy(self.Q, indices, self.epsilon)
        elif action_selection == "gradient":
            action = q_learn.gradient(self._calculate_deltas(server), step=(self.Q_obj.num_actions==3))

        elif action_selection == "hold":
            action = None

        
        if return_action:
            return action

        self.action = action
        return self._modify_behavior(self)

    def update_Q(self, threshold, reward_signal="constant", scaling_factor=100, set_reward=None):
        """
        This function assumes that a simulation has been run and total_value_future has been updated, and updates
        the internal Q-table based on which action is currently selected by the turbine agent.

        Args:
            threshold: If a constant reward signal, the threshold in
                the value function that is used to 
                determine if a significant change took place.

            reward_signal: What algorithm is used to allocate reward
                to the Q-Learning algorithm.

            scaling_factor: What value to scale the value function
                differential down by to prevent overflowing
                in the action selection routine.

            set_reward (double): What value, if any, to force the
                reward signal to take.
        """
        # skip this method if the turbine is currently ramping or is shut down
        if self.target_state is not None or self.shut_down:
            return

        # Determine learning rate.
        l = self.k[0] / (self.k[1] + self.k[2]*self.n[self.state_indices])
        #l = 0.5
        
        #diff = (self.total_value_future - self.total_value_present) + (self.total_value_future - self.value_baseline) / self.value_baseline 
        #NOTE: testing out different diff calculation

        if reward_signal == "variable" and set_reward is None:
            # Calculate difference between "future" value and "present" value.
            diff = (self.total_value_future - self.total_value_present)

            # NOTE: only remove scaling_factor if power is scaled already in push_data_to_server
            reward = diff/scaling_factor

        elif reward_signal == "constant" and set_reward is None:
            # Calculate difference between "future" value and "present" value.
            # if self.alias == "turbine_1":
            #     print("Inside update_Q, future value is", self.total_value_future, "and past value is", self.total_value_present)
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

            if self.verbose:
                print(self.alias, "receives reward signal of", reward)
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

        #Q_t = self.Q[self.action][self.state_indices]
        #Q_t = self.Q[self.state_indices][self.action]
        
        # accumulating traces
        #self.E[self.state_indices][self.action] = self.E[self.state_indices][self.action] + 1

        # Observe new state, using the internal function that doesn't overwrite any variables.
        future_state = self._observe_turbine_state(self)
        future_state_indices = self.sim_context.get_state_indices(targets=future_state)#self._get_state_indices(future_state)

        future_action = self.take_action(state_indices=future_state_indices, return_action=True)
        #next_Q = self.Q[future_state_indices][future_action]

        # The "future" Q value.
        #Q_t_1 = Q_t + l*(reward + self.discount*max_Q_t_1 - Q_t)

        # Update the Q table
        #self.Q[self.action][self.state_indices] = Q_t_1

        # print statement to see if table is being updated
        if set_reward is not None and self.verbose:
            print("Q table entry for state", self.state, "and action", self.action, "updated with reward", reward, "for agent", self.alias)

        #delta = reward + self.discount*max_Q_t_1 - Q_t

        #self.Q = self.Q + l*delta*self.E
        #self.E = self.discount*self.lamb*self.E

        self.Q_obj.update(self.state, self.action, reward, future_state, n=self.n)
        self.Q = self.Q_obj.return_q_table()

        # commented out temporarily to test eligibility traces
        #self.Q[self.state_indices][self.action] = Q_t_1
        #print(self.alias, "completes Q update with reward", reward)
        return reward

    def prob_sweep(self, fixed_state_indices, fixed_states):
        """
        NOTE: this method is not currently used or supported.
        """

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

    def reset_sim_context(self, fi):
        # reset fi that is used in the simulation context
        self.sim_context.reset(fi)

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
            error_type (int): no error (0), proportional error (1), or
                constant error (2).

            yaw_offset (double): the yaw angle offset, if constant 
                error.

            yaw_prop (double): the yaw angle percent error, if
                proportional error.

            tau (float): the "temperature" value if using Boltzmann
                action selection.

            epsilon (float): the probability value if using
                 epsilon-greedy action selection.
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
        #self.k = [0.9, 1, 0]
        self.n = np.zeros_like(self.n)

    def utilize_q_table(self, state_name="yaw_angle", state_map={"wind_speed":None, "wind_direction":None}, print_q_table=False, blur=False, sigma=2, return_table=False, method="smallest_diff", func=None):
        """
        Uses a filled Q-table to deterministically determine which
        index in the state space maximizes expected reward. This
        allows a turbine to choose a power maximizing setpoint.

        Args:
            state_name (string): State name specifying which state
                a setpoint should be returned for.

            state_map (dict): Dictionary mapping state names to their
                setpoint. Setting state names to None means that the
                value returned from the state space will be used.

            print_q_table (bool): Specifies whether or not the 
                Q-table fragment should be displayed visually.

            blur (bool): Specifies if a Gaussian blur should be
                applied.

            sigma (double): Sigma parameter to use if blur is True.

            return_table (bool): If True, the Q-table will be
                returned as an np array.

            method (string): If func is None, this parameter 
                specifies which of a set of predetermined 
                methods to use. Current options are:
                - smallest_diff: index of smallest difference
                    between increase and decrease actions
                - first_swap: index where decrease action first
                    surpasses increase action expected value, 
                    beginning at the lowest yaw angle.
                - lowest_total: index of the smalles combined
                    expected values of the increase and decrease
                    actions.
                - highest_stay: index corresponding to the 
                    highest expected value of the stay action
                    (assumes the stay action exists).
                - highest_stay_relative: index of the largest
                    difference between the stay action and the 
                    sum of the increase and decrease actions
                    (assumes the stay action exists)
                - one_past_highest_inc: one index past the index
                    of the highest expected value of the increase
                    action.
                - one_before_highest_dec: one index before the 
                    index of the highest expected value of the
                    decrease action.

            func: Custom user-defined function that, if specified,
                can run a different utilization algorithm than 
                is already defined using the method parameter.
        """
        state_values = self.sim_context.return_state(state_name)

        states_Q = self.sim_context.index_split(self.Q_obj.table, state_name, state_map)
        blurred_Q = gaussian_filter(states_Q, sigma=[sigma,0])

        if blur:
            read_table = blurred_Q
        else:
            read_table = states_Q

        if return_table:
            return read_table

        if print_q_table:
            plt.figure()
            plt.matshow(read_table)
            title = "Wind Speed: " + str(state_map["wind_speed"]) + "\n" + "Wind Direction: " + str(state_map["wind_direction"])
            plt.title(title, fontsize=20)
            plt.ylabel("Yaw Angle", fontsize=20)
            plt.yticks(fontsize=16)
            #plt.yticks([])
            plt.xlabel("Action", fontsize=20)
            plt.xticks(fontsize=16)
            plt.colorbar()

        if state_values.observed:

            if func is None:
                opt_index = 0
                max_index_Q = 0
                if self.Q_obj.num_actions == 2:
                    diffs = read_table[:,1] - read_table[:,0]
                    sums = read_table[:,1] + read_table[:,0]
                elif self.Q_obj.num_actions == 3:
                    diffs = read_table[:,2] - read_table[:,0]
                    sums = read_table[:,2] + read_table[:,0]

                if method == "smallest_diff":
                    initialized = False
                    for i, (a0,a1) in enumerate(zip(read_table[:,0], read_table[:,1])):
                        if ((a0 != 0) or (a1 !=0)) and not initialized:
                            min_diff_index = i
                            min_diff = np.abs(a0 - a1)
                            initialized = True
                        
                        if not(a0 == 0 and a1 == 0) and np.abs(a0 - a1) < min_diff:
                            min_diff_index = i
                            min_diff = np.abs(a0 - a1)
                    
                    opt_index = min_diff_index

                elif method == "first_swap":
                    #opt_index = self.sim_context.return_state(state_name="yaw_angle").get_index(target=self.opt_yaw)
                    for i,diff in enumerate(diffs):
                        if diff < 0:
                            opt_index = i
                            break
                        # if read_table[i,0] == read_table[i,1] and read_table[i,1] == read_table[i,2]:
                        #     # if there are no angles found for which the first condition is true, choose
                        #     # the first instance in which all actions have the same value
                        #     # NOTE: this assumes only three actions
                        #     max_index_Q = i
                        #     break

                elif method == "lowest_total":
                    opt_index = np.argmin(sums)

                elif method == "highest_stay":
                    # NOTE: this only works if there is a stay option, which is assume to be action 1
                    opt_index = np.argmax(read_table[:,1])

                elif method == "highest_stay_relative":
                    # NOTE: this only works if there is a stay option, which is assume to be action 1
                    opt_index = np.argmax(read_table[:,1] - (read_table[:,0] + read_table[:,2]))

                elif method == "one_past_highest_inc":
                    if self.Q_obj.num_actions == 2:
                        opt_index = np.argmax(read_table[:,1]) + 1
                    elif self.Q_obj.num_actions == 3:
                        opt_index = np.argmax(read_table[:,2]) + 1
                    opt_index = min(len(self.sim_context.return_state("yaw_angle").discrete_values), opt_index)

                elif method == "one_before_highest_dec":
                    opt_index = np.argmax(read_table[:,0]) - 1
                    opt_index = max(0, opt_index)
            else:
                opt_index = func(read_table)

        else:
            opt_index = np.argmax(read_table)

        state_values = self.sim_context.return_state(state_name)

        return state_values.discrete_values[opt_index]


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
            alias: The alias that corresponds to the correct
                communication channel.
            value: The data that will be posted to the communication
                channel.
        """
        self.channels[alias] = value

    def read_channel(self, alias):
        """
        Read a value from the server.

        Args:
            alias: The alias that corresponds to the correct
                communication channel.
        
        Returns:
            The value posted to the alias' communication channel.
        """
        return self.channels[alias]

    def read_delta(self, alias):

        agent = self._find_agent(alias)

        delta_P = agent.self_value_future - agent.self_value_present

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
            agent: Agent that is initiating action and needs to lock
                other turbines.
        """
        # locking phase
        if agent.verbose: print(agent.alias, "calls server.lock()")

        # remove aliases that are shut down from the delay map
        new_delay_map = {}
        for alias in agent.delay_map:
            if not self._find_agent(alias).shut_down:
                new_delay_map[alias] = agent.delay_map[alias]

        agent.delay_map = new_delay_map

        if agent.verbose:
            print(agent.alias, "delay_map:", agent.delay_map)

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
            if agent.verbose:
                print(agent.alias, "wake_delay is", agent.wake_delay)
            if alias == agent.alias and len(agent.delay_map) == 1:
                continue
            locked_turbine = self._find_agent(alias)
            # NOTE: this if statement might not be necessary
            if locked_turbine.wake_delay == 0 and not locked_turbine.shut_down:
                # set locked_turbine delay according to the delay map
                locked_turbine.delay = agent.delay_map[alias]
                locked_turbine.wake_delay = agent.delay_map[alias]
                if agent.verbose:
                    print(agent.alias, "sets", locked_turbine.alias, "wake delay to", locked_turbine.wake_delay)
                # set locked_by string so locked_turbine knows which agent initiated the lock
                locked_turbine.locked_by = agent.alias

            elif locked_turbine.wake_delay != 0:
                # if the turbine is already delayed, then the turbines are ramping and turbines are moving simultaneously
                # in this case, the larger of the delay values should be assigned to the agent

                locked_turbine.delay = max(agent.delay_map[alias], locked_turbine.delay)
                locked_turbine.wake_delay = max(agent.delay_map[alias], locked_turbine.delay)
                if agent.verbose:
                    print(agent.alias, "sets", locked_turbine.alias, "wake delay to", locked_turbine.wake_delay)

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
        if agent.check_delays() or agent.target_state is not None:
            return False

        for alias in agent.neighbors:
            if self._find_agent(alias).check_delays() or self._find_agent(alias).target_state is not None:
                #if agent.alias == "turbine_0": print("Turbine 0 has neighbor with delay > 0.")
                return False

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
        """
        Disables turbine and prevents it from moving.
        """
        locked_by = self._find_agent(agent.locked_by)
        locked_by.process_shut_down()
        return

    def change_wind_direction(self, diff):
        """
        Updates the wind direction differential entry for each agent.
        NOTE: this method is not currently used or implemented fully.
        """
        for agent in self.agents:
            agent.wind_dir_diff += diff