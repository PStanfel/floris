import numpy as np
import math
import random
import sys
from collections import deque

# File created by Paul Stanfel for CSM-Envision Energy Research in Wind Farm Control; alpha version not yet validated.

'''
This script contains functions that are useful to implement Q-learning tasks with regards
to action selection and Bellman equation updating.
'''

class Q():
    '''
    This class is intended to encapsulate the Q lookup and update processes. It should be 
    able to take in a state tuple and output a list of action values.

    Args:
        states: list of simulation states.
    '''
    def __init__(self, states):
        self.states = states
        self.obs_states = [state for state in self.states if state.observed]

        if all([state.state_type == "discrete" for state in self.states]):
            # should be a tabular function approximation only if all states are discrete
            self.approx_type = "table"
        else:
            self.approx_type = "ann"

        # the number of actions is the product of all the actions of all the controllable states
        self.num_actions = np.prod([state.num_actions for state in states if state.controlled])

        self.discount = 0.5

        if self.approx_type == "table":
            dim = [len(state.discrete_values) for state in self.obs_states]
            dim.append(self.num_actions)
            self.table = np.zeros(tuple(dim))
            self.E = np.zeros_like(self.table)
            self.lamb = 0
            self.k = [1,1,1]
        elif self.approx_type == "ann":
            # this import is slow, only do so if necessary
            from keras import backend as K
            from keras.models import Sequential
            from keras.layers import Dense, Activation
            from keras.optimizers import Adam
            from keras.metrics import mean_squared_error

            num_states = len(self.obs_states)
            num_actions = self.num_actions

            self.policy_net = Sequential([
                Dense(16, input_dim=num_states, activation='relu'),
                Dense(32, activation='relu'),
                Dense(16, activation='relu'),
                Dense(num_actions)
            ])
            
            self.target_net = self.policy_net

            self.policy_net.compile(Adam(lr=0.0001), loss='mean_squared_error')

            self.target_net.compile(Adam(lr=0.0001), loss='mean_squared_error')

            self.memory = deque(maxlen=100)

            self.batch_size = 10

            self.update_counter = 0
            self.update_threshold = 10

    def _get_state_indices(self, state_values):
        state_indices = []
        for i,state in enumerate(self.obs_states):
            state_indices.append(state.get_index(target=state_values[i]))

        return tuple(state_indices)

    def read(self, state_values):
        if len(state_values) != len(self.obs_states):
            error = "Invalid number of state values. state_values must have " + str(len(self.states)) + " elements."
            raise ValueError(error)

        if self.approx_type == "table":
            state_indices = self._get_state_indices(state_values)

            return self.table[state_indices]
        elif self.approx_type == "ann":
            s = np.reshape(state_values, (1, -1))

            return self.policy_net.predict(s)[0]

    def _modify_target_net(self):
        weights = self.policy_net.get_weights()
        target_weights = self.target_net.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_net.set_weights(target_weights)

    def update(self, state_values, action, reward, future_state_values, update_type="q_learn", future_action=None, n=None):
        valid_update_types = ["q_learn", "sarsa"]

        if update_type not in valid_update_types:
            raise ValueError("Invalid update type specified.")

        if update_type == "sarsa" and future_action is None:
            raise ValueError("Must specify future action index with SARSA update.")

        if len(state_values) != len(self.obs_states):
            error = "Invalid number of state values. state_values must have " + str(len(self.states)) + " elements."
            raise ValueError(error)

        if self.approx_type == "table":
            if n is None:
                raise ValueError("Must specify n for tabular function approximation.")

            state_indices = self._get_state_indices(state_values)

            # Determine learning rate.
            l = self.k[0] / (self.k[1] + self.k[2]*n[state_indices])

            Q_t = self.table[state_indices][action]

            # accumulating traces
            self.E[state_indices][action] += 1

            future_state_indices = self._get_state_indices(future_state_values)

            if update_type == "q_learn":
                target = np.max(self.table[future_state_indices])
            elif update_type == "sarsa":
                target = self.table[future_state_indices][future_action]

            delta = reward + self.discount*target - Q_t

            self.table = self.table + l*delta*self.E
            self.E = self.discount*self.lamb*self.E

            return
        elif self.approx_type == "ann":
            self.memory.append([np.array(state_values), action, reward, np.array(future_state_values)])

            if len(self.memory) < self.batch_size:
                return

            samples = random.sample(self.memory, self.batch_size)
            for sample in samples:
                s, a, r, s_ = sample

                # change dimensions for use in keras model
                s = np.reshape(s, (1,-1))
                s_ = np.reshape(s_, (1,-1))

                target = self.target_net.predict(s)

                Q_t_1 = max(self.target_net.predict(s_)[0])

                target[0][a] = r + self.discount*Q_t_1

                self.policy_net.fit(s, target, epochs=1, verbose=0)

            self.update_counter += 1

            if self.update_counter == self.update_threshold:
                self._modify_target_net()
                self.update_counter = 0

    def return_q_table(self):
        if self.approx_type == "table":
            return self.table
        elif self.approx_type == "ann":
            # there is no table for the ANN
            return np.zeros(1)
            #raise ValueError("Can't return Q-table for continuous state space.")

def boltzmann(Q_obj, state_values, tau, indices=None, return_probs=False):
    """"
    Performs a Boltzmann exploration search for use in a Q-learning algorithm.

    Args:
        Q: An RL Q table.
        indices: A tuple of indices within the state space that represent the current space of the turbine.
        tau: A positive value that determines exploration/exploitation tradeoff.
        return_probs: Boolean, indicates whether or not the method should return an array of
            probabilities or simply choose an action.

    Returns:
        action: An int representing which action should be selected. This must be interpreted by the 
        modify_behavior function.
    """
    num_actions = Q_obj.num_actions#np.shape(Q)[-1]
    Q_values = Q_obj.read(state_values=state_values)

    p = np.zeros(num_actions)
    #NOTE: changed to new Q order
    #p = np.zeros(np.shape(Q)[0])

    Q_sum = 0 

    # adds a saturation to make sure that Boltzmann action selection does not overflow
    max_float = sys.float_info.max
    upper_lim = tau * math.log(max_float / num_actions)

    for i in range(len(p)):
        #Q_s = Q[i][indices]
        #Q_s = min(Q[indices][i], upper_lim)
        Q_s = min(Q_values[i], upper_lim)
        #Q_s = Q[indices][i]
        Q_sum += math.exp(Q_s/tau)
    for i in range(len(p)):
        #Q_s = Q[i][indices]
        #Q_s = min(Q[indices][i], upper_lim)
        Q_s = min(Q_values[i], upper_lim)
        #Q_s = Q[indices][i]
        p[i] = math.exp(Q_s/tau) / Q_sum

    # for i in range(len(p)):
    #     Q_s = min(Q[indices][i], upper_lim)
    #     #Q_s = Q[indices][i]
    #     p[i] = math.exp(Q_s/tau) / Q_sum
    # p is a vector with probabilities of selecting an action
    # p must be interpreted to correspond to a given action

    if return_probs:
        return p

    N_r = random.uniform(0, 1)

    max_p = p[0]
    action = len(p) - 1
    for i in range(len(p)-1):
        if N_r < max_p:
            action = i
            break
        else:
            max_p += p[i+1]
    
    return action

def epsilon_greedy(Q, indices, epsilon):
    """
    Performs an epsilon-greedy action selection procedure.

    Args:
        Q: An RL Q table.
        indices: A tuple of indices within the state space that represent the current space of the turbine.
        tau: A positive value that determines exploration/exploitation tradeoff.

    Returns:
        action: An int representing which action should be selected. This must be interpreted by the 
        modify_behavior function.
    """

    #NOTE: changed to new Q order
    #num_actions = np.shape(Q)[0]
    num_actions = np.shape(Q)[-1]
    initial_index = random.choice(list(range(num_actions)))
    
    #NOTE: changed to new Q order
    # max_Q = Q[initial_index][indices]
    # best_action = initial_index

    # for i in range(num_actions):
    #     if Q[i][indices] > max_Q:
    #         max_Q = Q[i][indices]
    #         best_action = i
    best_action = np.argmax(Q[indices])

    action_probs = np.zeros(num_actions)

    for i in range(num_actions):
        if i == best_action:
            action_probs[i] = (1 - epsilon) + epsilon / num_actions
        else:
            action_probs[i] = epsilon / num_actions

    N_r = random.uniform(0, 1)

    total = action_probs[0]
    action = len(action_probs) - 1
    for i in range(len(action_probs)-1):
        if N_r < total:
            action = i
            break
        else:
            total += action_probs[i+1]
    
    return action

def gradient(deltas):
    """
    Performs a deterministic gradient control action update based on first-order backward differencing.

    Args:
        deltas: An iterable that has the difference in value function as its first element and the difference
        in control input as its second element.

    Returns:
        action: An int representing which action should be selected. 0 means decrease, 1 means stay, and 2 
        means increase. Unlike the other action selection algorithms, this cannot be interpreted otherwise
        by the modify_behavior method.
    """
    # delta_V = deltas[0]
    # delta_input = deltas[1]

    # if delta_input == 0:
    #     # if there is a zero in the denominator, default to increasing
    #     action = 2
    #     return action

    # grad = delta_V / delta_input
    if any([pair[1] == 0 for pair in deltas]):
        action = 2
        return action

    grad = 0
    for pair in deltas:
        grad += pair[0] / pair[1]

    if grad < 0:
        # decrease if the gradient is negative
        action = 0
    elif grad >= 0:
        # >= means that the control input will default to increasing if the gradient is zero
        action = 2

    return action

def find_state_indices(discrete_states, measured_values):
    """
    Determines the indices in a discretized state space that correspond most closely to the measured state values.

    Args:
        discrete_states: A list of lists representing the discrete state space for every state variable.
        measured_values: The measured state value for each state variable, in the same order as discrete_states.
    """
    # Determine indices of state values that are closest to the discretized states
    state_indices = []

    for i in range(len(discrete_states)):
        index = np.abs(discrete_states[i] - measured_values[i]).argmin()
        state_indices.append(index)
    # Tuple (k,j,...) which refers to indices in the state space that correspond to the current system state
    return tuple(state_indices)

def max_Q(Q, indices):
    """
    Determines the maximum expected Q value for use in the Q-learning algorithm

    Args:
        Q: An RL Q table.
        indices: A tuple of indices within the state space that represent the current space of the turbine.
    """
    #NOTE: changed to new Q order
    # max_value = Q[0][indices]
    # for i in range(np.shape(Q)[0]):
    #     if Q[i][indices] > max:
    #         max_value = Q[i][indices]
    max_value = np.max(Q[indices])
    return max_value