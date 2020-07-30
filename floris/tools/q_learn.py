import numpy as np
import math
import random
import sys

# File created by Paul Stanfel for CSM-Envision Energy Research in Wind Farm Control; alpha version not yet validated.

'''
This script contains functions that are useful to implement Q-learning tasks with regards
to action selection and Bellman equation updating.
'''

def boltzmann(Q, indices, tau, return_probs=False):
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
    num_actions = np.shape(Q)[-1]
    
    p = np.zeros(num_actions)
    #NOTE: changed to new Q order
    #p = np.zeros(np.shape(Q)[0])

    Q_sum = 0 

    # adds a saturation to make sure that Boltzmann action selection does not overflow
    max_float = sys.float_info.max
    upper_lim = tau * math.log(max_float / num_actions)

    for i in range(len(p)):
        #Q_s = Q[i][indices]
        Q_s = min(Q[indices][i], upper_lim)
        #Q_s = Q[indices][i]
        Q_sum += math.exp(Q_s/tau)
    for i in range(len(p)):
        #Q_s = Q[i][indices]
        Q_s = min(Q[indices][i], upper_lim)
        #Q_s = Q[indices][i]
        p[i] = math.exp(Q_s/tau) / Q_sum

    for i in range(len(p)):
        Q_s = min(Q[indices][i], upper_lim)
        #Q_s = Q[indices][i]
        p[i] = math.exp(Q_s/tau) / Q_sum
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
    delta_V = deltas[0]
    delta_input = deltas[1]

    if delta_input == 0:
        # if there is a zero in the denominator, default to increasing
        action = 2
        return action

    grad = delta_V / delta_input

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