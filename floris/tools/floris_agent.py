# File created by Paul Stanfel for CSM-Envision Energy Research in Wind Farm Control; alpha version not yet validated.

'''
This file contains functions that are designed to simplify the execution of the farm training 
scripts. These are intended to be passed into a TurbineAgent object to customize its
behavior.
'''

from floris.tools import agent_server_coord
import math
import random
import numpy as np
from floris.tools import iterate
import itertools
import copy
from floris.tools import q_learn

# Helper functions

def _adjust_coordinates(position, phi):
    """
    Rotates coordinates to a given wind direction angle for easier comparison of wind turbine
    positions relative to the wind direction.

    Args:
        position: Tuple of (x,y) coordinates of a turbine.
        phi: Float, wind direction in radians
    """
    adjusted_x = position[0] * math.cos(phi) + position[1] * math.sin(phi)
    adjusted_y = -position[0] * math.sin(phi) + position[1] * math.cos(phi)
    return (adjusted_x, adjusted_y)

def _set_delay_map(turbine_agent):
    """
    Rotates a wind farm to the wind direction and determines which turbines should be locked
    by the control action of a given agent, setting the TurbineAgent delay_map dictionary
    accordingly.

    Args: 
        turbine_agent: TurbineAgent object to be set.
    """
    phi = turbine_agent.model.wind_direction() + 180
    phi *= math.pi/180

    self_adjusted_position = _adjust_coordinates(turbine_agent.position, phi)

    delay_map = {}

    # set the delay of all downstream turbines based on the action that a turbine just selected
    for alias in turbine_agent.farm_turbines:
        if not alias == turbine_agent.alias and alias in turbine_agent.neighbors:
            adjusted_position = _adjust_coordinates(turbine_agent.farm_turbines[alias], phi)

            x = self_adjusted_position[0] - adjusted_position[0]
            U_inf = turbine_agent.model.wind_speed()

            # if the simulation resolution is not 1 second, this delay value must change
            tau = x / U_inf

            delay_map[alias] = round(tau)
    
    if len(delay_map) > 0:
        self_delay = max([delay_map[alias] for alias in delay_map])
        delay_map[turbine_agent.alias] = self_delay

        # NOTE: testing what happens when the turbine only locks itself
        # self_delay = max([delay_map[alias] for alias in delay_map])
        # turbine_agent.delay_map = {turbine_agent.alias: self_delay}
        # END NOTE
    else:
        turbine_agent.delay_map = {}

    # NOTE: comment this out if above note is uncommented
    turbine_agent.delay_map = delay_map
    # END NOTE

# Miscellaneous functions/classes

class FlorisModel():
    """
    Model that encapsulates all modeling information that a TurbineAgent would need.

    Args:
        fi: FlorisUtilities object.
        turbine: Turbine object.

    Returns:
        FlorisModel: An instantiated FlorisModel object.
    """
    def __init__(self, fi, turbine, index):
        self.fi = fi
        self.turbine = turbine
        self.index = index

        self.state_map = {"yaw_angle": self.yaw_angle,
                          "wind_speed": self.wind_speed,
                          "wind_direction": self.wind_direction}

    def yaw_angle(self):
        
        yaw_angle = self.turbine.yaw_angle + self.wind_direction()
        #print("Wind direction to add to yaw angle:", self.fi.floris.farm.wind_direction)
        # if any(self.fi.wind_dir_change_turb):
        #     print(self.fi.wind_dir_change_turb)
        #     # self.fi.wind_dir_change is a list of booleans, so if any of them are true the wind direction needs to be shifted
        #     yaw_angle -= self.fi.wind_dir_shift
        #     print("Wind dir shift of", self.fi.wind_dir_shift, "subtracted.")
            
        #     # NOTE: this only works for instantaneous farm wind direction changes
        #     for i,flag in enumerate(self.fi.wind_dir_change_turb):
        #         if flag:
        #             # change one True element to False
        #             self.fi.wind_dir_change_turb[i] = False
        #             break
        #print("Absolute yaw angle:", yaw_angle)
        return yaw_angle

    def wind_speed(self):
        return self.fi.floris.farm.wind_map.turbine_wind_speed[self.index]#turbine_wind_speed[self.index]

    def wind_direction(self):
        wind_dir = self.fi.floris.farm.wind_map.turbine_wind_direction[self.index]
        if wind_dir > 180: wind_dir -= 360

        return wind_dir

    def get_state_methods(self, state_names):
        state_methods = {}
        for state_name in state_names:
            if state_name in self.state_map.keys():
                state_methods[state_name] = self.state_map[state_name]
            else:
                error = state_name + " is not a valid state."
                raise ValueError(error)

        return state_methods

class State():
    """
    Class that encapsulates the behavior of a simulation state.

    Args:
        name: Name of the state (string).
        method: Function that describes what should be returned by the State object
        state_type: Specifies whether state is continuous or discrete (string). Current options are:
            - continuous
            - discrete
        discrete_values: If the state is discrete, a list of values that the state space can be. The value returned by method will be set to the closest value in this list (array-like).
        controlled: Whether this state can be controlled or not (boolean). NOTE: currently, controllable states are assumed to be discrete.
        error_type: The type of measurement error (string). Current options are:
            - prop: proportional error
            - offset: static offset error
            - none: no error
        error_value: The value, if any, to adjust the state measurement by, according to the type of error selected by error_type (float).
        action_type: Specifies how actions are to be selected (string). NOTE: these are designed currently around yaw misalignment control, and also assume a discrete yaw state space. Current options are:
            - step: the turbine has the choice of either not moving, increasing, or decreasing by an interval defined by the discrete state space.
            - jump: the turbine chooses a yaw angle from any of the possible yaw angles in the discrete state space, at 1 degree intervals.
            - skew: the turbine can either choose the yaw angle passed by the skew argument, or choose a yaw angle above or below the skew argument. The amount above or below is currently hardcoded to 5. NOTE: this action type is not complete.
        skew: The yaw angle that is used to determine yaw setpoints under the "skew" action type (float).

    Returns:
        FlorisModel: An instantiated FlorisModel object.
    """
    def __init__(self, name, method, state_type, discrete_values=None, controlled=False, observed=True, error_type="none", error_value=0, action_type="step", skew=0):
        self.name = name
        self.method = method
        self.state_type = state_type
        self.discrete_values = discrete_values
        self.error_type = error_type
        self.error_value = error_value
        self.controlled = controlled
        self.observed = observed

        self.get_state()

        valid_errors = ["offset", "prop", "none"]
        valid_states = ["discrete", "continuous"]

        if self.error_type not in valid_errors:
            error = self.name + " has invalid error type."
            raise ValueError(error)

        if self.state_type not in valid_states:
            error = self.name + " has invalid state type."
            raise ValueError(error)

        if self.state_type == "discrete" and self.discrete_values is None:
            raise ValueError("Discrete state type must have discrete state space specified.")

        self.action_type = action_type

        if self.action_type == "step":
            # this means that actions are inc, dec, and stay
            self.num_actions = 3
        elif self.action_type == "jump":
            # this means that actions are a yaw angle setpoint at one degree increments
            self.num_actions = int(round(discrete_values[-1]) - round(discrete_values[0]))
        elif self.action_type == "skew":
            self.num_actions = 3
            self.skew = skew

    def get_state(self, target=None, round_to_bin=True):
        """
        Returns the errored state or the closest discrete state value to the errored state.

        Args:
            target: An optional value that specifies which value the closest state value should be returned for. If None, value will be returned for the current 
            (errored) state value.
            round_to_bin: Whether or not the errored state should be rounded to a discrete state bin(boolean).
        """
        if not self.state_type == "discrete" and round_to_bin:
            # this method is only effective for discrete state spaces
            return None

        if target is None:
            actual_value = self.method()

            # include error type into the measurement
            if self.error_type == "offset":
                errored_value = actual_value + self.error_value
            elif self.error_type == "prop":
                errored_value = actual_value / (1 - self.error_value)
            elif self.error_type == "none":
                errored_value = actual_value

            if round_to_bin:
                #return errored_value
                return self.discrete_values[self.get_index(target=errored_value)]
            else:
                return errored_value

        else:
            if round_to_bin:
                return self.discrete_values[self.get_index(target=target)]
            else:
                return target

    def get_index(self, target=None):
        """
        Returns an index into the discrete state space.

        Args:
            target: An optional value that specifies which value the index should be returned
            for. If None, index will be returned for the current (errored) state value.
        """
        if not self.state_type == "discrete":
            # this method is only effective for discrete state spaces
            return None

        # Determine indices of state values that are closest to the discretized states
        if target is None:
            index = np.abs(self.discrete_values - self.method()).argmin()
        else:
            index = np.abs(self.discrete_values - target).argmin()

        return index

    def set_state(self, agent):
        """
        Returns a value that can be passed into FLORIS to be run. While trying to be agnostic to the type of control action, this method was built to handle yaw angle control. NOTE: this assumes that the state is discrete. 

        Args:
            agent: TurbineAgent object that is returning a control action.
        """
        
        if not self.controlled:
            # state can't be set if this is not a controllable state
            return None

        # Calculate the smallest allowable yaw increment
        delta_state_value = self.discrete_values[1] - self.discrete_values[0] 

        if self.action_type == "jump":
            # under jump action type, action number corresponds directly to yaw angle setpoint.
            # BUG: what about negative yaw angles?
            return agent.action
        elif self.action_type == "skew":
            # under skew sction type, turbine can oscillate by a fixed amount around a setpoint determined by self.skew. Oscillation amount currently hardcoded to 5 degrees.
            delta_state_value = 5
            yaw_angle = ((agent.action % 3) - 1)*delta_state_value + self.skew
            return yaw_angle

        state_value = self.get_state()

        if agent.target_state is None:

            # set "present" value to calculate delta, if needed (for example, in a gradient)
            agent.control_action_present = agent.control_action_future

            # action 0 -> decrement yaw angle
            # action 1 -> keep constant yaw angle
            # action 2 -> increment yaw angle
            if agent.action == 0:
                state_value = state_value - delta_state_value
            elif agent.action == 1:
                state_value = state_value
            elif agent.action == 2:
                state_value = state_value + delta_state_value
            elif agent.action is None:
                state_value = state_value
            else:
                print("Invalid action chosen.")

            # make sure yaw angle does not exceed limits of state space
            if state_value < self.discrete_values[0]:
                state_value = self.discrete_values[0]
            elif state_value > self.discrete_values[-1]:
                state_value = self.discrete_values[-1]

            # set "future" value to calculate delta, if needed (for example, in a gradient)
            agent.control_action_future = state_value

            # adjust yaw angle observation based on what type of error is being used
            if self.error_type == "offset":
                state_value_error = state_value - self.error_value
            elif self.error_type == "prop":
                state_value_error = state_value * (1 - self.error_value)
            elif self.error_type == "none":
                state_value_error = state_value

            # calculate delay for all downstream turbines within the neighborhood
            if not agent.action == 1:
                _set_delay_map(agent)
            else:
                agent.delay_map = {}

            return state_value_error
        else:
            agent.delay_map = {}
            diff = agent.target_state[0] - state_value
            yaw_rate = agent.model.turbine.yaw_rate
            if abs(diff) < yaw_rate:
                state_value = agent.target_state[0]
                agent.target_state = None
                # NOTE: test
                #turbine_agent.delay_map = {}
            else:
                state_value += np.sign(diff) * agent.model.turbine.yaw_rate
                #print(agent.alias, str(np.sign(diff) * agent.model.turbine.yaw_rate))
                #print(agent.alias, str(state_value))
                _set_delay_map(agent)
                

            # adjust yaw angle observation based on what type of error is being used
            if self.error_type == "offset":
                state_value_error = state_value - self.error_value
            elif self.error_type == "prop":
                state_value_error = state_value * (1 - self.error_value)
            elif self.error_type == "none":
                state_value_error = state_value
            #print(turbine_agent.alias, "yaw angle:", yaw_angle)
            return state_value_error

class SimContext():
    """

    error_info: {state_name: (error_type, value)}
    discrete_state_map: {state_name: discrete_states} (order matters)

    state_info: {state_name: (discrete_states, error_type, error_value)}
    """

    def __init__(self, states):

        self.states = states
        self.obs_states = [state for state in self.states if state.observed]

    def blank_tables(self):   

        dim = [len(state.discrete_values) for state in self.obs_states]
        n = np.zeros(tuple(dim))

        num_action_list = []

        for state in self.obs_states:
            num_action_list.append(state.num_actions)
        
        if len(num_action_list) == 0:
            total_num_actions = 0
        else:
            total_num_actions = 1
            for num_actions in num_action_list:
                total_num_actions = total_num_actions*num_actions
        
        dim.append(total_num_actions)
        Q = np.zeros(tuple(dim)) # internal Q-table

        Q_obj = q_learn.Q(self.states)

        return [n, Q, Q_obj]

    def observe_state(self, agent):

        state_values = []

        for state in self.obs_states:
            state_value = state.get_state()

            # # add yaw angle to wind direction to get absolute yaw angle
            # if state.name == "yaw_angle":
            #     state_value += wind_dir
            #     print("observed yaw angle:", state_value)
            state_values.append(state_value)

        return tuple(state_values)

    def modify_behavior(self, agent):
        wind_dir = 0

        # if wind direction and yaw angle are both states, yaw angle will need to be added to wind direction
        if self.find("yaw_angle") is not None and self.find("wind_direction") is not None:
            wind_dir = self.find("wind_direction").get_state(round_to_bin=False)
        setpoints = []

        for state in self.states:
            if state.controlled == True:
                setpoint = state.set_state(agent)

                # subtract yaw angle from wind direction to get absolute yaw angle
                if state.name == "yaw_angle" and setpoint is not None:
                    setpoint -= wind_dir

                setpoints.append(setpoint)
        return tuple(setpoints)

    def index_split(self, table, state_name="yaw_angle", state_map={"wind_speed":None, "wind_direction":None}):

        before_indices = []
        after_indices = []

        append_array = before_indices

        sub_array = copy.deepcopy(table)

        for state in self.obs_states:
            if state.name != state_name:
                append_array.append(state.get_index(state_map[state.name]))
            else:
                append_array = after_indices
        #print("before_indices: ", before_indices)
        #print("after_indices: ", after_indices)
        for index in before_indices:
            sub_array = sub_array[index]

        for index in after_indices:
            sub_array = sub_array[:,index]

        return sub_array

    def return_state(self, state_name):
        for state in self.states:
            if state.name == state_name:
                return state

        error = state_name + " is not a valid state name."
        raise ValueError(error)

    def change_error(self, state_name, error_type, error_value):
        state = self.return_state(state_name)

        state.error_type = error_type
        state.error_value = error_value

        return

    def get_state_indices(self, targets=None):
        state_indices = []
        for i,state in enumerate(self.obs_states):
            if targets is None:
                state_indices.append(state.get_index())
            else:
                state_indices.append(state.get_index(target=targets[i]))

        return tuple(state_indices)

    def find(self, state_name, return_index=False):

        for i,state in enumerate(self.states):
            if state.name == state_name:
                if return_index:
                    return i
                else:
                    return state
        
        return None

def find_neighbors(turbine_agent):
    """
    Determine which turbines in the wind farm are in the neighborhood of a given agent.

    Args:
        turbine_agent: A TurbineAgent object.
    """
    downwind = turbine_agent.downwind
    crosswind = turbine_agent.crosswind

    phi = turbine_agent.model.wind_direction() + 180
    phi *= math.pi/180

    # rotate turbine
    self_adjusted_position = _adjust_coordinates(turbine_agent.position, phi)
    
    # rotate other turbines in wind farm and use neighborhood dimensions in TurbineAgent
    # to determine which turbines are in the rectangular neighborhood
    for alias in turbine_agent.farm_turbines:
        if not alias == turbine_agent.alias:
            adjusted_position = _adjust_coordinates(turbine_agent.farm_turbines[alias], phi)

            downwind_delta = self_adjusted_position[0] - adjusted_position[0]

            crosswind_delta = self_adjusted_position[1] - adjusted_position[1]

            if abs(crosswind_delta) <= crosswind and abs(downwind_delta) <= downwind:
                if downwind_delta > 0:
                    if alias not in turbine_agent.neighbors: turbine_agent.neighbors.append(alias)
                if downwind_delta < 0:
                    if alias not in turbine_agent.reverse_neighbors: turbine_agent.reverse_neighbors.append(alias)

# State observation functions

def observe_turbine_state_yaw(turbine_agent):
    """
    Makes state observation of wind direction and turbine yaw, potentially adding a yaw error.
    """
    #print("ERROR TYPE:", turbine_agent.error_type)
    wind_dir = turbine_agent.model.fi.floris.farm.wind_direction
    yaw_angle = turbine_agent.model.turbine.yaw_angle 
    if turbine_agent.verbose: print("Current yaw angle:", yaw_angle)
    # adjust yaw angle observation based on what type of error is being used
    if turbine_agent.error_type == 0:
        yaw_angle_error = yaw_angle
    elif turbine_agent.error_type == 1:
        yaw_angle_error = yaw_angle / (1 - turbine_agent.yaw_prop)
    elif turbine_agent.error_type == 2:
        yaw_angle_error = yaw_angle + turbine_agent.yaw_offset

    if turbine_agent.verbose: print(turbine_agent.alias, "observes angle as", yaw_angle_error)
    return (wind_dir, yaw_angle_error)

def observe_turbine_state_yaw_abs(turbine_agent):
    """
    Makes state observation of wind direction and turbine yaw, potentially adding a yaw error.
    This method calculates the wind turbine yaw angle as an absolute yaw angle, not relative to the
    wind direction.
    """
    #print("ERROR TYPE:", turbine_agent.error_type)
    wind_dir = turbine_agent.model.fi.floris.farm.wind_direction
    yaw_angle = turbine_agent.model.turbine.yaw_angle - turbine_agent.model.fi.wind_dir_shift 
    if turbine_agent.verbose: print("Current yaw angle:", yaw_angle)
    # adjust yaw angle observation based on what type of error is being used
    if turbine_agent.error_type == 0:
        yaw_angle_error = yaw_angle
    elif turbine_agent.error_type == 1:
        yaw_angle_error = yaw_angle / (1 - turbine_agent.yaw_prop)
    elif turbine_agent.error_type == 2:
        yaw_angle_error = yaw_angle + turbine_agent.yaw_offset

    if turbine_agent.verbose: print(turbine_agent.alias, "observes angle as", yaw_angle_error)
    return (wind_dir, yaw_angle_error)

def observe_turbine_state_sp_dir_yaw(turbine_agent):
    """
    Makes state observation of wind speed, wind direction, and turbine yaw, potentially 
    adding a yaw error.
    """
    wind_speed = turbine_agent.model.fi.floris.farm.wind_speed
    wind_direction = turbine_agent.model.fi.floris.farm.wind_direction
    yaw_angle = turbine_agent.model.turbine.yaw_angle
    #if turbine_agent.verbose: print("Yaw angle: ", yaw_angle, "\nWind Speed: ", wind_speed)

    if turbine_agent.error_type == 0:
        yaw_angle_error = yaw_angle
    elif turbine_agent.error_type == 1:
        yaw_angle_error = yaw_angle / (1 - turbine_agent.yaw_prop)
    elif turbine_agent.error_type == 2:
        yaw_angle_error = yaw_angle + turbine_agent.yaw_offset

    return (wind_speed, wind_direction, yaw_angle_error)

def observe_turbine_state_wind_speed(turbine_agent):
    """
    Makes state observation of wind speed and turbine yaw, potentially adding a yaw error.
    """
    wind_speed = turbine_agent.model.fi.floris.farm.wind_speed
    yaw_angle = turbine_agent.model.turbine.yaw_angle
    #if turbine_agent.verbose: print("Yaw angle: ", yaw_angle, "\nWind Speed: ", wind_speed)

    if turbine_agent.error_type == 0:
        yaw_angle_error = yaw_angle
    elif turbine_agent.error_type == 1:
        yaw_angle_error = yaw_angle / (1 - turbine_agent.yaw_prop)
    elif turbine_agent.error_type == 2:
        yaw_angle_error = yaw_angle + turbine_agent.yaw_offset

    return (wind_speed, yaw_angle_error)

def observe_turbine_state_wind_direction(turbine_agent):
    """
    Makes state observation of wind direction and turbine yaw, potentially adding a yaw error.
    """
    wind_direction = turbine_agent.model.fi.floris.farm.wind_direction
    yaw_angle = turbine_agent.model.turbine.yaw_angle
    #if turbine_agent.verbose: print("Yaw angle: ", yaw_angle, "\nWind Speed: ", wind_speed)

    if turbine_agent.error_type == 0:
        yaw_angle_error = yaw_angle
    elif turbine_agent.error_type == 1:
        yaw_angle_error = yaw_angle / (1 - turbine_agent.yaw_prop)
    elif turbine_agent.error_type == 2:
        yaw_angle_error = yaw_angle + turbine_agent.yaw_offset

    return (wind_direction, yaw_angle_error)

def observe_turbine_state_continuous(turbine_agent):
    """
    Makes state observation for a continuous state space. NOTE: not well-tested.
    """
    wind_dir = turbine_agent.model.fi.floris.farm.wind_direction

    return (wind_dir,)


# Behavior modification functions

num_actions_yaw = 3

def modify_behavior_yaw(turbine_agent):
    """
    Map action integers to increments or decrements in yaw angle. Assumes a state space of 
    two dimensions, wind speed and yaw.
    """
    # Calculate the smallest allowable yaw increment
    delta_yaw = turbine_agent.discrete_states[1][1] - turbine_agent.discrete_states[1][0] 

    # set "present" value to calculate delta, if needed (for example, in a gradient)
    turbine_agent.control_action_present = turbine_agent.control_action_future

    # action 0 -> decrement yaw angle
    # action 1 -> keep constant yaw angle
    # action 2 -> increment yaw angle
    if turbine_agent.action == 0:
        yaw_angle = turbine_agent.model.turbine.yaw_angle - delta_yaw
    elif turbine_agent.action == 1:
        yaw_angle = turbine_agent.model.turbine.yaw_angle
    elif turbine_agent.action == 2:
        yaw_angle = turbine_agent.model.turbine.yaw_angle + delta_yaw
    else:
        print("Invalid action chosen.")

    # make sure yaw angle does not exceed limits of state space
    if yaw_angle < turbine_agent.discrete_states[1][0]:
        yaw_angle = turbine_agent.discrete_states[1][0]
    elif yaw_angle > turbine_agent.discrete_states[1][-1]:
        yaw_angle = turbine_agent.discrete_states[1][-1]

    # set "future" value to calculate delta, if needed (for example, in a gradient)
    turbine_agent.control_action_future = yaw_angle

    return [yaw_angle]

def modify_behavior_sp_dir_yaw(turbine_agent):
    """
    Map action integers to increments or decrements in yaw angle. Assumes a state space of 
    three dimensions, wind speed, wind direction, and yaw.
    """
    # Calculate the smallest allowable yaw increment
    delta_yaw = turbine_agent.discrete_states[2][1] - turbine_agent.discrete_states[2][0] 

    # set "present" value to calculate delta, if needed (for example, in a gradient)
    turbine_agent.control_action_present = turbine_agent.control_action_future

    # action 0 -> decrement yaw angle
    # action 1 -> keep constant yaw angle
    # action 2 -> increment yaw angle
    if turbine_agent.action == 0:
        yaw_angle = turbine_agent.model.turbine.yaw_angle - delta_yaw
    elif turbine_agent.action == 1:
        yaw_angle = turbine_agent.model.turbine.yaw_angle
    elif turbine_agent.action == 2:
        yaw_angle = turbine_agent.model.turbine.yaw_angle + delta_yaw
    else:
        print("Invalid action chosen.")

    # make sure yaw angle does not exceed limits of state space
    if yaw_angle < turbine_agent.discrete_states[2][0]:
        yaw_angle = turbine_agent.discrete_states[2][0]
    elif yaw_angle > turbine_agent.discrete_states[2][-1]:
        yaw_angle = turbine_agent.discrete_states[2][-1]

    # set "future" value to calculate delta, if needed (for example, in a gradient)
    turbine_agent.control_action_future = yaw_angle

    return [yaw_angle]


num_actions_delay = 3

def modify_behavior_delay(turbine_agent):
    """
    Map action integers to increments or decrements in yaw angle in a wake delayed context. In
    this implementation, the concept of neighborhoods and locking is used to prevent turbines
    from moving during sensitive time frames. Assumes a state space of two dimensions, wind 
    speed, and yaw.
    """
    yaw_angle = turbine_agent.model.turbine.yaw_angle

    if turbine_agent.target_state is None:
        # if target_state is None, then the turbine is not currently ramping
        delta_yaw = turbine_agent.discrete_states[1][1] - turbine_agent.discrete_states[1][0]
    else:
        # use the turbine's yaw rate as delta_yaw if the turbine is ramping
        delta_yaw = turbine_agent.model.turbine.yaw_rate

    # set "present" value to calculate delta, if needed (for example, in a gradient)
    turbine_agent.control_action_present = turbine_agent.control_action_future

    if turbine_agent.target_state is None:
        #print("ERROR TYPE:", turbine_agent.error_type)
        # Calculate the smallest allowable yaw increment
         

        # action 0 -> decrement yaw angle
        # action 1 -> keep constant yaw angle
        # action 2 -> increment yaw angle
        
        yaw_angle = yaw_angle + (turbine_agent.action % 3 - 1) * delta_yaw
        #print(turbine_agent.alias, "sets angle to", yaw_angle)
        if turbine_agent.verbose: print(turbine_agent.alias, "chose action", turbine_agent.action)

        # make sure yaw angle does not exceed limits of state space
        if yaw_angle < turbine_agent.discrete_states[1][0]:
            yaw_angle = turbine_agent.discrete_states[1][0]
        elif yaw_angle > turbine_agent.discrete_states[1][-1]:
            yaw_angle = turbine_agent.discrete_states[1][-1]

        # adjust yaw angle observation based on what type of error is being used
        if turbine_agent.error_type == 0:
            yaw_angle_error = yaw_angle
        elif turbine_agent.error_type == 1:
            yaw_angle_error = yaw_angle * (1 - turbine_agent.yaw_prop)
        elif turbine_agent.error_type == 2:
            yaw_angle_error = yaw_angle - turbine_agent.yaw_offset

        # calculate delay for all downstream turbines within the neighborhood
        if not turbine_agent.action == 1:
            _set_delay_map(turbine_agent)
        else:
            turbine_agent.delay_map = {}

        # NOTE: right now, since yaw angle is read straight from the turbine, there is no need to adjust
        # the yaw_angle_error, since the value that was read originally was not errored
        yaw_angle_error = yaw_angle

        # set "future" value to calculate delta, if needed (for example, in a gradient)
        turbine_agent.control_action_future = yaw_angle_error

        return [yaw_angle_error]
        """ phi = turbine_agent.model.fi.floris.farm.wind_direction + 180
        phi *= math.pi/180
        self_adjusted_position = _adjust_coordinates(turbine_agent.position, phi)
        delay_map = {}
        # there is no delay needed if a turbine does not change yaw angle
        if not turbine_agent.action == 1:
            # set the delay of all downstream turbines based on the action that a turbine just selected
            for alias in turbine_agent.farm_turbines:
                if not alias == turbine_agent.alias and alias in turbine_agent.neighbors:
                    adjusted_position = _adjust_coordinates(turbine_agent.farm_turbines[alias], phi)
                    x = self_adjusted_position[0] - adjusted_position[0]
                    U_inf = turbine_agent.model.fi.floris.farm.wind_speed
                    tau = x / U_inf
                    delay_map[alias] = round(tau)
            if len(delay_map) > 0:
                self_delay = max([delay_map[alias] for alias in delay_map])
                delay_map[turbine_agent.alias] = self_delay
            turbine_agent.delay_map = delay_map """
            #print(turbine_agent.alias, "takes action", turbine_agent.action, "and sets yaw angle to", yaw_angle)
    else:
        turbine_agent.delay_map = {}
        diff = turbine_agent.target_state[0] - yaw_angle
        yaw_rate = turbine_agent.model.turbine.yaw_rate
        if abs(diff) < yaw_rate:
            yaw_angle = turbine_agent.target_state[0]
            turbine_agent.target_state = None
            # NOTE: test
            #turbine_agent.delay_map = {}
        else:
            yaw_angle += np.sign(diff) * turbine_agent.model.turbine.yaw_rate
            _set_delay_map(turbine_agent)
            
        #print(turbine_agent.alias, "yaw angle:", yaw_angle)
        return [yaw_angle]

def modify_behavior_sp_dir_yaw_delay(turbine_agent):
    """
    Map action integers to increments or decrements in yaw angle in a wake delayed context. In
    this implementation, the concept of neighborhoods and locking is used to prevent turbines
    from moving during sensitive time frames. Assumes a state space of two dimensions, wind 
    speed, and yaw.
    """
    yaw_angle = turbine_agent.model.turbine.yaw_angle
    delta_yaw = turbine_agent.discrete_states[2][1] - turbine_agent.discrete_states[2][0]
    if turbine_agent.target_state is None:
        #print("ERROR TYPE:", turbine_agent.error_type)
        # Calculate the smallest allowable yaw increment
         

        # action 0 -> decrement yaw angle
        # action 1 -> keep constant yaw angle
        # action 2 -> increment yaw angle
        
        yaw_angle = yaw_angle + (turbine_agent.action % 3 - 1) * delta_yaw
        #print(turbine_agent.alias, "sets angle to", yaw_angle)
        if turbine_agent.verbose: print(turbine_agent.alias, "chose action", turbine_agent.action)

        # make sure yaw angle does not exceed limits of state space
        if yaw_angle < turbine_agent.discrete_states[2][0]:
            yaw_angle = turbine_agent.discrete_states[2][0]
        elif yaw_angle > turbine_agent.discrete_states[2][-1]:
            yaw_angle = turbine_agent.discrete_states[2][-1]

        # adjust yaw angle observation based on what type of error is being used
        if turbine_agent.error_type == 0:
            yaw_angle_error = yaw_angle
        elif turbine_agent.error_type == 1:
            yaw_angle_error = yaw_angle * (1 - turbine_agent.yaw_prop)
        elif turbine_agent.error_type == 2:
            yaw_angle_error = yaw_angle - turbine_agent.yaw_offset

        # calculate delay for all downstream turbines within the neighborhood
        if not turbine_agent.action == 1:
            _set_delay_map(turbine_agent)
        else:
            turbine_agent.delay_map = {}

        yaw_angle_error = yaw_angle

        return [yaw_angle_error]
    else:
        turbine_agent.delay_map = {}
        diff = turbine_agent.target_state[0] - yaw_angle
        yaw_rate = turbine_agent.model.turbine.yaw_rate
        if abs(diff) < yaw_rate:
            yaw_angle = turbine_agent.target_state[0]
            turbine_agent.target_state = None
        else:
            yaw_angle += np.sign(diff) * turbine_agent.model.turbine.yaw_rate
        print(turbine_agent.alias, "yaw angle:", yaw_angle)
        return [yaw_angle]

num_actions_continuous = 6

def modify_behavior_continuous(turbine_agent):
    """
    Map action integers to increments or decrements in yaw angle. Assumes a state space of 
    two dimensions, wind speed, and yaw, but assumes state variables are continuous as opposed
    to other implementations which assume discrete state spaces. NOTE: not well-tested.
    """
    yaw_angle = turbine_agent.action * 5

    if not turbine_agent.action == 0:
        _set_delay_map(turbine_agent)
    else:
        turbine_agent.delay_map = {}

    if turbine_agent.verbose: print(turbine_agent.alias, "chose action", turbine_agent.action)

    return yaw_angle

# Value functions

def _power_opt(turbine_agent):
    """
    Returns wind turbine power for use in optimization.
    """
    return turbine_agent.model.turbine.power# / (3 - len(turbine_agent.neighbors))

def value_function_power_opt(turbine_agent, server, time=None):
    total_value = 0

    #if self.alias == "turbine_1":
    #    scale_factor = 10
    #else:
    #    scale_factor = 1

    for alias in turbine_agent.neighbors:
        total_value += server.read_channel(alias)
    # for alias in self.reverse_neighbors:
    #     total_value += server.read_channel(alias)
    # TODO figure out if it should be server.read_channel(self.alias) or self.evaluate_value_function
    total_value += _power_opt(turbine_agent)

    if time == 0:
        turbine_agent.total_value_present = total_value
    elif time == 1:
        turbine_agent.total_value_future = total_value

    # NOTE: remove this line, it is to see what convergence is like when the turbines have unlimited comm.
    #total_value = sum([turbine.power for turbine in self.model.fi.floris.farm.turbines])
    return total_value

def value_function_baseline(turbine_agent, server, time=None):
    total_value = 0

    #if self.alias == "turbine_1":
    #    scale_factor = 10
    #else:
    #    scale_factor = 1

    for alias in turbine_agent.neighbors:
        total_value += server.read_channel(alias)
    # for alias in self.reverse_neighbors:
    #     total_value += server.read_channel(alias)
    # TODO figure out if it should be server.read_channel(self.alias) or self.evaluate_value_function
    total_value += _power_opt(turbine_agent)

    if time == 0:
        turbine_agent.total_value_present = total_value
    elif time == 1:
        turbine_agent.total_value_future = total_value

    # NOTE: remove this line, it is to see what convergence is like when the turbines have unlimited comm.
    #total_value = sum([turbine.power for turbine in self.model.fi.floris.farm.turbines])
    return total_value - turbine_agent.value_baseline

def _power_normalized(turbine_agent):
    """
    """
    return turbine_agent.model.turbine.power / (3 - len(turbine_agent.neighbors))

def value_function_normalized(turbine_agent, server, time=None):
    total_value = 0

    """ if self.alias == "turbine_1":
        scale_factor = 10
    else:
        scale_factor = 1 """

    for alias in turbine_agent.neighbors:
        total_value += server.read_channel(alias)
    # for alias in self.reverse_neighbors:
    #     total_value += server.read_channel(alias)
    # TODO figure out if it should be server.read_channel(self.alias) or self.evaluate_value_function
    total_value += _power_normalized(turbine_agent)

    if time == 0:
        turbine_agent.total_value_present = total_value
    elif time == 1:
        turbine_agent.total_value_future = total_value

    # NOTE: remove this line, it is to see what convergence is like when the turbines have unlimited comm.
    #total_value = sum([turbine.power for turbine in self.model.fi.floris.farm.turbines])
    return total_value

# Need to keep these here as well to maintain correct dependencies in some scripts.

iterate_floris = iterate.iterate_floris
iterate_floris_delay = iterate.iterate_floris_delay