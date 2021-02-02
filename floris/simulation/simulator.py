import numpy as np
import copy
import itertools
import statistics
import floris.tools.train_run as tr

class Simulator():
    """
    Class that runs a quasi-dynamic simulation using pre-trained LUTs.

    Args:
        fi: FlorisUtilities object
        lut_dict: Dictionary mapping an integer to a LUT object. This integer should correspond to the Turbine.number parameter set in TurbineMap, and associates a Turbine within FlorisUtilities to a LUT. 

    Returns:
        Simulator: An instantiated Simulator object.
    """
    def __init__(self, fi, lut_dict):

        self.fi = fi
        self.lut_dict = lut_dict

        # NOTE: this assumes that all LUTs have the same discrete state space as the zeroth element in the dictionary
        self.discrete_states = self.lut_dict[0].discrete_states

        self.bin_counts = [ [] for _ in range(len(self.discrete_states)) ]

        # how many seconds should elapse before yawing
        self.filter_window = 60
        self.filter_count = 0

        # Boolean that determines if any turbines are currently yawing and, as a result, if the filtered observation process should be taking place
        self.yawing = False

        self.yaw_rate = 0.3 #deg/sec

        # list of yaw angle setpoints
        self.setpoints = []

    def _accumulate(self, state_observations, indices=None):
        """
        Accumulate observations to determine filtered yaw setpoint.

        Args:
            state_observations: Tuple of relevant wind field measurements.
            indices: List of integers corresponding to which indices in the discrete state vector each state measurement corresponds to. If None, will assume that observations are given in the same order as the discrete state space.
        """
        if indices is None:
            indices = [num for num in range(len(state_observations))]

        for i,state_observation in enumerate(state_observations):
            bin_num = np.abs(self.discrete_states[indices[i]] - state_observation).argmin()

            self.bin_counts[indices[i]].append(bin_num)

        self.filter_count += 1 

        if self.filter_count == self.filter_window:
            mode_measurements = []
            for i in range(len(state_observations)):
                # if the set length is not equal to the list length, there are duplicate modes, will just use the previous wind speed
                try:
                    mode_bin_num = statistics.mode(self.bin_counts[i])
                    self.bin_counts[i].clear()

                    mode_measurements.append(self.discrete_states[i][mode_bin_num])
                except:
                    for i in range(len(state_observations)):
                        self.bin_counts[i].clear()
                    return None
            
            # reset filter counter
            self.filter_count = 0

            return tuple(mode_measurements)

        else:
            return None


    def simulate(self, wind_profiles, mean_wind_speeds, learn=False, yaw_error=None, blur=False, sigma=1, method="smallest_diff", func=None):
        """
        Run a simulation with a given wind profile.

        Args:
            wind_profiles: Simulation wind profiles, The expected format is [wind_speed_profile, wind_direction_profile]. A valid profile is a dictionary with the key being the iteration the change occurs at and the value being the value that should be changed to.

            learn: A boolean specifiying whether, if possible, the agents should continue to learn during the course of the simulation. NOTE: not currently implemented.
        """
        self.fi.reinitialize_flow_field(wind_speed=8, wind_direction=270)
        #self.fi.calculate_wake([0 for turbine in self.fi.floris.farm.turbines])
        # reset buffers
        for turbine in self.fi.floris.farm.turbines:
            turbine.wind_field_buffer.reset()

        if learn: 
            turbine_agents = []
            for turbine in self.fi.floris.farm.turbines:
                agent = self.lut_dict[turbine.number].agent
                #agent.filtered = True
                agent.average_power = True
                agent.reset_sim_context(self.fi)
                agent.model.fi = self.fi
                agent.model.turbine = turbine
                agent.sim_context.make_states_noisy()
                agent.sim_context.clear_bin_counts()
                agent.configure_dynamic()
                state_map = {"wind_speed": 8, "wind_direction": 0}
                agent.utilize_q_table(state_name="yaw_angle", state_map=state_map, print_q_table=True)
    
                if yaw_error is not None:
                    state = agent.sim_context.return_state("yaw_angle")
                    state.error_type = "offset"
                    state.error_value = yaw_error
                turbine_agents.append(agent)

            # NOTE: this assumes every LUT has the same server
            server = self.lut_dict[0].server

            wind_speed_profile = wind_profiles[0]
            wind_direction_profile = wind_profiles[1]

            action_selection = "boltzmann"

            reward_signal = "constant"

            for agent in turbine_agents:
                agent.func = func
            [powers, turbine_yaw_angles, turbine_error_yaw_angles, turbine_values, rewards] = \
            tr.run_farm(self.fi, turbine_agents, server, wind_speed_profile, mean_wind_speeds, wind_direction_profile=wind_direction_profile, action_selection=action_selection, reward_signal=reward_signal)

            return (powers, powers, turbine_yaw_angles)

        self.fi.reinitialize_flow_field(wind_speed=8, wind_direction=270)
        self.fi.floris.farm.flow_field.mean_wind_speed = mean_wind_speeds[0]
        self.fi.calculate_wake()
        wind_speed_profile = wind_profiles[0]

        wind_dir_profile = wind_profiles[1]

        powers = []
        true_powers = []
        turbine_yaw_angles = [ [] for turbine in self.fi.floris.farm.turbines]

        self.setpoints.clear()
        for j,turbine in enumerate(self.fi.floris.farm.turbines):
            setpoint = self.lut_dict[turbine.number].read((self.fi.floris.farm.flow_field.mean_wind_speed, wind_dir_profile[0]), all_states=False, blur=blur, sigma=sigma, method=method, func=func)
            self.setpoints.append(setpoint-yaw_error)

        for i in itertools.count():
            if i == max(wind_speed_profile.keys()) or i == max(wind_dir_profile.keys()):
                return (true_powers, powers, turbine_yaw_angles)

            if i in mean_wind_speeds:
                self.fi.floris.farm.flow_field.mean_wind_speed = mean_wind_speeds[i]

            if i in wind_speed_profile:
                self.fi.reinitialize_flow_field(wind_speed=wind_speed_profile[i], sim_time=i)

            if i in wind_dir_profile:
                self.fi.reinitialize_flow_field(wind_direction=wind_dir_profile[i]+270, sim_time=i)

            state = (self.fi.floris.farm.wind_speed[0], self.fi.floris.farm.wind_direction[0])


            mode_measurement = None
            if not self.yawing:
                mode_measurement = self._accumulate(state)

            current_yaw_angles = [turbine.yaw_angle for turbine in self.fi.floris.farm.turbines]

            if mode_measurement is not None:
                self.setpoints = []

                for j,turbine in enumerate(self.fi.floris.farm.turbines):
                    setpoint = self.lut_dict[turbine.number].read(mode_measurement, all_states=False, blur=blur, sigma=sigma, method=method, func=func)
                    self.setpoints.append(setpoint-yaw_error)

            self.yawing = False
            yaw_angles = [None for _ in self.fi.floris.farm.turbines]
            if len(self.setpoints) > 0:

                for j,(yaw_angle,setpoint) in enumerate(zip(current_yaw_angles,self.setpoints)):
                    if abs(setpoint - yaw_angle) < self.yaw_rate:
                        yaw_angles[j] = setpoint
                    else:
                        yaw_angles[j] = yaw_angle + np.sign(setpoint-yaw_angle)*self.yaw_rate
                        self.yawing = True

            self.fi.calculate_wake(sim_time=i, yaw_angles=yaw_angles)

            power = sum([turbine.power for turbine in self.fi.floris.farm.turbines])

            powers.append(power)

            yaw_angles = [turbine.yaw_angle for turbine in self.fi.floris.farm.turbines]

            for j, yaw_angle in enumerate(yaw_angles):
                turbine_yaw_angles[j].append(yaw_angle)

            #self.fi.calculate_wake(yaw_angles=yaw_angles)

            power = sum([turbine.power for turbine in self.fi.floris.farm.turbines])

            true_powers.append(power)