import numpy as np
import copy
import itertools
import statistics

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
        self.lut_dict = copy.deepcopy(lut_dict)

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
        #print(state_observations)
        if indices is None:
            indices = [num for num in range(len(state_observations))]

        for i,state_observation in enumerate(state_observations):
            bin_num = np.abs(self.discrete_states[indices[i]] - state_observation).argmin()

            self.bin_counts[indices[i]].append(bin_num)

        self.filter_count += 1 

        if self.filter_count == self.filter_window:
            mode_measurements = []
            for i in range(len(state_observations)):
                mode_bin_num = statistics.mode(self.bin_counts[i])
                self.bin_counts[i].clear()

                mode_measurements.append(self.discrete_states[i][mode_bin_num])
            
            # reset filter counter
            self.filter_count = 0

            return tuple(mode_measurements)

        else:
            return None


    def simulate(self, wind_profiles, learn=False):
        """
        Run a simulation with a given wind profile.

        Args:
            wind_profiels: Simulation wind profiles, The expected format is [wind_speed_profile, wind_direction_profile]. A valid profile is a dictionary with the key being the iteration the change occurs at and the value being the value that should be changed to.

            learn: A boolean specifiying whether, if possible, the agents should continue to learn during the course of the simulation. NOTE: not currently implemented.
        """
        if learn:
            raise NotImplementedError()

        self.fi.reinitialize_flow_field(wind_speed=8, wind_direction=270)
        self.fi.calculate_wake()
        wind_speed_profile = wind_profiles[0]

        wind_dir_profile = wind_profiles[1]

        powers = []
        true_powers = []
        turbine_yaw_angles = [ [] for turbine in self.fi.floris.farm.turbines]

        for i in itertools.count():

            if i == max(wind_speed_profile.keys()) or i == max(wind_dir_profile.keys()):
                return (true_powers, powers, turbine_yaw_angles)

            if i in wind_speed_profile:
                self.fi.reinitialize_flow_field(wind_speed=wind_speed_profile[i], sim_time=i)

            if i in wind_dir_profile:
                self.fi.reinitialize_flow_field(wind_direction=wind_dir_profile[i], sim_time=i)

            state = (self.fi.floris.farm.wind_speed[0], self.fi.floris.farm.wind_direction[0])


            mode_measurement = None
            if not self.yawing:
                mode_measurement = self._accumulate(state)

            current_yaw_angles = [turbine.yaw_angle for turbine in self.fi.floris.farm.turbines]

            if mode_measurement is not None:
                self.setpoints = []

                for turbine in self.fi.floris.farm.turbines:
                    setpoint = self.lut_dict[turbine.number].read(mode_measurement, all_states=False)
                    self.setpoints.append(setpoint)

            self.yawing = False
            yaw_angles = [None for _ in self.fi.floris.farm.turbines]
            if len(self.setpoints) > 0:

                for i,(yaw_angle,setpoint) in enumerate(zip(current_yaw_angles,self.setpoints)):
                    if abs(setpoint - yaw_angle) < self.yaw_rate:
                        yaw_angles[i] = setpoint
                    else:
                        yaw_angles[i] = yaw_angle + np.sign(setpoint-yaw_angle)*self.yaw_rate
                        self.yawing = True

            self.fi.calculate_wake(yaw_angles=yaw_angles, sim_time=i)

            power = sum([turbine.power for turbine in self.fi.floris.farm.turbines])

            powers.append(power)

            yaw_angles = [turbine.yaw_angle for turbine in self.fi.floris.farm.turbines]

            for i, yaw_angle in enumerate(yaw_angles):
                turbine_yaw_angles[i].append(yaw_angle)

            self.fi.calculate_wake(yaw_angles=yaw_angles)

            power = sum([turbine.power for turbine in self.fi.floris.farm.turbines])

            true_powers.append(power)