import floris.tools as wfct
from floris.tools.agent_server_coord import Server, TurbineAgent
import floris.tools.floris_agent as fa
from floris.tools.optimization.scipy.yaw import YawOptimization
import floris.tools.train_run as tr
import numpy as np
import matplotlib.pyplot as plt
import floris.tools.q_learn as q
import copy

class Trainer():
    """
    Trainer is a class that encapsulates the wind farm training process using a variety of different methods.
    It is intended to provide a high-level interface that can take in simulation parameters and produce a LUT
    that could be implemented on a wind farm. For methods that allow for a dynamically updating LUT, it is 
    intended for this class to be able to be passed into a method for a different simulation software to use
    the learned information to make decisions. 

    Args:
        fi: An instantiated FlorisInterface object with the layout defined.

        parameters: A dictionary containing the following (potentially) key-value pairs. If any are not
            supplied, default values will be assigned.

            - **wind_speeds**: A list of wind speeds to determine the LUT for. Default 8 to 20 m/s 
                with 1 m/s spacing.
            - **wind_directions**: A list of wind directions to determine the LUT for. Default 270 deg 
                (one element list)
            - **yaw_angles**: A list of yaw angles that defines the resolution of the state space. Default -30 
                deg to 30 deg with 1 deg spacing. Only the first and last values are used for static LUT methods.
            - **neighborhood_dims**: A two element list of the form [downwind, crosswind] that specifies the
                dimensions of a neighborhood in the downwind and crosswind directions. Default [14D, 1D] with 
                D representing rotor diameter
            - **wind_profiles**: Training wind profiles, for Q-table methods. The expected format is 
                [wind_speed_profile, wind_direction_profile]. A valid profile is a dictionary with the key being 
                the iteration the change occurs at and the value being the value that should be changed to. If not 
                provided, will allocate a fixed simulation iteration to each wind speed/wind direction pair to 
                train Q-table.
            - **training_method**: A string specifying which method should be used to populate the table. 
                Options include:
                - static: creates a LUT based on static optimization of FLORIS, which is not able to change in the 
                field
                - not_static: dummy variable to test Q-table training, TODO: expand to other training methods

        training_method: An instantiated TrainingMethod object.

        dyn_train: A boolean indicating if the Trainer should use the quasi-dynamic environment. If True, the quasi-dynamic environment will be used. If False, the steady-state environment will be used.
    Returns:
        Trainer: An instantiated Trainer object.
    """
    def __init__(self, fi, parameters, training_method, dyn_train=False):
        self.fi = fi
        self.parameters = copy.deepcopy(parameters)
        self.tm = training_method
        self.dyn_train = dyn_train

        if "wind_profiles" not in self.parameters:
            self.parameters["wind_profiles"] = None
        if "wind_speeds" not in self.parameters and self.parameters["wind_profiles"] is None:
            print("Using default wind speed resolution...")
            high = 15
            low = 8
            self.parameters["wind_speeds"] = np.linspace(low, high, high-low+1)
        if "wind_directions" not in self.parameters and self.parameters["wind_profiles"] is None:
            print("Using default wind direction resolution...")
            high = 315
            low = 225
            self.parameters["wind_directions"] = np.array([270])
        if "yaw_angles" not in self.parameters:
            print("Using default yaw angle resolution...")
            high = 45
            low = 0
            step = 0.3
            self.parameters["yaw_angles"] = np.arange(low, high, step)#np.linspace(low, high, 100)#high-low+1)
        if "neighborhood_dims" not in self.parameters:
            print("Using default neighborhood dimensions...")
            D = self.fi.floris.farm.turbines[0].rotor_diameter
            self.parameters["neighborhood_dims"] = [14*D, D]

        if "wind_speeds" in self.parameters:
            self.parameters["wind_speeds"] = np.array(self.parameters["wind_speeds"])
        if "wind_directions" in self.parameters:
            self.parameters["wind_directions"] = np.array(self.parameters["wind_directions"])
        self.parameters["yaw_angles"] = np.array(self.parameters["yaw_angles"])

    def wind_speeds(self):
        return self.parameters["wind_speeds"]

    def wind_directions(self):
        return self.parameters["wind_directions"]

    def train(self, tau=None, epsilon=None, discount=None, value_function=None, file_prefix=None):
        print("Beginning LUT training...")
        if self.tm.static:
            return self._train_static()
        else:
            return self._train_dynamic(tau, epsilon, discount, value_function, file_prefix)

    def _train_static(self):
        discrete_states = [self.parameters["wind_speeds"], self.parameters["wind_directions"], \
            self.parameters["yaw_angles"]]

        turbines = self.fi.floris.farm.flow_field.turbine_map.turbines

        aliases = ["turbine_" + str(i) for i in range(len(turbines))]

        num_wind_speeds = len(self.parameters["wind_speeds"])
        num_wind_directions = len(self.parameters["wind_directions"])

        tables = [np.zeros((num_wind_speeds, num_wind_directions)) for _ in turbines]

        min_yaw = self.parameters["yaw_angles"][0]
        max_yaw = self.parameters["yaw_angles"][-1]
        for i,wind_speed in enumerate(self.parameters["wind_speeds"]):
            for j,wind_direction in enumerate(self.parameters["wind_directions"]):
                self.fi.reinitialize_flow_field(wind_speed=wind_speed, wind_direction=wind_direction)
                # Instantiate the Optimization object
                yaw_opt = YawOptimization(self.fi, minimum_yaw_angle=min_yaw, maximum_yaw_angle=max_yaw)

                # Perform optimization
                yaw_angles = yaw_opt.optimize(verbose=False)

                self._fill_tables(tables, yaw_angles, i, j)                

        luts = []
        for alias,table in zip(aliases,tables):
            lut = LUT(self.tm, discrete_states, table=table, alias=alias)
            luts.append(lut)

        return luts

    def _fill_tables(self, tables, yaw_angles, i, j):
        for yaw_angle,table in zip(yaw_angles, tables):
            table[i][j] = yaw_angle

        return 

    def _train_dynamic(self, tau, epsilon, discount, value_function=None, file_prefix=None):
        # TODO: include value_function as part of training_method, not _train_dynamic
        discrete_states = [self.parameters["wind_speeds"], \
            self.parameters["wind_directions"], \
            self.parameters["yaw_angles"]]

        neighborhood_dims = self.parameters["neighborhood_dims"]

        fi = self.fi

        layout_x = [vec.x1 for vec in fi.floris.farm.flow_field.turbine_map.coords]
        layout_y = [vec.x2 for vec in fi.floris.farm.flow_field.turbine_map.coords]

        turbines = fi.floris.farm.flow_field.turbine_map.turbines

        aliases = ["turbine_" + str(i) for i in range(len(turbines))]

        farm_turbines = {alias: (x,y) for alias,x,y in zip(aliases,layout_x,layout_y)}

        if value_function is None:
            value_function = fa.value_function_power_opt

        wind_speed = discrete_states[0]
        wind_dir = discrete_states[1]
        yaw = discrete_states[2]

        #discrete_states = [discrete_states[0], discrete_states[2]]

        turbine_agents = []
        for index,turbine in enumerate(turbines):
                model = fa.FlorisModel(fi, turbine, index)

                wind_speed_state = fa.State(name="wind_speed", method=model.wind_speed, state_type="discrete", discrete_values=wind_speed, error_type="none", controlled=False)
                wind_direction_state = fa.State(name="wind_direction", method=model.wind_direction, state_type="discrete", discrete_values=wind_dir, error_type="none", controlled=False)
                yaw_angle_state = fa.State(name="yaw_angle", method=model.yaw_angle, state_type="discrete", discrete_values=yaw, error_type="none", controlled=True)

                sim_context = fa.SimContext([wind_speed_state, yaw_angle_state])#wind_direction_state, yaw_angle_state])

                agent = TurbineAgent(aliases[index], "no one of consequence", 
                                    farm_turbines=farm_turbines, 
                                    observe_turbine_state=sim_context.observe_state,#fa.observe_turbine_state_sp_dir_yaw, 
                                    modify_behavior=sim_context.modify_behavior,#fa.modify_behavior_sp_dir_yaw, 
                                    num_actions=fa.num_actions_yaw, 
                                    value_function=value_function, 
                                    find_neighbors=fa.find_neighbors, 
                                    neighborhood_dims=neighborhood_dims,
                                    model=model,
                                    yaw_prop=0.2,
                                    yaw_offset=5,
                                    error_type=0,
                                    tau=tau,
                                    epsilon=epsilon,
                                    discount=discount,
                                    sim_context=sim_context)
                agent.verbose = False
                turbine_agents.append(agent)
        server = Server(turbine_agents)
        
        num_iterations = self.tm.iterations

        if self.parameters["wind_profiles"] is not None:
            wind_profiles = self.parameters["wind_profiles"]
            if wind_profiles[0] is None or wind_profiles[1] is None:
                raise ValueError("Invalid wind speed or wind direction profile.")
            else:
                wind_speed_profile = wind_profiles[0]
                wind_direction_profile = wind_profiles[1]
        else:
            wind_speeds = self.parameters["wind_speeds"]
            wind_speed_profile = tr.create_constant_wind_profile(wind_speeds, num_iterations)

            # TODO: add code to make a valid wind direction profile (will require modifying train_farm)
            wind_direction_profile = tr.create_constant_wind_profile([270], max(wind_speed_profile.keys()))

        action_selection = self.tm.action_selection
        reward_signal = self.tm.reward_signal
        coord = self.tm.coord
        opt_window = self.tm.opt_window

        if not self.dyn_train:
            [powers, turbine_yaw_angles, turbine_error_yaw_angles, turbine_values, rewards, prob_list] = \
            tr.train_farm(fi, turbine_agents, server, wind_speed_profile, wind_direction_profile=wind_direction_profile, action_selection=action_selection, reward_signal=reward_signal,\
                coord=coord, opt_window=opt_window, print_iter=False)
        else:
            [powers, turbine_yaw_angles, turbine_error_yaw_angles, turbine_values, rewards] = \
            tr.run_farm(fi, turbine_agents, server, wind_speed_profile, wind_direction_profile=wind_direction_profile, action_selection=action_selection, reward_signal=reward_signal)
        plt.plot(powers)
        iterations = range(num_iterations)

        if file_prefix is not None:
            yaw_name = file_prefix + "_yaw.npy"
            power_name = file_prefix + "_power.npy"
            iteration_name = file_prefix + "_iteration.npy"

            np.save(yaw_name, turbine_yaw_angles)
            np.save(power_name, powers)
            np.save(iteration_name, iterations)
        #plt.plot(powers)
        #plt.show()
        luts = []
        for alias,agent in zip(aliases,turbine_agents):
            lut = LUT(self.tm, discrete_states, agent=agent)
            luts.append(lut)

        return luts

    def _set_training_method(self, training_method):
        training_methods = ["static"]

        if training_method not in training_methods:
                    raise ValueError("Invalid training method selected")   

class LUT():
    def __init__(self, training_method, discrete_states, table=None, alias=None, agent=None):
        self._tm = training_method
        self.discrete_states = discrete_states

        if self._tm.static:
            self._table = table
            self.alias = alias
        else:
            self.agent = agent
            self._table = self.agent.Q
            self.alias = self.agent.alias

        return

    def read(self, state=None, all_states=True):
        """
        This method is designed to provide a high-level read of the LUT that is abstracted away from the
        algorithm that created it. 

        Args:
            - state: Tuple. Which state, if any, to read for. Should be of the form (wind_speed, wind_dir).
            - all_states: Boolean. If True, read() will return a table that has a yaw angle for every yaw 
                angle or direction in the discrete states, and state will be ignored. If False, will only 
                read the table entry for the given state.
        """
        if not all_states and state is None:
            raise ValueError("State must be specified if all_states is False.")

        if self._tm.static:
            if all_states:
                return self._table
            else:
                # add dummy yaw angle
                state = state + (0,)
                # NOTE: probably don't need this, so probably don't need to import q_learn
                state_indices = q.find_state_indices(self.discrete_states, state)
                return self._table[state_indices[0:-1]]
        else:
            num_wind_speeds = len(self.discrete_states[0])
            num_wind_directions = len(self.discrete_states[1])

            if all_states:
                table = np.zeros((num_wind_speeds, num_wind_directions))

                for i,wind_speed in enumerate(self.discrete_states[0]):
                    for j,wind_direction in enumerate(self.discrete_states[1]):
                        # TODO: change utilize_q_table method so I don't need to put a dummy yaw angle in
                        state = (wind_speed, wind_direction, 0)
                        state_map = {"wind_speed":wind_speed, "wind_direction":wind_direction}
                        # NOTE: probably don't need this, so probably don't need to import q_learn
                        state_indices = q.find_state_indices(self.discrete_states, state)

                        table[i][j] = self.agent.utilize_q_table(state_name="yaw_angle",state_map=state_map)#axis=[0,1], state=state)
                return table
            else:
                # add dummy yaw angle
                state = state + (0,)

                # NOTE: probably don't need this, so probably don't need to import q_learn
                state_indices = q.find_state_indices(self.discrete_states, state)

                state_map = {"wind_speed": state[0], "wind_direction": state[1]}

                return self.agent.utilize_q_table(state_name="yaw_angle", state_map=state_map)
    
    #def _read_q_table(self, wind_speed, wind_direction):


class TrainingMethod():
    """
    TrainingMethod is intended to be a simple interface to provide all the information that is needed
    to customize a training routine.

    Args:
        num_turbines: Int, number of turbines in the wind farm.
        static: Boolean, whether or not a static table should be created.
        coord: String specifying how coordination should be accomplished. If None, no
        coordination will be used, and execution will be simultaneous. Current options are:
            - up_first: Optimize from upstream to downstream
            - down_first: Optimize from downstream to upstream
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
        opt_window: Number of simulation iterations that each turbine or group of turbines is
        given to optimize. Ignored if coord is None or static is True.
        iterations: An integer specifying the number of simulation iterations that the farm is trained 
        for. If None and coord is not None, will be set to opt_window * num_turbines.
        name: String, training method name.

    """
    def __init__(self, num_turbines, static, 
                    coord=None, 
                    action_selection=None, 
                    reward_signal=None, 
                    opt_window=None,
                    iterations=None,
                    name=None):
        self.num_turbines = num_turbines
        self.static = static

        self.coord = coord
        
            
        self.action_selection = action_selection
        self.reward_signal = reward_signal
        self.opt_window = opt_window

        self.name = name

        if self.coord is None:
            self.iterations = iterations
        else:
            if iterations is None:
                self.iterations = self.opt_window * self.num_turbines
            else:
                self.iterations = iterations

        self.validate()

    def validate(self):
        """
        Verifies that current parameters are valid.
        """
        if not self.static:
            # if training method is not static, action_selection and reward_signal must be specified
            if self.action_selection is None or self.reward_signal is None:
                raise ValueError("action_selection and reward_signal must be specified for non-static training method.")
            # if coordination is being used, opt_window must be specified
            if self.coord is not None and self.opt_window is None:
                raise ValueError("opt_window must be specified for coordinated training method.")
            # if coordination is not being used, iterations must be specified
            if self.coord is None and self.iterations is None:
                raise ValueError("iterations must be specified if farm is not coordinated.")

        valid_action_selection = ["boltzmann", "epsilon", "gradient"]
        valid_reward_signal = ["constant", "variable"]
        valid_coord = ["up_first", "down_first"]

        if self.action_selection is not None:
            if self.action_selection not in valid_action_selection:
                raise ValueError("Invalid action selection algorithm.")
        
        if self.reward_signal is not None:
            if self.reward_signal not in valid_reward_signal:
                raise ValueError("Invalid reward signal.")

        if self.coord is not None:
            if self.coord not in valid_coord:
                raise ValueError("Invalid coordination algorithm.")