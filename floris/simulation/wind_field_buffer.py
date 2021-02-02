import numpy as np
import bisect

class WindFieldBuffer():
    """
    This class encapsulates all methods needed to buffer the necessary inputs for the quasi-dynamic FLORIS approximation.

    Args
    combination_function: The function that should be used to
        combine wakes. If None, the visualization approximation
        will be used.
    num_turbines (int): The number of turbines in the wind fam.
    number (int): Identifying number for the buffer.
    x (list): List of x coordinates
    y (list): List of y coordinates
    z (list): List of z coordinates
    """
    def __init__(self, combination_function=None, num_turbines=None, number=None, x=None, y=None, z=None):
        self.number = number

        if x is not None and y is not None and z is not None:
            # dims is interpreted as (num_x, num_y, num_z) with num_x, num_y, and num_z representing the number of points at which wind speed is desired to be measured at
            dims = np.shape(x)

            self.x = x
            self.y = y
            self.z = z

            self._future_wind_field_speeds = [ [ [ [] for el1 in range(dims[2])] for el2 in range(dims[1]) ] for el3 in range(dims[0]) ]

            self._current_wind_field_speeds = [ [ [ None for el1 in range(dims[2])] for el2 in range(dims[1]) ] for el3 in range(dims[0]) ]

        if combination_function is not None:
            # NOTE: this condition is used to tell if the buffer is for computations or visualizations
            self._future_wind_speeds = []
            self._current_wind_speed = None

            self._future_wind_dirs = []
            self._current_wind_dir = None
            self._current_coord = None

            self._future_wake_deficits = [ [] for _ in range(num_turbines) ]
            self._current_wake_deficits = [ None for _ in range(num_turbines) ]

            self._combination_function = combination_function

    def _add_to_wind_field_buffer(self, wind_speed, indices, delay, sim_time):
        """
        Adds wind speed to visualization buffer.

        Args
        wind_speed (double): New wind speed to add
        indices (tuple): Index into x, y, and z coordinate lists
        delay (int): Delay in seconds between current time and
            when wind speed will go into effect.
        sim_time (int): Current simulation time.
        """

        if delay < 0:
            return
        i = indices[0]
        j = indices[1]
        k = indices[2]
        old_wind_speed = self._current_wind_field_speeds[i][j][k]
        test_tuple = (wind_speed, delay+sim_time)
        #print("Delay:", delay)
        # for _ in range(delay):
        #     self._future_wind_field_speeds[i][j][k].append(old_wind_speed)

        delayed_time = delay + sim_time

        slice_index = bisect.bisect_left([wind_field[1] for wind_field in self._future_wind_field_speeds[i][j][k]], delayed_time)

        self._future_wind_field_speeds[i][j][k].insert(slice_index, (wind_speed, delayed_time))

        self._future_wind_field_speeds[i][j][k] = self._future_wind_field_speeds[i][j][k][:slice_index+1]

        #self._future_wind_field_speeds[i][j][k].append(wind_speed)
        #self._future_wind_field_speeds[i][j][k].append(test_tuple)
        #print(self._future_wind_field_speeds[i][j][k])

    def add_wind_field(self, new_wind_field, propagate_wind_speed, sim_time, first_x=None):
        """
        Buffers in a new wind field to the visualization buffer.

        Args
        new_wind_field: The new wind field.
        propagate_wind_speed (double): Speed at which changes should
            propagate (ie mean wind speed).
        sim_time (int): Current simulation time.
        first_x: x coordinate of the first turbine. If None, this 
            will be assumed to be the smallest x coordinate in the 
            wind field. This can be used to have propagation 
            originate from different points in the wind field, such 
            as individual turbines.

        """
        # add a new wind field to the buffer
        if first_x is None:
            first_x = np.min(self.x)#self.x[0][0][0]
        print("propagate_wind_speed:", propagate_wind_speed)
        for i in range(len(new_wind_field)):
            for j in range(len(new_wind_field[i])):
                for k in range(len(new_wind_field[i][j])):

                    new_wind_speed = new_wind_field[i][j][k]

                    if propagate_wind_speed is None:
                        self._current_wind_field_speeds[i][j][k] = new_wind_speed
                    else:
                        diff_x = self.x[i][j][k] - first_x
                        delay = round(diff_x / propagate_wind_speed)
                        delay = int(delay)
                        self._add_to_wind_field_buffer(new_wind_speed, (i,j,k), delay, sim_time)
        return np.min(self.x)

    def get_wind_field(self, sim_time):
        """
        Return the current, dynamic state of the wind field for the
        visualization approximation.

        Args
        sim_time (int): Current simulation time.
        """
        for i in range(len(self._current_wind_field_speeds)):
            for j in range(len(self._current_wind_field_speeds[i])):
                for k in range(len(self._current_wind_field_speeds[i][j])):
                    # if len(self._future_wind_field_speeds[i][j][k]) > 0:
                    #     self._current_wind_field_speeds[i][j][k] = self._future_wind_field_speeds[i][j][k][0][1]

                    #     self._future_wind_field_speeds[i][j][k].pop(0)
                    #     #print(temp)
                    #     #print(self._future_wind_field_speeds[i][j][k])
                    if len(self._future_wind_field_speeds[i][j][k]) > 0 and self._future_wind_field_speeds[i][j][k][0][1] == sim_time:
                        self._current_wind_field_speeds[i][j][k] = self._future_wind_field_speeds[i][j][k][0][0]

                        self._future_wind_field_speeds[i][j][k].pop(0)


        return np.array(self._current_wind_field_speeds)

    def add_wind_direction(self, new_wind_direction, sim_time, old_wind_direction=None, old_coord=None):
        """
        Add wind direction into the computational buffer.

        Args
        new_wind_direction (double): Wind direction to add to the 
            buffer.
        sim_time (int): Current simulation time.
        old_wind_direction (double): The previous wind direction
            NOTE: I don't think this input is necessary.
        old_coord: Previous turbine coordinate.
            NOTE: This input is not necessary, and was used in 
            a previous iteration of the code.

        """
        self._future_wind_dirs.append((new_wind_direction, sim_time))

    def add_wind_speed(self, new_wind_speed, delayed_time):
        """
        This method is intended to add a new wind speed to the buffer.

        Args:
            new_wind_speed (float): Wind speed that should be added 
                to the buffer.
            delayed_time: Simulation time that the wind speed should
                go into effect at (int).
        """
        # NOTE: I think bisect_left is the correct choice, maybe bisect_right
        slice_index = bisect.bisect_left([wind_speed[1] for wind_speed in self._future_wind_speeds], delayed_time)

        self._future_wind_speeds.insert(slice_index, (new_wind_speed, delayed_time))

        self._future_wind_speeds = self._future_wind_speeds[:slice_index+1]
        #print("_future_wind_speeds for", self.number, ":")
        #print(self._future_wind_speeds)

        #print("wind_speed added at", delayed_time)

        #self._future_wind_speeds.append((new_wind_speed, sim_time))

    def initialize_wind_speed(self, old_wind_speed, new_wind_speed, overwrite=False):
        """
        This method is intended to set the initial wind speed if it is not already set.

        Args:
            old_wind_speed (float): Wind speed that non-overwritten
                turbines should be set to if current wind speed 
                is None.
            new_wind_speed (float): Wind speed that overwritten 
                turbine should be set to.

            overwrite (bool): Whether or not the current wind speed
                should be overwritten.
        """
        if self._current_wind_speed is None and not overwrite:

            self._current_wind_speed = old_wind_speed
        elif overwrite:

            self._current_wind_speed = new_wind_speed

        return

    def initialize_wind_direction(self, old_wind_direction, new_wind_direction, coord, overwrite=False):
        """
        This method is intended to set the initial wind direction if it is not already set.
        NOTE: This method is currently not implemented correctly.

        Args:
            wind_direction: Wind direction in degrees relative to 270 to be initialized (float).

            coord: Turbine coordinate relative to wind direction.

            overwrite: Whether or not the current wind direction should be overwritten (boolean).
        """

        if self._current_wind_dir is None and not overwrite:
            self._current_wind_dir = old_wind_direction
            self._current_coord = coord
            self._wind_dir_shift = 0

        elif overwrite:

            if self._current_wind_dir is not None:
                self._wind_dir_shift = new_wind_direction - self._current_wind_dir

            else:
                self._wind_dir_shift = new_wind_direction - old_wind_direction

            self._current_wind_dir = new_wind_direction
            self._current_coord = coord

        return

    def get_wind_direction(self, wind_direction, coord, send_wake, sim_time):
        """
        Method to determine what wind direction the flow field should be set to.

        Args:
            wind_direction (float): Wind direction in degrees
                relative to 270 that the farm should be set to if
                there are no wind directions stored in the internal
                buffer (float).
            coord: Turbine coord that should be set if there are no
                wind directions stored in the internal buffer.
            send_wake (bool): Variable specifying whether or not the
                turbine currently should propagate its wake downstream.
                NOTE: I do not think that this variable is necessary any more.
            sim_time (int): Current simulation time (int).

        Returns:
            Tuple of wind direction setpoint, coordinate setpoint, and send_wake, a boolean that signifies whether or not a turbine needs to propagate its wake downstream.
        """

        if len(self._future_wind_dirs) > 0 and self._future_wind_dirs[0][1] == sim_time:
            send_wake_temp = False#True
            pop_wind_dir = self._future_wind_dirs[0][0]

            if self._current_wind_dir is not None: 
                self._wind_dir_shift = pop_wind_dir - self._current_wind_dir

            self._current_wind_dir = pop_wind_dir

            self._future_wind_dirs.pop(0)
        else:
            send_wake_temp = False

        if self._current_wind_dir is not None:
            wind_direction_set = self._current_wind_dir
        else:
            wind_direction_set = wind_direction
            self._wind_dir_shift = 0


        return (wind_direction_set, send_wake or send_wake_temp)

    def get_wind_speed(self, wind_speed, send_wake, sim_time):
        """
        Method to determine what wind speed the flow field should be set to.

        Args:
            wind_speed (float): Wind speed that the farm should be
                set to if there are no wind speeds stored in the
                internal buffer.
            send_wake (bool): Variable specifying whether or not the
                turbine currently should propagate its wake
                downstream. NOTE: I do not think that this variable
                is necessary any more.
            sim_time (int): Current simulation time.

        Returns:
            Tuple of wind direction setpoint, coordinate setpoint, and send_wake, a boolean that signifies whether or not a turbine needs to propagate its wake downstream.
        """

        if len(self._future_wind_speeds) > 0 and self._future_wind_speeds[0][1] == sim_time:
            send_wake_temp = False#True
            self._current_wind_speed = self._future_wind_speeds[0][0]
            self._future_wind_speeds.pop(0)
        else:
            send_wake_temp = False
        
        if self._current_wind_speed is not None:
            wind_speed_set = self._current_wind_speed
        else:
            wind_speed_set = wind_speed

        return (wind_speed_set, send_wake or send_wake_temp)

    @property
    def wind_speed(self):
        return self._current_wind_speed

    @property
    def wind_direction(self):
        return self._current_wind_dir

    def _search_and_combine_u_wakes(self, wake_deficit, wake_dims, index, sim_time):
        """
        Finds all wake deficits at a given simulation time in the buffer and averages them.

        Args:
            wake_deficit (np.array): Wake deficit that should be used
                if there are no wake deficits in the buffer.
            wake_dims: Dimensions of the wake deficit matrix (tuple).
            index (int): What index of the wake deficit buffer to
                look at.
            sim_time (int): The current simulation time.
        """
        current_effects = []
 
        for wake_effect in self._future_wake_deficits[index]:
            if wake_effect[1] == sim_time:
                current_effects.append(wake_effect[0])

        if len(current_effects) == 0:
            return wake_deficit
        else:
            if index == self.number: print("loading wake deficit for self")
            return np.mean(current_effects, axis=0)

    def get_u_wake(self, wake_dims, send_wake, sim_time):
        """
        This method returns the filled u_wake matrix for the buffer.

        Args:
            wake_dims (tuple, ints): Dimensions of the wake deficit
                matrix.
            send_wake (bool): Variable specifying whether or not the
                turbine currently should propagate its wake
                downstream.
            sim_time (int): Current simulation time.
        """

        send_wake_temp = False

        # iterate through the turbine.wake_effects list, which contains a wake_effect entry for every turbine in the farm
        for k in range(len(self._future_wake_deficits)):
            # the third element of the wake_effect entry is the index of the turbine (in the sorted_map) that the wake 
            # corresponds to 
            new_u_wake = self._search_and_combine_u_wakes(self._current_wake_deficits[k], wake_dims, k, sim_time)
            old_u_wake = self._current_wake_deficits[k]

            #send_wake_temp = send_wake_temp or not(new_u_wake is None and old_u_wake is None and (np.array([new_u_wake == old_u_wake])).all())
            #send_wake_temp = send_wake_temp or not (np.array([new_u_wake == old_u_wake])).all()

            self._current_wake_deficits[k] = new_u_wake

        # combine the effects of all the turbines together
        u_wake = np.zeros(wake_dims)
        for i in range(len(self._current_wake_deficits)):
            if self._current_wake_deficits[i] is not None: 
                # the first element of the turb_u_wakes entry is the actual wake deficit contribution from the 
                # turbine at that index
                u_wake = self._combination_function(u_wake, self._current_wake_deficits[i])
        return (u_wake, send_wake or send_wake_temp)

    def initialize_wake_deficit(self, wake_deficit, index):
        """
        This method is intended to set an initial wake deficit after the most upstream turbine has determined its
        wake deficit.

        Args:
            wake_deficit: The wake deficit that the agent should be initialized with (np array)

            index: The index of the wake deficit buffer the wake deficit should be added at (this corresponds to which turbine number caused the wake) (int).
        """

        # for el in self._current_wake_deficits:
        #     if el is None:
        #         self._current_wake_deficits[index] = (wake_deficit, None)
        #print("Initialize wake deficits called.")
        if self._current_wake_deficits[index] is None:
            self._current_wake_deficits[index] = wake_deficit

    def add_wake_deficit(self, new_wake_deficit, index, delayed_time):
        """
        This method is intended to add wake deficit matrices into the buffer. 

        Args:
            new_wake_deficit: The new wake deficit to be added (np array).

            index: The index of the wake deficit buffer the wake deficit should be added at (this corresponds to which turbine number caused the wake) (int).

            delayed_time: The simulation time that the wake should come into effect at (int).
        """
        # NOTE: I think bisect_left is the correct choice, maybe bisect_right
        slice_index = bisect.bisect_left([wake_deficit[1] for wake_deficit in self._future_wake_deficits[index]], delayed_time)

        self._future_wake_deficits[index].insert(slice_index, (new_wake_deficit, delayed_time))

        self._future_wake_deficits[index] = self._future_wake_deficits[index][:slice_index+1]

        # the below line does not include any checking to make sure that there are no earlier-time wake effects already in the buffer
        #self._future_wake_deficits[index].append((new_wake_deficit, sim_time))

    def reset(self):
        """
        Resets all buffers.
        """

        self._future_wind_speeds = []
        self._current_wind_speed = None

        self._future_wind_dirs = []
        self._current_wind_dir = None
        self._current_coord = None

        self._future_wake_deficits = [ [] for _ in range(len(self._future_wake_deficits)) ]
        self._current_wake_deficits = [ None for _ in range(len(self._current_wake_deficits)) ]

    @property
    def wind_dir_shift(self):
        wind_dir_shift = self._wind_dir_shift
        self._wind_dir_shift = 0
        return wind_dir_shift