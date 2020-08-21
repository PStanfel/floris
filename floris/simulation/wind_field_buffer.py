import numpy as np
import bisect

class WindFieldBuffer():
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
            self._future_wind_speeds = []
            self._current_wind_speed = None

            self._future_wind_dirs = []
            self._current_wind_dir = None
            self._current_coord = None

            self._future_wake_deficits = [ [] for _ in range(num_turbines) ]
            self._current_wake_deficits = [ None for _ in range(num_turbines) ]

            self._combination_function = combination_function

    def _add_to_wind_field_buffer(self, wind_speed, indices, delay, sim_time):

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
        # add a new wind field to the buffer
        if first_x is None:
            first_x = np.min(self.x)#self.x[0][0][0]

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

    def add_wind_direction(self, old_wind_direction, new_wind_direction, sim_time, old_coord):
        self._future_wind_dirs.append((old_wind_direction, new_wind_direction, sim_time, old_coord))

    def add_wind_speed(self, new_wind_speed, delayed_time):
        """
        This method is intended to add a new wind speed to the buffer.

        Args:
            new_wind_speed: Wind speed that should be added to the buffer (float).

            delayed_time: Simulation time that the wind speed should go into effect at (int).
        """
        # NOTE: I think bisect_left is the correct choice, maybe bisect_right
        slice_index = bisect.bisect_left([wind_speed[1] for wind_speed in self._future_wind_speeds], delayed_time)

        self._future_wind_speeds.insert(slice_index, (new_wind_speed, delayed_time))

        self._future_wind_speeds = self._future_wind_speeds[:slice_index+1]

        #print("wind_speed added at", delayed_time)

        #self._future_wind_speeds.append((new_wind_speed, sim_time))

    def initialize_wind_speed(self, old_wind_speed, new_wind_speed, overwrite=False):
        """
        This method is intended to set the initial wind speed if it is not already set.

        Args:
            old_wind_speed: Wind speed that non-overwritten turbines should be set to if current wind speed is None (float).

            new_wind_speed: Wind speed that overwritten turbine should be set to (float).

            overwrite: Whether or not the current wind speed should be overwritten (boolean).
        """
        if self._current_wind_speed is None and not overwrite:
            self._current_wind_speed = old_wind_speed
        elif overwrite:
            self._current_wind_speed = new_wind_speed
        #print(self.number, "called initialize_wind_speed, current wind speed is", self._current_wind_speed)
        return

    def initialize_wind_direction(self, wind_direction, coord, overwrite=False):
        """
        This method is intended to set the initial wind direction if it is not already set.
        NOTE: This method is currently not implemented correctly.

        Args:
            wind_direction: Wind direction in degrees relative to 270 to be initialized (float).

            coord: Turbine coordinate relative to wind direction.

            overwrite: Whether or not the current wind direction should be overwritten (boolean).
        """

        if self._current_wind_dir is None:
            self._current_wind_dir = wind_direction
            self._current_coord = coord
            #print("Current wind direction None, setting to", wind_direction)
        elif overwrite:
            self._current_wind_dir = wind_direction
            self._current_coord = coord
            #print("Overwrite is True, setting wind direction to", wind_direction)
        return

    def get_wind_direction(self, wind_direction, coord, send_wake, sim_time):
        """
        Method to determine what wind direction the flow field should be set to.

        Args:
            wind_direction: Wind direction in degrees relative to 270 that the farm should be set to if there are no wind directions stored in the internal buffer (float).

            coord: Turbine coord that should be set if there are no wind directions stored in the internal buffer.

            send_wake: Variable specifying whether or not the turbine currently should propagate its wake downstream.

            sim_time: Current simulation time (int).

        Returns:
            Tuple of wind direction setpoint, coordinate setpoint, and send_wake, a boolean that signifies whether or not a turbine needs to propagate its wake downstream.
        """

        if len(self._future_wind_dirs) > 0 and self._future_wind_dirs[0][1] == sim_time:
            send_wake_temp = True
            self._current_wind_dir = self._future_wind_dirs[0][0]
            self._current_coord = self._future_wind_dirs[0][3]
            self._future_wind_dirs.pop(0)
        else:
            send_wake_temp = False

        if self._current_wind_dir is not None and self._current_coord is not None:
            wind_direction_set = self._current_wind_dir
            coord_set = self._current_coord
        else:
            wind_direction_set = wind_direction
            coord_set = coord

        return (wind_direction_set, coord_set, send_wake or send_wake_temp)

    def get_wind_speed(self, wind_speed, send_wake, sim_time):
        """
        Method to determine what wind speed the flow field should be set to.

        Args:
            wind_speed: Wind speed that the farm should be set to
            if there are no wind speeds stored in the internal buffer (float).

            send_wake: Variable specifying whether or not the turbine currently should propagate its wake downstream (boolean).

            sim_time: Current simulation time (int).

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

    def _search_and_combine_u_wakes(self, wake_deficit, wake_dims, index, sim_time):
        """
        Finds all wake deficits at a given simulation time in the buffer and averages them.

        Args:
            wake_deficit: Wake deficit that should be used if there are no wake deficits in the buffer (np array).
            
            wake_dims: Dimensions of the wake deficit matrix (tuple).

            index: What index of the wake deficit buffer to look at (int).

            sim_time: The current simulation time (int).
        """
        current_effects = []
 
        for wake_effect in self._future_wake_deficits[index]:
            if wake_effect[1] == sim_time:
                current_effects.append(wake_effect[0])

        #if len(current_effects) >= 1: print("Well, this should not happen.")
        if len(current_effects) == 0:
            return wake_deficit
        else:
            return np.mean(current_effects, axis=0)

    def get_u_wake(self, wake_dims, send_wake, sim_time):
        """
        This method returns the filled u_wake matrix for the buffer.

        Args:
            wake_dims: Dimensions of the wake deficit matrix (tuple, ints)

            send_wake: Variable specifying whether or not the turbine currently should propagate its wake downstream (boolean).

            sim_time: Current simulation time (int).
        """

        send_wake_temp = False
        #print(self._current_wake_deficits)
        # iterate through the turbine.wake_effects list, which contains a wake_effect entry for every turbine in the farm
        for k in range(len(self._future_wake_deficits)):
            # the third element of the wake_effect entry is the index of the turbine (in the sorted_map) that the wake 
            # corresponds to 
            new_u_wake = self._search_and_combine_u_wakes(self._current_wake_deficits[k], wake_dims, k, sim_time)
            old_u_wake = self._current_wake_deficits[k]
            # if self.number == 1:
            #     print((np.array([new_u_wake == old_u_wake])).all())
            #     print(new_u_wake)
            #     print(old_u_wake)
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

        if self._current_wake_deficits[index] is None:
            
            self._current_wake_deficits[index] = wake_deficit

            #print(self.number, "initializes wake deficit.")

        #print(self._current_wake_deficits)

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
        #print(new_wake_deficit, "added at", delayed_time)
        #print("_current_wake_deficits is", self._current_wake_deficits)
        #print("wake deficit added at time", delayed_time)
        #print("Turbine", self.number, "receives wake effect from Turbine", index, "at time", sim_time)
        #print(new_wake_deficit)

        # the below line does not include any checking to make sure that there are no earlier-time wake effects already in the buffer
        #self._future_wake_deficits[index].append((new_wake_deficit, sim_time))