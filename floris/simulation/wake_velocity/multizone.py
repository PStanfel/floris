# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from ...utilities import cosd
from .base_velocity_deficit import VelocityDeficit
import numpy as np


class MultiZone(VelocityDeficit):
    """
    Floris is a subclass of 
    :py:class:`floris.simulation.wake_velocity.VelocityDeficit` that is 
    used to compute the wake velocity deficit based on the original 
    multi-zone FLORIS model. See: 

    Gebraad, P. M. O. et al., "A Data-driven model for wind plant power 
    optimization by yaw control." *Proc. American Control Conference*, 
    Portland, OR, 2014.

    Gebraad, P. M. O. et al., "Wind plant power optimization through 
    yaw control using a parametric model for wake effects - a CFD 
    simulation study." *Wind Energy*, 2016.

    Args:
        parameter_dictionary: A dictionary as generated from the 
            input_reader; it should have the following key-value pairs:

            -   **turbulence_intensity**: A dictionary containing the 
                following key-value pairs used to calculate wake-added 
                turbulence intensity from an upstream - turbine, using 
                the approach of Crespo, A. and Herna, J. "Turbulence 
                characteristics in wind-turbine wakes." *J. Wind Eng 
                Ind Aerodyn*. 1996.:

                -   **initial**: A float that is the initial ambient 
                    turbulence intensity, expressed as a decimal 
                    fraction.
                -   **constant**: A float that is the constant used to 
                    scale the wake-added turbulence intensity.
                -   **ai**: A float that is the axial induction factor 
                    exponent used in in the calculation of wake-added 
                    turbulence.
                -   **downstream**: A float that is the exponent 
                    applied to the distance downtream of an upstream 
                    turbine normalized by the rotor diameter used in 
                    the calculation of wake-added turbulence.

            - **floris**: A dictionary containing the following 
                key-value pairs:

                -   **me**: A list of three floats that help determine 
                    the slope of the diameters of the three wake zones 
                    (near wake, far wake, mixing zone) as a function of 
                    downstream distance.
                -   **we**: A float that is the scaling parameter used 
                    to adjust the wake expansion, helping to determine 
                    the slope of the diameters of the three wake zones 
                    as a function of downstream distance, as well as 
                    the recovery of the velocity deficits in the wake 
                    as a function of downstream distance.
                -   **aU**: A float that is a parameter used to 
                    determine the dependence of the wake velocity 
                    deficit decay rate on the rotor yaw angle.
                -   **bU**: A float that is another parameter used to 
                    determine the dependence of the wake velocity 
                    deficit decay rate on the rotor yaw angle.
                -   **mU**: A list of three floats that are parameters 
                    used to determine the dependence of the wake 
                    velocity deficit decay rate for each of the three 
                    wake zones on the rotor yaw angle.

    Returns:
        An instantiated Floris object.
    """

    def __init__(self, parameter_dictionary):
        super().__init__(parameter_dictionary)
        self.model_string = "multizone"
        model_dictionary = self._get_model_dict()
        self.me = [float(n) for n in model_dictionary["me"]]
        self.we = float(model_dictionary["we"])
        self.aU = float(model_dictionary["aU"])
        self.bU = float(model_dictionary["bU"])
        self.mU = [float(n) for n in model_dictionary["mU"]]

    def function(self, x_locations, y_locations, z_locations, turbine,
                 turbine_coord, deflection_field, flow_field):
        """
        Using the original FLORIS multi-zone wake model, this method 
        calculates and returns the wake velocity deficits, caused by 
        the specified turbine, relative to the freestream velocities at 
        the grid of points comprising the wind farm flow field.

        Args:
            x_locations: An array of floats that contains the 
                streamwise direction grid coordinates of the flow field 
                domain (m).
            y_locations: An array of floats that contains the grid 
                coordinates of the flow field domain in the direction 
                normal to x and parallel to the ground (m).
            z_locations: An array of floats that contains the grid 
                coordinates of the flow field domain in the vertical 
                direction (m).
            turbine: A :py:obj:`floris.simulation.turbine` object that 
                represents the turbine creating the wake.
            turbine_coord: A :py:obj:`floris.utilities.Vec3` object 
                containing the coordinate of the turbine creating the 
                wake (m).
            deflection_field: An array of floats that contains the 
                amount of wake deflection in meters in the y direction 
                at each grid point of the flow field.
            flow_field: A :py:class:`floris.simulation.flow_field` 
                object containing the flow field information for the 
                wind farm.

        Returns:
            Three arrays of floats that contain the wake velocity 
            deficit in m/s created by the turbine relative to the 
            freestream velocities for the u, v, and w components, 
            aligned with the x, y, and z directions, respectively. The 
            three arrays contain the velocity deficits at each grid 
            point in the flow field. 
        """

        mu = self.mU / cosd(self.aU + self.bU * turbine.yaw_angle)

        # distance from wake centerline
        rY = abs(y_locations - (turbine_coord.x2 + deflection_field))
        # rZ = abs(z_locations - (turbine.hub_height))
        dx = x_locations - turbine_coord.x1

        # wake zone diameters
        nearwake = turbine.rotor_radius + self.we * self.me[0] * dx
        farwake = turbine.rotor_radius + self.we * self.me[1] * dx
        mixing = turbine.rotor_radius + self.we * self.me[2] * dx

        # initialize the wake field
        c = np.zeros(x_locations.shape)

        # near wake zone
        mask = rY <= nearwake
        c += mask * (turbine.rotor_diameter /
                     (turbine.rotor_diameter + 2 * self.we * mu[0] * dx))**2
        #mask = rZ <= nearwake
        #c += mask * (radius / (radius + we * mu[0] * dx))**2

        # far wake zone
        # ^ is XOR, x^y:
        #   Each bit of the output is the same as the corresponding bit in x
        #   if that bit in y is 0, and it's the complement of the bit in x
        #   if that bit in y is 1.
        # The resulting mask is all the points in far wake zone that are not
        # in the near wake zone
        mask = (rY <= farwake) ^ (rY <= nearwake)
        c += mask * (turbine.rotor_diameter /
                     (turbine.rotor_diameter + 2 * self.we * mu[1] * dx))**2
        #mask = (rZ <= farwake) ^ (rZ <= nearwake)
        #c += mask * (radius / (radius + we * mu[1] * dx))**2

        # mixing zone
        # | is OR, x|y:
        #   Each bit of the output is 0 if the corresponding bit of x AND
        #   of y is 0, otherwise it's 1.
        # The resulting mask is all the points in mixing zone that are not
        # in the far wake zone and not in  near wake zone
        mask = (rY <= mixing) ^ ((rY <= farwake) | (rY <= nearwake))
        c += mask * (turbine.rotor_diameter /
                     (turbine.rotor_diameter + 2 * self.we * mu[2] * dx))**2
        #mask = (rZ <= mixing) ^ ((rZ <= farwake) | (rZ <= nearwake))
        #c += mask * (radius / (radius + we * mu[2] * dx))**2

        # filter points upstream
        c[x_locations - turbine_coord.x1 < 0] = 0

        return 2 * turbine.aI * c * flow_field.wind_map.grid_wind_speed, \
               np.zeros(np.shape(c)), np.zeros(np.shape(c))

    @property
    def me(self):
        """
        A list of three floats that help determine the slope of the diameters
            of the three wake zones (near wake, far wake, mixing zone) as a
            function of downstream distance.
        Args:
            me (list): Three floats that help determine the slope of the
                diameters of the three wake zones.
        Returns:
            float: Three floats that help determine the slope of the diameters
                of the three wake zones.
        """
        return self._me

    @me.setter
    def me(self, value):
        if type(value) is list and len(value) == 3 and \
                            all(type(val) is float for val in value) is True:
            self._me = value
        elif type(value) is list and len(value) == 3 and \
                            all(type(val) is float for val in value) is False:
            self._me = [float(val) for val in value]
        else:
            raise ValueError("Invalid value given for me: {}".format(value))

    @property
    def we(self):
        """
        A float that is the scaling parameter used to adjust the wake expansion,
            helping to determine the slope of the diameters of the three wake
            zones as a function of downstream distance, as well as the recovery
            of the velocity deficits in the wake as a function of downstream
            distance.
        Args:
            we (float, int): Scaling parameter used to adjust the wake
                expansion.
        Returns:
            float: Scaling parameter used to adjust the wake expansion.
        """
        return self._we

    @we.setter
    def we(self, value):
        if type(value) is float:
            self._we = value
        elif type(value) is int:
            self._we = float(value)
        else:
            raise ValueError("Invalid value given for we: {}".format(value))

    @property
    def aU(self):
        """
        A float that is a parameter used to determine the dependence of the
            wake velocity deficit decay rate on the rotor yaw angle.
        Args:
            aU (float, int): Parameter used to determine the dependence of the
                wake velocity deficit decay rate on the rotor yaw angle.
        Returns:
            float: Parameter used to determine the dependence of the wake
                velocity deficit decay rate on the rotor yaw angle.
        """
        return self._aU

    @aU.setter
    def aU(self, value):
        if type(value) is float:
            self._aU = value
        elif type(value) is int:
            self._aU = float(value)
        else:
            raise ValueError("Invalid value given for aU: {}".format(value))

    @property
    def bU(self):
        """
        A float that is a parameter used to determine the dependence of the
            wake velocity deficit decay rate on the rotor yaw angle.
        Args:
            bU (float, int): Parameter used to determine the dependence of the
                wake velocity deficit decay rate on the rotor yaw angle.
        Returns:
            float: Parameter used to determine the dependence of the wake
                velocity deficit decay rate on the rotor yaw angle.
        """
        return self._bU

    @bU.setter
    def bU(self, value):
        if type(value) is float:
            self._bU = value
        elif type(value) is int:
            self._bU = float(value)
        else:
            raise ValueError("Invalid value given for bU: {}".format(value))

    @property
    def mU(self):
        """
        A list of three floats that are parameters used to determine the
            dependence of the wake velocity deficit decay rate for each of the
            three wake zones on the rotor yaw angle.
        Args:
            me (list): Three floats that are parameters used to determine the
                dependence of the wake velocity deficit decay rate.
        Returns:
            float: Three floats that are parameters used to determine the
                dependence of the wake velocity deficit decay rate.
        """
        return self._mU

    @mU.setter
    def mU(self, value):
        if type(value) is list and len(value) == 3 and \
                            all(type(val) is float for val in value) is True:
            self._mU = value
        elif type(value) is list and len(value) == 3 and \
                            all(type(val) is float for val in value) is False:
            self._mU = [float(val) for val in value]
        else:
            raise ValueError("Invalid value given for mU: {}".format(value))