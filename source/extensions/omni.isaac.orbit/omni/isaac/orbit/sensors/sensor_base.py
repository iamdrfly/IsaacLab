# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for sensors.

This class defines an interface for sensors similar to how the :class:`omni.isaac.orbit.robot.robot_base.RobotBase` class works.
Each sensor class should inherit from this class and implement the abstract methods.
"""

from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Sequence

import omni.isaac.core.utils.prims as prim_utils
import omni.kit.app
import omni.physx
from omni.isaac.core.simulation_context import SimulationContext

if TYPE_CHECKING:
    from .sensor_base_cfg import SensorBaseCfg


class SensorBase(ABC):
    """The base class for implementing a sensor.

    The implementation is based on lazy evaluation. The sensor data is only updated when the user
    tries accessing the data through the :attr:`data` property or sets ``force_compute=True`` in
    the :meth:`update` method. This is done to avoid unnecessary computation when the sensor data
    is not used.

    The sensor is updated at the specified update period. If the update period is zero, then the
    sensor is updated at every simulation step.
    """

    def __init__(self, cfg: SensorBaseCfg):
        """Initialize the sensor class.

        Args:
            cfg: The configuration parameters for the sensor.
        """
        # check that config is valid
        if cfg.history_length < 0:
            raise ValueError(f"History length must be greater than 0! Received: {cfg.history_length}")
        # store inputs
        self.cfg = cfg
        # flag for whether the sensor is initialized
        self._is_initialized = False

        # add callbacks for stage play/stop
        physx_interface = omni.physx.acquire_physx_interface()
        self._initialize_handle = physx_interface.get_simulation_event_stream_v2().create_subscription_to_pop_by_type(
            int(omni.physx.bindings._physx.SimulationEvent.RESUMED), self._initialize_callback
        )
        self._invalidate_initialize_handle = (
            physx_interface.get_simulation_event_stream_v2().create_subscription_to_pop_by_type(
                int(omni.physx.bindings._physx.SimulationEvent.STOPPED), self._invalidate_initialize_callback
            )
        )
        # add callback for debug visualization
        if self.cfg.debug_vis:
            app_interface = omni.kit.app.get_app_interface()
            self._debug_visualization_handle = app_interface.get_post_update_event_stream().create_subscription_to_pop(
                self._debug_vis_callback
            )
        else:
            self._debug_visualization_handle = None

    def __del__(self):
        """Unsubscribe from the callbacks."""
        self._initialize_handle.unsubscribe()
        self._invalidate_initialize_handle.unsubscribe()
        if self._debug_visualization_handle is not None:
            self._debug_visualization_handle.unsubscribe()

    """
    Properties
    """

    @property
    def device(self) -> str:
        """Memory device for computation."""
        return self._device

    @property
    @abstractmethod
    def data(self) -> Any:
        """Data from the sensor.

        This property is only updated when the user tries to access the data. This is done to avoid
        unnecessary computation when the sensor data is not used.

        For updating the sensor when this property is accessed, you can use the following
        code snippet in your sensor implementation:

        .. code-block:: python

            # update sensors if needed
            self._update_outdated_buffers()
            # return the data (where `_data` is the data for the sensor)
            return self._data
        """
        raise NotImplementedError

    """
    Operations
    """

    def set_debug_vis(self, debug_vis: bool):
        """Sets whether to visualize the sensor data.

        Args:
            debug_vis: Whether to visualize the sensor data.

        Raises:
            RuntimeError: If the asset debug visualization is not enabled.
        """
        if not self.cfg.debug_vis:
            raise RuntimeError("Debug visualization is not enabled for this sensor.")

    def reset(self, env_ids: Sequence[int] | None = None):
        """Resets the sensor internals.

        Args:
            env_ids: The sensor ids to reset. Defaults to None.
        """
        # Resolve sensor ids
        if env_ids is None:
            env_ids = ...
        # Reset the timestamp for the sensors
        self._timestamp[env_ids] = 0.0
        self._timestamp_last_update[env_ids] = 0.0
        # Set all reset sensors to outdated so that they are updated when data is called the next time.
        self._is_outdated[env_ids] = True

    def update(self, dt: float, force_recompute: bool = False):
        # Update the timestamp for the sensors
        self._timestamp += dt
        self._is_outdated |= self._timestamp - self._timestamp_last_update + 1e-6 >= self.cfg.update_period
        # Update the buffers
        # TODO (from @mayank): Why is there a history length here when it doesn't mean anything in the sensor base?!?
        #   It is only for the contact sensor but there we should redefine the update function IMO.
        if force_recompute or (self.cfg.history_length > 0):
            self._update_outdated_buffers()

    """
    Implementation specific.
    """

    @abstractmethod
    def _initialize_impl(self):
        """Initializes the sensor-related handles and internal buffers."""
        # Obtain Simulation Context
        sim = SimulationContext.instance()
        if sim is not None:
            self._device = sim.device
            self._sim_physics_dt = sim.get_physics_dt()
        else:
            raise RuntimeError("Simulation Context is not initialized!")
        # Count number of environments
        env_prim_path_expr = self.cfg.prim_path.rsplit("/", 1)[0]
        self._num_envs = len(prim_utils.find_matching_prim_paths(env_prim_path_expr))
        # Boolean tensor indicating whether the sensor data has to be refreshed
        self._is_outdated = torch.ones(self._num_envs, dtype=torch.bool, device=self._device)
        # Current timestamp (in seconds)
        self._timestamp = torch.zeros(self._num_envs, device=self._device)
        # Timestamp from last update
        self._timestamp_last_update = torch.zeros_like(self._timestamp)

    @abstractmethod
    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the sensor data for provided environment ids.

        This function does not perform any time-based checks and directly fills the data into the
        data container.

        Args:
            env_ids: The indices of the sensors that are ready to capture.
        """
        raise NotImplementedError

    def _debug_vis_impl(self):
        """Visualizes the sensor data.

        This is an empty function that can be overridden by the derived class to visualize the sensor data.

        Note:
            Visualization of sensor data may add overhead to the simulation. It is recommended to disable
            visualization when running the simulation in headless mode.
        """
        pass

    """
    Simulation callbacks.
    """

    def _initialize_callback(self, event):
        """Initializes the scene elements.

        Note:
            PhysX handles are only enabled once the simulator starts playing. Hence, this function needs to be
            called whenever the simulator "plays" from a "stop" state.
        """
        if not self._is_initialized:
            self._initialize_impl()
            self._is_initialized = True

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        self._is_initialized = False

    def _debug_vis_callback(self, event):
        """Visualizes the sensor data."""
        self._debug_vis_impl()

    """
    Helper functions.
    """

    def _update_outdated_buffers(self):
        """Fills the sensor data for the outdated sensors."""
        outdated_env_ids = self._is_outdated.nonzero().squeeze(-1)
        if len(outdated_env_ids) > 0:
            # obtain new data
            self._update_buffers_impl(outdated_env_ids)
            # update the timestamp from last update
            self._timestamp_last_update[outdated_env_ids] = self._timestamp[outdated_env_ids]
            # set outdated flag to false for the updated sensors
            self._is_outdated[outdated_env_ids] = False
