# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ray-cast sensor."""


from __future__ import annotations

from dataclasses import MISSING

from omni.isaac.orbit.utils import configclass

from ..sensor_base_cfg import SensorBaseCfg
from .patterns.patterns_cfg import PatternBaseCfg
from .ray_caster import RayCaster


@configclass
class RayCasterCfg(SensorBaseCfg):
    """Configuration for the ray-cast sensor."""

    @configclass
    class OffsetCfg:
        """The offset pose of the sensor's frame from the sensor's parent frame."""

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """Quaternion rotation ``(w, x, y, z)`` w.r.t. the parent frame. Defaults to (1.0, 0.0, 0.0, 0.0)."""

    cls_name = RayCaster

    mesh_prim_paths: list[str] = MISSING
    """The list of mesh primitive paths to ray cast against.

    Note:
        Currently, only a single static mesh is supported. We are working on supporting multiple
        static meshes and dynamic meshes.
    """

    offset: OffsetCfg = OffsetCfg()
    """The offset pose of the sensor's frame from the sensor's parent frame. Defaults to identity."""

    attach_yaw_only: bool = MISSING
    """Whether the rays' starting positions and directions only track the yaw orientation.

    This is useful for ray-casting height maps, where only yaw rotation is needed.
    """

    pattern_cfg: PatternBaseCfg = MISSING
    """The pattern that defines the local ray starting positions and directions."""

    max_distance: float = 100.0
    """Maximum distance (in meters) from the sensor to ray cast to. Defaults to 100.0."""
