# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This sub-module contains spawners that spawn assets from files.

Currently, the following spawners are supported:

* :class:`UsdFileCfg`: Spawn an asset from a USD file.
* :class:`UrdfFileCfg`: Spawn an asset from a URDF file.
* :class:`GroundPlaneCfg`: Spawn a ground plane using the grid-world USD file.

"""

from __future__ import annotations

from .from_files import spawn_from_urdf, spawn_from_usd, spawn_ground_plane
from .from_files_cfg import GroundPlaneCfg, UrdfFileCfg, UsdFileCfg

__all__ = [
    # usd
    "UsdFileCfg",
    "spawn_from_usd",
    # urdf
    "UrdfFileCfg",
    "spawn_from_urdf",
    # ground plane
    "GroundPlaneCfg",
    "spawn_ground_plane",
]
