# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from omni.isaac.lab.utils import configclass

from . import supsi_hf_terrains
from .hf_terrains_cfg import HfTerrainBaseCfg
import numpy as np

"""
Different height field terrain configurations.
"""

@configclass
class SupsiSlopedTerrainCfg(HfTerrainBaseCfg):

    function = supsi_hf_terrains.supsi_sloped_terrain

    angle_range: tuple[int, int] = MISSING
    """The angle of the slope (in degrees)."""

    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""

    inverted: bool = False
    """Whether the pyramid is inverted. Defaults to False."""

    difficulty: float = 1.0

    plane_step = 30 # approx since it's 79 total size here

@configclass
class SupsiExpTerrainCfg(HfTerrainBaseCfg):

    function = supsi_hf_terrains.supsi_exp_terrain

    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""

    inverted: bool = False
    """Whether the pyramid is inverted. Defaults to False."""

    difficulty: float = 1.0

    plane_step = 30 # approx since it's 79 total size here

@configclass
class SupsiSingleCubeTerrainCfg(HfTerrainBaseCfg):

    function = supsi_hf_terrains.supsi_single_cube_terrain

    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""

    inverted: bool = False
    """Whether the pyramid is inverted. Defaults to False."""

    difficulty: float = 1.0

    plane_step = 30 # approx since it's 79 total size here

    cube_dim = 20
    overlap = False


@configclass
class SupsiMultiCubeTerrainCfg(HfTerrainBaseCfg):

    function = supsi_hf_terrains.supsi_multi_cube_terrain

    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""

    inverted: bool = False
    """Whether the pyramid is inverted. Defaults to False."""

    difficulty: float = 1.0

    plane_step = 30 # approx since it's 79 total size here

    cube_dim = 20
    overlap = False