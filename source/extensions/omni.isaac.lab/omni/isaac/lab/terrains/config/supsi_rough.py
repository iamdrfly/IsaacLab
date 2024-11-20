# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import omni.isaac.lab.terrains as terrain_gen
import numpy as np

from ..terrain_generator_cfg import TerrainGeneratorCfg

SUPSI_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    color_scheme="height",
    # show_flat_patches=False,
    # --------------
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
            flat_patch_sampling={}
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
)
"""Rough terrains configuration."""


CUBES_SUPSI_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    color_scheme="height",
    # show_flat_patches=False,
    # --------------
    size=(15.0, 15.0),
    border_width=20.0,
    # for faster load, set row=5 and col=8
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.05, 0.23),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.05, 0.23),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "boxes": terrain_gen.MeshRandomGridTerrainCfg(
        #     proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        # ),
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.5
        ),
        # "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        #     proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        # ),
        # "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        #     proportion=0.6, slope_range=(0.0, 25*np.pi/180), platform_width=2.0, border_width=0.25 #era 0.4 tad
        # ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.5, slope_range=(.0, 30*np.pi/180), platform_width=2.0, border_width=0.25 #era 0.4 tad
        ),
        # "supsi": terrain_gen.SupsiSlopedTerrainCfg(
        #     proportion=0.35, angle_range=(0, 90), platform_width=2.0, border_width=1.25
        # ),
        # "supsi_inv": terrain_gen.SupsiSlopedTerrainCfg(
        #     proportion=0.1, angle_range=(0, 90), platform_width=2.0, border_width=1.25, inverted=True
        # )
        "supsi_single_cube": terrain_gen.SupsiSingleCubeTerrainCfg(
            proportion=0.2, cube_dim=35, platform_width=2.0, border_width=1.25, inverted=False
        ),
        "supsi_multi_cube_nearby": terrain_gen.SupsiMultiCubeTerrainCfg(
            proportion=0.4, cube_dim=30, platform_width=2.0, border_width=1.25, inverted=False, overlap=False
        ),
        "supsi_multi_cube_overlap": terrain_gen.SupsiMultiCubeTerrainCfg(
            proportion=0.4, cube_dim=30, platform_width=2.0, border_width=1.25, inverted=False, overlap=True
        ),
    },
)
"""Rough terrains configuration."""