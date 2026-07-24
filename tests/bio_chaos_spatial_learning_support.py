# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_bio_chaos_spatial_learning.py

from __future__ import annotations

"""Tests for de-omitted modules: chaos, analysis, physics, robotics, learning, spatial, bio."""
import numpy as np
import pytest
from sc_neurocore.bio.neuromodulation import NeuromodulatorSystem
from sc_neurocore.chaos.rng import ChaoticRNG
from sc_neurocore.analysis.explainability import SpikeToConceptMapper
from sc_neurocore.physics.wolfram_hypergraph import WolframHypergraph
from sc_neurocore.robotics.swarm import SwarmCoupling
from sc_neurocore.learning.neuroevolution import SNNGeneticEvolver
from sc_neurocore.spatial.representations import VoxelGrid, PointCloud
from sc_neurocore.spatial.transformer_3d import SpatialTransformer3D
from sc_neurocore.layers.sc_learning_layer import SCLearningLayer


class _Individual:
    def __init__(self):
        self.weights = np.random.rand(4, 4)


__all__ = [
    "np",
    "pytest",
    "NeuromodulatorSystem",
    "ChaoticRNG",
    "SpikeToConceptMapper",
    "WolframHypergraph",
    "SwarmCoupling",
    "SNNGeneticEvolver",
    "VoxelGrid",
    "PointCloud",
    "SpatialTransformer3D",
    "SCLearningLayer",
    "_Individual",
]
