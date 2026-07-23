# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_research_modules.py

from __future__ import annotations

"""
Tests for research/contrib modules that were at 0% coverage:
  - chaos/rng.py (ChaoticRNG)
  - analysis/explainability.py (SpikeToConceptMapper)
  - analysis/kardashev.py (KardashevEstimator)
  - analysis/consciousness.py (PhiEvaluator)
  - bio/neuromodulation.py (NeuromodulatorSystem)
  - spatial/representations.py (VoxelGrid, PointCloud)
  - spatial/transformer_3d.py (SpatialTransformer3D)
  - physics/wolfram_hypergraph.py (WolframHypergraph)
  - core/mdl_parser.py (MindDescriptionLanguage, MDLSpecification)
  - learning/neuroevolution.py (SNNGeneticEvolver)
  - robotics/swarm.py (SwarmCoupling)
"""
import pytest
import numpy as np
from sc_neurocore.chaos.rng import ChaoticRNG
from sc_neurocore.analysis.explainability import SpikeToConceptMapper
from speculative.analysis_kardashev import KardashevEstimator
from speculative.analysis_consciousness import PhiEvaluator
from sc_neurocore.bio.neuromodulation import NeuromodulatorSystem
from sc_neurocore.spatial.representations import VoxelGrid, PointCloud
from sc_neurocore.spatial.transformer_3d import SpatialTransformer3D
from sc_neurocore.physics.wolfram_hypergraph import WolframHypergraph
from sc_neurocore.core.mdl_parser import MindDescriptionLanguage, MDLSpecification
from sc_neurocore.learning.neuroevolution import SNNGeneticEvolver
from sc_neurocore.robotics.swarm import SwarmCoupling
from sc_neurocore.layers.sc_learning_layer import SCLearningLayer

__all__ = ['pytest', 'np', 'ChaoticRNG', 'SpikeToConceptMapper', 'KardashevEstimator', 'PhiEvaluator', 'NeuromodulatorSystem', 'VoxelGrid', 'PointCloud', 'SpatialTransformer3D', 'WolframHypergraph', 'MindDescriptionLanguage', 'MDLSpecification', 'SNNGeneticEvolver', 'SwarmCoupling', 'SCLearningLayer']
