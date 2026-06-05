# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Universal Stochastic Computing Framework

from __future__ import annotations

"""
SC-NeuroCore — Universal Stochastic Computing Framework
========================================================

Core public API surface for neuromorphic stochastic computing.
Import the classes you need directly::

    from sc_neurocore import StochasticLIFNeuron, SCDenseLayer, BitstreamEncoder

For hardware-level bit-true models::

    from sc_neurocore.neurons import FixedPointLIFNeuron, FixedPointLFSR

Module Tiers
------------
- **core**     — Production-ready: neurons, synapses, layers, sources, utils,
  recorders, accel.  Imported by default.
- **research** -- Functional but experimental: hdc, solvers, transformers,
  quantum, robotics, bio, physics, etc.  Import explicitly, e.g.
  ``from sc_neurocore.quantum.hybrid import QuantumStochasticLayer``.
- **adapters** -- High-level domain mappings: holonomic (SCPN), audio, etc.
- **contrib**  -- Speculative / theoretical modules have been moved to the
  ``research/`` directory at the repository root. See ``research/README.md``.
"""


__version__ = "3.15.18"

# ── Datasets ────────────────────────────────────────────────────────────────
from . import datasets  # noqa: F401
from . import plasticity

# ── Neurons ──────────────────────────────────────────────────────────────────
from .neurons import (
    BaseNeuron,
    StochasticLIFNeuron,
    FixedPointLIFNeuron,
    FixedPointLFSR,
    FixedPointBitstreamEncoder,
    HomeostaticLIFNeuron,
    StochasticDendriticNeuron,
    SCIzhikevichNeuron,
)

# ── Synapses ─────────────────────────────────────────────────────────────────
from .synapses import (
    BitstreamSynapse,
    BitstreamDotProduct,
    StochasticSTDPSynapse,
    RewardModulatedSTDPSynapse,
)

# ── Layers ───────────────────────────────────────────────────────────────────
from .layers import (
    SCDenseLayer,
    SCConv2DLayer,
    SCLearningLayer,
    VectorizedSCLayer,
    SCRecurrentLayer,
    MemristiveDenseLayer,
    SCFusionLayer,
    StochasticAttention,
)

# ── Sources ──────────────────────────────────────────────────────────────────
from .sources import BitstreamCurrentSource

# ── Utilities ────────────────────────────────────────────────────────────────
from .utils import (
    RNG,
    BitstreamEncoder,
    BitstreamAverager,
    generate_bernoulli_bitstream,
    generate_sobol_bitstream,
    bitstream_to_probability,
    deprecated,
    estimate_memory,
)

# ── Recorders ────────────────────────────────────────────────────────────────
from .recorders import BitstreamSpikeRecorder
from .license import (
    CommercialLicenseStatus,
    get_license_status,
    load_license_from_env,
    set_license_key,
    validate_license_key,
)

from .exceptions import (
    SCNeuroError,
    SCEncodingError,
    SCConfigError,
    SCWeightError,
    SCDependencyError,
    SCHardwareError,
    SCCompilerError,
)

__all__ = [
    # General
    "plasticity",
    # Datasets
    "datasets",
    # Neurons
    "BaseNeuron",
    "StochasticLIFNeuron",
    "FixedPointLIFNeuron",
    "FixedPointLFSR",
    "FixedPointBitstreamEncoder",
    "HomeostaticLIFNeuron",
    "StochasticDendriticNeuron",
    "SCIzhikevichNeuron",
    # Synapses
    "BitstreamSynapse",
    "BitstreamDotProduct",
    "StochasticSTDPSynapse",
    "RewardModulatedSTDPSynapse",
    # Layers
    "SCDenseLayer",
    "SCConv2DLayer",
    "SCLearningLayer",
    "VectorizedSCLayer",
    "SCRecurrentLayer",
    "MemristiveDenseLayer",
    "SCFusionLayer",
    "StochasticAttention",
    # Sources
    "BitstreamCurrentSource",
    # Utilities
    "RNG",
    "BitstreamEncoder",
    "BitstreamAverager",
    "generate_bernoulli_bitstream",
    "generate_sobol_bitstream",
    "bitstream_to_probability",
    "deprecated",
    "estimate_memory",
    # Recorders
    "BitstreamSpikeRecorder",
    # Licensing
    "CommercialLicenseStatus",
    "get_license_status",
    "load_license_from_env",
    "set_license_key",
    "validate_license_key",
    # Exceptions
    "SCNeuroError",
    "SCEncodingError",
    "SCConfigError",
    "SCWeightError",
    "SCDependencyError",
    "SCHardwareError",
    "SCCompilerError",
]
