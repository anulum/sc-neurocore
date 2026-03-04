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
- **research** ÔÇö Functional but experimental: hdc, solvers, transformers,
  quantum, robotics, bio, physics, etc.  Import explicitly, e.g.
  ``from sc_neurocore.quantum.hybrid import QuantumStochasticLayer``.
- **adapters** ÔÇö High-level domain mappings: holonomic (SCPN), audio, etc.
- **contrib**  ÔÇö Speculative / theoretical modules have been moved to the
  ``research/`` directory at the repository root. See ``research/README.md``.
"""


__version__ = "3.7.0"

# ÔöÇÔöÇ Adapters ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
from .adapters import base as adapter_base


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
)

# ── Recorders ────────────────────────────────────────────────────────────────
from .recorders import BitstreamSpikeRecorder

__all__ = [
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
    # Recorders
    "BitstreamSpikeRecorder",
]
