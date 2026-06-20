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


import importlib
from typing import TYPE_CHECKING, Any

__version__ = "3.15.34"

# Public symbols load lazily (PEP 562) so ``import sc_neurocore`` stays
# lightweight and, crucially, torch-free: eager submodule imports pulled torch
# (via plasticity/datasets/layers), which segfaults a downstream consumer's
# coverage tracer. Importing a symbol or submodule triggers its module on first
# access. The TYPE_CHECKING block keeps the names visible to type checkers/IDEs.
if TYPE_CHECKING:
    from . import datasets, plasticity
    from .exceptions import (
        SCCompilerError,
        SCConfigError,
        SCDependencyError,
        SCEncodingError,
        SCHardwareError,
        SCNeuroError,
        SCWeightError,
    )
    from .layers import (
        MemristiveDenseLayer,
        SCConv2DLayer,
        SCDenseLayer,
        SCFusionLayer,
        SCLearningLayer,
        SCRecurrentLayer,
        StochasticAttention,
        VectorizedSCLayer,
    )
    from .license import (
        CommercialLicenseStatus,
        get_license_status,
        load_license_from_env,
        set_license_key,
        validate_license_key,
    )
    from .neurons import (
        BaseNeuron,
        FixedPointBitstreamEncoder,
        FixedPointLFSR,
        FixedPointLIFNeuron,
        HomeostaticLIFNeuron,
        SCIzhikevichNeuron,
        StochasticDendriticNeuron,
        StochasticLIFNeuron,
    )
    from .recorders import BitstreamSpikeRecorder
    from .sources import BitstreamCurrentSource
    from .synapses import (
        BitstreamDotProduct,
        BitstreamSynapse,
        RewardModulatedSTDPSynapse,
        StochasticSTDPSynapse,
    )
    from .utils import (
        RNG,
        BitstreamAverager,
        BitstreamEncoder,
        bitstream_to_probability,
        deprecated,
        estimate_memory,
        generate_bernoulli_bitstream,
        generate_sobol_bitstream,
    )

#: Public submodules exposed as attributes of the package.
_LAZY_SUBMODULES = frozenset({"datasets", "plasticity"})

#: Public symbol name → the submodule that defines it.
_SYMBOL_SOURCES: dict[str, str] = {
    "BaseNeuron": "neurons",
    "StochasticLIFNeuron": "neurons",
    "FixedPointLIFNeuron": "neurons",
    "FixedPointLFSR": "neurons",
    "FixedPointBitstreamEncoder": "neurons",
    "HomeostaticLIFNeuron": "neurons",
    "StochasticDendriticNeuron": "neurons",
    "SCIzhikevichNeuron": "neurons",
    "BitstreamSynapse": "synapses",
    "BitstreamDotProduct": "synapses",
    "StochasticSTDPSynapse": "synapses",
    "RewardModulatedSTDPSynapse": "synapses",
    "SCDenseLayer": "layers",
    "SCConv2DLayer": "layers",
    "SCLearningLayer": "layers",
    "VectorizedSCLayer": "layers",
    "SCRecurrentLayer": "layers",
    "MemristiveDenseLayer": "layers",
    "SCFusionLayer": "layers",
    "StochasticAttention": "layers",
    "BitstreamCurrentSource": "sources",
    "RNG": "utils",
    "BitstreamEncoder": "utils",
    "BitstreamAverager": "utils",
    "generate_bernoulli_bitstream": "utils",
    "generate_sobol_bitstream": "utils",
    "bitstream_to_probability": "utils",
    "deprecated": "utils",
    "estimate_memory": "utils",
    "BitstreamSpikeRecorder": "recorders",
    "CommercialLicenseStatus": "license",
    "get_license_status": "license",
    "load_license_from_env": "license",
    "set_license_key": "license",
    "validate_license_key": "license",
    "SCNeuroError": "exceptions",
    "SCEncodingError": "exceptions",
    "SCConfigError": "exceptions",
    "SCWeightError": "exceptions",
    "SCDependencyError": "exceptions",
    "SCHardwareError": "exceptions",
    "SCCompilerError": "exceptions",
}


def __getattr__(name: str) -> Any:
    """Lazily resolve public symbols and submodules (PEP 562).

    Keeps ``import sc_neurocore`` free of heavy/optional dependencies (notably
    torch), which is required for downstream coverage tracers not to crash.
    """
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    source = _SYMBOL_SOURCES.get(name)
    if source is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{source}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List the lazily-exposed public API plus the module's own attributes."""
    return sorted(set(__all__) | set(globals()))


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
