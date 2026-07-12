# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Quantum Annealing Bridge

"""Compatibility façade for modular quantum-annealing tooling.

The bridge compiles SC network structures into validated Ising/QUBO models,
runs classical or optional native/D-Wave solvers, and exposes bounded analysis,
embedding, transformation, decomposition, and export responsibilities.
"""

from __future__ import annotations

from sc_neurocore.bridges import annealing_backends as _backends
from sc_neurocore.bridges.annealing_analysis import (
    EmbeddingAnalyzer,
    EnergyLandscape,
    SampleAggregator,
    TTSAnalyzer,
)
from sc_neurocore.bridges.annealing_compilers import (
    SCBitstreamQUBO,
    SCToIsing,
    SCToQUBO,
)
from sc_neurocore.bridges.annealing_decomposition import ProblemDecomposer
from sc_neurocore.bridges.annealing_hardware import HardwareGraph, ChainBreakResolver
from sc_neurocore.bridges.annealing_io import (
    export_bqm,
    export_ising_json,
    export_qubo_json,
    visualize_ising,
)
from sc_neurocore.bridges.annealing_models import (
    CouplerSpec,
    IsingModel,
    ProblemType,
    QUBOModel,
    QubitSpec,
)
from sc_neurocore.bridges.annealing_solvers import DWaveInterface, SimulatedAnnealer
from sc_neurocore.bridges.annealing_transforms import (
    AnnealingSchedule,
    GaugeTransform,
    SCPrecisionEncoder,
)


# Compatibility observables retained for callers that report optional backend
# availability. Runtime code owns these dependencies in annealing_backends.
_HAS_DIMOD = _backends.HAS_DIMOD
_HAS_DWAVE = _backends.HAS_DWAVE
_HAS_RUST_QA = _backends.HAS_RUST_QA
dimod = _backends.dimod
DWaveSampler = _backends.DWaveSampler
EmbeddingComposite = _backends.EmbeddingComposite
_rust_ising_energy = _backends._rust_ising_energy
_rust_batch_energy = _backends._rust_batch_energy
_rust_sa = _backends._rust_simulated_annealing


__all__ = [
    "AnnealingSchedule",
    "ChainBreakResolver",
    "CouplerSpec",
    "DWaveInterface",
    "EmbeddingAnalyzer",
    "EnergyLandscape",
    "GaugeTransform",
    "HardwareGraph",
    "IsingModel",
    "ProblemDecomposer",
    "ProblemType",
    "QUBOModel",
    "QubitSpec",
    "SampleAggregator",
    "SCBitstreamQUBO",
    "SCPrecisionEncoder",
    "SCToIsing",
    "SCToQUBO",
    "SimulatedAnnealer",
    "TTSAnalyzer",
    "export_bqm",
    "export_ising_json",
    "export_qubo_json",
    "visualize_ising",
]


# Preserve historical pickle/import paths even though implementations now live
# in responsibility modules.
for _public_name in __all__:
    globals()[_public_name].__module__ = __name__
del _public_name
