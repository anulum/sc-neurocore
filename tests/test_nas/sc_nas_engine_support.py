# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_sc_nas_engine.py

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from types import ModuleType
import pytest
from sc_neurocore.nas import sc_nas_engine as nas_module
from sc_neurocore.nas.sc_nas_engine import (
    DecorrelationStrategy,
    EvolutionaryNAS,
    FPGAResourceBudget,
    LayerConfig,
    NASObjective,
    NASReport,
    NASVerilogEmitter,
    NeuronType,
    SCCandidate,
    SCFitnessEvaluator,
    pareto_front,
    run_nas,
)
from sc_neurocore.optimizer.sc_optimizer import LayerProfile
from sc_neurocore.optimizer.surrogate_sc_optimizer import (
    SurrogateLayerConfig,
    SurrogateOptimizerReport,
)

RustTournament = Callable[[list[float], int, int, int], list[int]]


def _surrogate_cfg(length: int) -> SurrogateLayerConfig:
    return SurrogateLayerConfig(
        bitstream_length=length,
        decorrelator="LFSR",
        mode="SC",
        precision_bits=8,
        lfsr_polynomial="x16+x14+x13+x11+1",
        luts_used=100,
        power_used=1.0,
        latency_cycles=length,
        accuracy_score=0.99,
        utility_score=0.95,
    )


class _FakeSurrogateOptimiser:
    def __init__(self) -> None:
        self.calls: list[list[LayerProfile]] = []

    def optimise(self, network: list[LayerProfile]) -> SurrogateOptimizerReport:
        self.calls.append(network)
        return SurrogateOptimizerReport(
            config={
                profile.id: _surrogate_cfg(128 + index * 128)
                for index, profile in enumerate(network)
            },
            total_luts=100 * len(network),
            total_power_mw=float(len(network)),
            total_latency_cycles=256,
            mean_accuracy=0.99,
            training_points=16,
            target_name="unit-fpga",
        )


__all__ = [
    "importlib",
    "sys",
    "Callable",
    "ModuleType",
    "pytest",
    "nas_module",
    "DecorrelationStrategy",
    "EvolutionaryNAS",
    "FPGAResourceBudget",
    "LayerConfig",
    "NASObjective",
    "NASReport",
    "NASVerilogEmitter",
    "NeuronType",
    "SCCandidate",
    "SCFitnessEvaluator",
    "pareto_front",
    "run_nas",
    "LayerProfile",
    "SurrogateLayerConfig",
    "SurrogateOptimizerReport",
    "RustTournament",
    "_surrogate_cfg",
    "_FakeSurrogateOptimiser",
]
