# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Shared folded-area estimator test fixtures

"""Shared folded-resource construction for area-estimator tests."""

from __future__ import annotations

from sc_neurocore.nir_bridge.fpga_compiler import FoldedResourceMetrics

_DW = 16


def _metrics(
    *,
    neurons: int = 64,
    pe_instances: int = 1,
    shared_multipliers: int = 8,
    populations: int = 2,
    state_vars: int = 1,
) -> FoldedResourceMetrics:
    return FoldedResourceMetrics(
        neurons=neurons,
        state_vars_per_neuron=state_vars,
        pe_instances=pe_instances,
        shared_multipliers=shared_multipliers,
        state_ram_bits=neurons * state_vars * _DW,
        cycles_per_tick=neurons + 1,
        direct_neuron_instances=neurons,
        populations=populations,
    )
