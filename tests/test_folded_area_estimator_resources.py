# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Folded-area compute resource tests

"""Calibrated PE, multiplier, DSP, and ROM terms for folded area."""

from __future__ import annotations

import math

from sc_neurocore.energy.folded_estimator import estimate_folded_area
from sc_neurocore.energy.fpga_models import EVENT_NEURON, LIF_NEURON
from tests.folded_area_estimator_support import _DW, _metrics


def test_pe_pool_luts_match_calibrated_neuron_cost() -> None:
    est = estimate_folded_area(_metrics(pe_instances=3), target="ice40", data_width=_DW)
    assert est.pe_luts == 3 * LIF_NEURON.luts  # one combinational PE per distinct type


def test_event_driven_uses_event_neuron_and_aer_infra() -> None:
    m = _metrics(pe_instances=2)
    lif = estimate_folded_area(m, target="ice40", data_width=_DW)
    evt = estimate_folded_area(m, target="ice40", data_width=_DW, event_driven=True)
    assert evt.pe_luts == 2 * EVENT_NEURON.luts
    assert evt.pe_luts < lif.pe_luts  # the event neuron is cheaper
    assert evt.infra_luts > lif.infra_luts  # AER encoder + router added
    # Lower switching activity ⇒ lower dynamic power for the same total LUTs scale.
    assert evt.dynamic_power_mw < lif.dynamic_power_mw


def test_dsp_target_uses_one_dsp_per_multiplier_no_lut_multiply() -> None:
    m = _metrics(shared_multipliers=8)
    art = estimate_folded_area(m, target="artix7", data_width=_DW)
    assert art.dsps == 8  # artix7 has DSP slices
    assert art.multiplier_luts == 0


def test_dspless_target_uses_lut_multiply_no_dsp() -> None:
    m = _metrics(shared_multipliers=8)
    ice = estimate_folded_area(m, target="ice40", data_width=_DW)
    assert ice.dsps == 0  # ice40 has no DSP slices
    assert ice.multiplier_luts == 8 * ((_DW * _DW) // 4)


def test_rom_luts_scale_with_multipliers_and_log_neurons() -> None:
    m = _metrics(neurons=64, shared_multipliers=5)
    est = estimate_folded_area(m, target="ice40", data_width=_DW)
    depth = max(1, math.ceil(math.log2(64)))
    assert est.rom_luts == 5 * _DW * depth


def test_no_multiplier_no_rom_for_pure_spiking_network() -> None:
    # A recurrent/inter-pop spiking network has zero shared multipliers (spike-gated),
    # so neither the multiplier nor the weight-ROM terms contribute.
    m = _metrics(shared_multipliers=0)
    est = estimate_folded_area(m, target="ice40", data_width=_DW)
    assert est.multiplier_luts == 0
    assert est.rom_luts == 0
    assert est.dsps == 0
