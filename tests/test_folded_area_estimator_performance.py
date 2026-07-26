# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Folded-area latency and capacity tests

"""Latency, energy, fit, and utilisation contracts for folded area."""

from __future__ import annotations

import pytest

from sc_neurocore.energy.folded_estimator import estimate_folded_area
from sc_neurocore.energy.fpga_models import TARGETS
from tests.folded_area_estimator_support import _DW, _metrics


def test_latency_is_cycles_per_tick_and_drives_energy() -> None:
    m = _metrics(neurons=64)  # cycles_per_tick = 65
    est = estimate_folded_area(m, target="ice40", data_width=_DW, clock_mhz=100.0)
    assert est.latency_cycles == 65
    expected_nj = est.dynamic_power_mw * (65 / (100.0 * 1e6)) * 1e6
    assert est.energy_per_tick_nj == pytest.approx(expected_nj)


def test_fit_and_utilisation_against_target_budget() -> None:
    small = estimate_folded_area(_metrics(neurons=16, shared_multipliers=2), target="ice40")
    assert small.fits_on_target
    assert 0.0 < small.lut_utilisation_pct < 100.0
    # A network whose LUT-based multipliers blow past the ice40 budget does not fit.
    huge = estimate_folded_area(
        _metrics(neurons=4096, shared_multipliers=4096), target="ice40", data_width=_DW
    )
    assert not huge.fits_on_target
    assert huge.lut_utilisation_pct > 100.0


def test_dsp_overflow_marks_unfit_even_when_luts_fit() -> None:
    # More multipliers than the target's DSP slices ⇒ does not fit, despite small LUTs.
    m = _metrics(neurons=8, shared_multipliers=TARGETS["artix7"].total_dsp + 1)
    est = estimate_folded_area(m, target="artix7", data_width=_DW)
    assert est.dsps > TARGETS["artix7"].total_dsp
    assert not est.fits_on_target
