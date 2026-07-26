# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Folded-area storage and infrastructure tests

"""Flip-flop, BRAM, and infrastructure terms for folded area."""

from __future__ import annotations

import pytest

from sc_neurocore.energy.folded_estimator import estimate_folded_area
from sc_neurocore.energy.fpga_models import AXI_LITE
from tests.folded_area_estimator_support import _DW, _metrics


def test_flip_flops_cover_spike_bus_double_buffer_and_sequencer() -> None:
    m = _metrics(neurons=64, populations=2)
    est = estimate_folded_area(m, target="ice40", data_width=_DW)
    assert est.spike_bus_ffs == 2 * 64  # spike accumulator + committed spike bus
    # pidx (log2 populations) + nidx (log2 neurons) + phase + tick_done
    assert est.sequencer_ffs == 1 + 6 + 2
    assert est.total_ffs == est.spike_bus_ffs + est.sequencer_ffs


def test_bram_kb_from_state_ram_bits() -> None:
    m = _metrics(neurons=100, state_vars=2)  # 100 * 2 * 16 = 3200 bits
    est = estimate_folded_area(m, target="ecp5", data_width=_DW)
    assert est.bram_bits == 3200
    assert est.total_bram_kb == pytest.approx(3200 / 8192.0)


def test_infra_excluded_when_requested() -> None:
    m = _metrics()
    with_infra = estimate_folded_area(m, target="ice40", data_width=_DW)
    without = estimate_folded_area(m, target="ice40", data_width=_DW, include_infra=False)
    assert with_infra.infra_luts == AXI_LITE.luts
    assert without.infra_luts == 0
    assert without.total_luts == with_infra.total_luts - AXI_LITE.luts
