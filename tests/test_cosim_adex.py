# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — AdEx co-simulation contracts

"""AdEx Q16.16 parity contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.cosim_support import (
    HAS_IVERILOG,
    _python_spike_count,
    _verilog_spike_count_q1616,
)

REPOSITORY = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
class TestQ1616Precision:
    """Q16.16 precision mode: 16 integer + 16 fractional bits (32-bit).

    Q16.16 combines Q8.8's wide integer range [-32768, +32767] with
    1/65536 ≈ 0.000015 resolution. This is the "gold standard" for
    hardware neuron fidelity, suitable for all model dynamics.
    """

    @pytest.mark.parametrize(
        ("current", "expected_events"),
        [(0.0, 0), (200.0, 2), (500.0, 6), (1000.0, 12)],
    )
    def test_adex_q1616_parity(self, current: float, expected_events: int) -> None:
        """Q16.16 must preserve the maintained event count across four drives."""
        py_spikes = _python_spike_count("adex", 500, current)
        vlog_spikes = _verilog_spike_count_q1616("adex", 500, current)
        assert py_spikes == vlog_spikes == expected_events


def test_committed_yosys_report_proves_nontrivial_coarse_synthesis() -> None:
    """The committed Q16.16 design must match the executed Yosys receipt."""
    report = json.loads(
        (REPOSITORY / "hdl/reports/yosys_adex_q1616_2026-08-30.json").read_text(encoding="utf-8")
    )
    module = report["modules"]["\\sc_adex"]
    assert module["num_processes"] == 0
    assert module["num_cells"] == 52014
    assert module["num_cells_by_type"]["$_DFF_PN0_"] == 50
    assert module["num_cells_by_type"]["$_DFF_PN1_"] == 15
    assert module["num_cells_by_type"]["$_MUX_"] == 6732
