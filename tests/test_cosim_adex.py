# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — AdEx co-simulation contracts

"""AdEx Q16.16 parity contracts."""

from __future__ import annotations

import pytest

from tests.cosim_support import (
    HAS_IVERILOG,
    _python_spike_count,
    _verilog_spike_count_q1616,
)


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
class TestQ1616Precision:
    """Q16.16 precision mode: 16 integer + 16 fractional bits (32-bit).

    Q16.16 combines Q8.8's wide integer range [-32768, +32767] with
    1/65536 ≈ 0.000015 resolution. This is the "gold standard" for
    hardware neuron fidelity, suitable for all model dynamics.
    """

    def test_adex_q1616_parity(self) -> None:
        """Adaptive-exponential IF (exp spike + adaptation + reset) is bit-true at Q16.16."""
        py_spikes = _python_spike_count("adex", 500, 1000.0)
        vlog_spikes = _verilog_spike_count_q1616("adex", 500, 1000.0)
        assert py_spikes > 0 and vlog_spikes > 0
        gap_pct = abs(py_spikes - vlog_spikes) / max(py_spikes, 1) * 100
        assert gap_pct <= 2.0, (
            f"AdEx Q16.16 gap {gap_pct:.1f}% (Python={py_spikes}, Verilog={vlog_spikes})"
        )
