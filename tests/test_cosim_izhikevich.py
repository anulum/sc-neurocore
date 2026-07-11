# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Izhikevich co-simulation contracts

"""Izhikevich Q16.16 parity contracts."""

from __future__ import annotations

import pytest

from tests.cosim_support import (
    HAS_IVERILOG,
    _python_spike_count,
    _verilog_spike_count_q1616,
)

_N_STEPS = 200
_INPUT_CURRENT = 50.0


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
class TestQ1616Precision:
    """Q16.16 precision mode: 16 integer + 16 fractional bits (32-bit).

    Q16.16 combines Q8.8's wide integer range [-32768, +32767] with
    1/65536 ≈ 0.000015 resolution. This is the "gold standard" for
    hardware neuron fidelity, suitable for all model dynamics.
    """

    def test_izhikevich_q1616_candidate_reset_parity(self) -> None:
        """Q16.16 preserves exact Izhikevich parity with candidate-based recovery reset.

        The coarse Q8.8 path shifts one marginal spike after correcting ``u = u + d``
        to read the integrated candidate. At Q16.16 the same 200-step, ``I=50``
        operating point reproduces all 25 float64 spikes, proving the semantic fix
        does not trade fidelity for the Q8.8 baseline count.
        """
        python_spikes = _python_spike_count("izhikevich", _N_STEPS, _INPUT_CURRENT)
        verilog_spikes = _verilog_spike_count_q1616("izhikevich", _N_STEPS, _INPUT_CURRENT)

        assert python_spikes == verilog_spikes == 25
