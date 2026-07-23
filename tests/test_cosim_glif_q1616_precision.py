# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestQ1616Precision from former test_cosim_glif.py

"""Focused suite: TestQ1616Precision from former test_cosim_glif.py."""

from __future__ import annotations

from tests.cosim_glif_support import *  # noqa: F403

@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
class TestQ1616Precision:
    """Q16.16 GLIF co-simulation fidelity."""

    @pytest.mark.parametrize(
        ("current", "expected_spikes"),
        ((0.0, 0), (15.0, 0), (22.0, 23), (30.0, 54), (45.0, 86), (50.0, 95)),
        ids=("rest", "subthreshold", "onset", "tonic", "high-drive", "strong-drive"),
    )
    def test_glif_q1616_parity(self, current: float, expected_spikes: int) -> None:
        """GLIF has exact hand/schema/Q16.16 spike-count parity across six regimes.

        The schema mirrors the maintained four-state, candidate-first classical-RK4
        hand model with level ``v >= theta`` detection and adaptive reset. Hand model
        and schema runner agree exactly at every operating point. The compiler lowers
        reset expressions from the integrated candidate and exposes the same post-reset
        state in RTL, so Q16.16 preserves the complete spike count despite quantising
        ``a_theta=0.01`` and the adaptive increments. Rest, subthreshold, onset, tonic,
        and high-drive regimes are all enrolled rather than one selected current.
        """
        n_steps = 1000
        hand_spikes = _glif_hand_spike_count(n_steps, current)
        schema_spikes = _python_spike_count("glif", n_steps, current)
        verilog_spikes = _verilog_spike_count_q1616("glif", n_steps, current)

        assert hand_spikes == schema_spikes == verilog_spikes == expected_spikes, (
            f"GLIF exact Q16.16 mismatch at I={current}: "
            f"hand={hand_spikes}, schema={schema_spikes}, verilog={verilog_spikes}"
        )
