# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestQ1616Precision from former test_cosim_hindmarsh_rose.py

"""Focused suite: TestQ1616Precision from former test_cosim_hindmarsh_rose.py."""

from __future__ import annotations

from tests.cosim_hindmarsh_rose_support import *  # noqa: F403

@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
class TestQ1616Precision:
    """Q16.16 Hindmarsh-Rose co-simulation fidelity."""

    @pytest.mark.parametrize(
        ("current", "expected_spikes"),
        ((0.0, 0), (2.0, 0), (3.0, 26), (4.0, 40), (5.0, 52)),
        ids=("I=0", "I=2", "I=3", "I=4", "I=5"),
    )
    def test_hindmarsh_rose_q1616_parity(self, current: float, expected_spikes: int) -> None:
        """The hand model, schema, and Q16.16 RTL agree over 2,000 RK4 steps.

        The enrolled set spans two silent points and three bursting rates. The schema
        mirrors the maintained simultaneous three-state RK4 flow, rising-edge
        ``x >= x_threshold`` observation, and no-reset semantics. The 2,000-step
        horizon is deliberate: longer chaotic trajectories are separately classified
        below instead of being presented as indefinite fixed-point identity.
        """
        n_steps = 2000
        hand_spikes = _hindmarsh_rose_hand_spike_count(n_steps, current)
        schema_spikes = _python_spike_count("hindmarsh_rose", n_steps, current)
        rtl_spikes = _verilog_spike_count_q1616("hindmarsh_rose", n_steps, current)

        assert hand_spikes == schema_spikes == rtl_spikes == expected_spikes, (
            f"Hindmarsh-Rose three-way mismatch at I={current}: hand={hand_spikes}, "
            f"schema={schema_spikes}, verilog={rtl_spikes}"
        )

    @pytest.mark.parametrize(
        ("current", "expected_float", "expected_rtl"),
        ((2.0, 9, 10), (3.0, 48, 49), (4.0, 85, 86), (5.0, 114, 115)),
        ids=("I=2", "I=3", "I=4", "I=5"),
    )
    def test_hindmarsh_rose_q1616_long_window_boundary(
        self, current: float, expected_float: int, expected_rtl: int
    ) -> None:
        """The 5,000-step chaotic boundary is an explicit one-crossing exclusion."""
        n_steps = 5000
        hand_spikes = _hindmarsh_rose_hand_spike_count(n_steps, current)
        schema_spikes = _python_spike_count("hindmarsh_rose", n_steps, current)
        rtl_spikes = _verilog_spike_count_q1616("hindmarsh_rose", n_steps, current)

        assert hand_spikes == schema_spikes == expected_float
        assert rtl_spikes == expected_rtl == expected_float + 1
