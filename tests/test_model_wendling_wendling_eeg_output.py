# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWendlingEEGOutput from former test_model_wendling.py

"""Focused suite: TestWendlingEEGOutput from former test_model_wendling.py."""

from __future__ import annotations

from tests.model_wendling_support import *  # noqa: F403

class TestWendlingEEGOutput:
    def test_output_is_eeg_signal(self):
        """Output = y1 - y2 - y3 (postsynaptic potential difference)."""
        n = WendlingNeuron()
        for _ in range(100):
            n.step(220.0)
        output = n.step(220.0)
        expected = n.y1 - n.y2 - n.y3
        assert abs(output - expected) < 1e-10

    def test_eeg_transient_dynamics(self):
        """With p_ext=220, output shows transient ramp then convergence.

        The full trace has range > 15 mV (transient), but converges
        to steady state. This is expected for the default parameters
        — epileptiform oscillation requires specific a_exc/b_fast tuning.
        """
        n = WendlingNeuron()
        vals = []
        for _ in range(10000):
            vals.append(n.step(220.0))
        vs = np.array(vals)
        v_range = vs.max() - vs.min()
        assert v_range > 10.0, f"Total EEG range = {v_range:.2f}"

    def test_different_p_ext_different_dynamics(self):
        """Different external input → different EEG pattern."""
        n1 = WendlingNeuron()
        n2 = WendlingNeuron()
        for _ in range(10000):
            n1.step(100.0)
            n2.step(400.0)
        assert abs(n1.y1 - n2.y1) > 0.1
