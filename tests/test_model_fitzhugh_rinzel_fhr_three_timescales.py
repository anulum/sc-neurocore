# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFHRThreeTimescales from former test_model_fitzhugh_rinzel.py

"""Focused suite: TestFHRThreeTimescales from former test_model_fitzhugh_rinzel.py."""

from __future__ import annotations

from tests.model_fitzhugh_rinzel_support import *  # noqa: F403


class TestFHRThreeTimescales:
    def test_y_ultra_slow(self):
        """mu=0.0001 keeps y much slower than w over short horizons."""
        n = FitzHughRinzelNeuron()
        w0, y0 = n.w, n.y
        for _ in range(100):
            n.step(0.5)
        dw = abs(n.w - w0)
        dy = abs(n.y - y0)
        assert dw > 100 * dy, f"dw={dw:.6f}, dy={dy:.6f}"

    def test_y_modulates_oscillation(self):
        """Different y-nullcline offsets change the driven trajectory."""
        n1 = FitzHughRinzelNeuron(c=-0.5)
        n2 = FitzHughRinzelNeuron(c=-1.0)
        s1 = len(_run(n1, current=0.5, steps=10000))
        s2 = len(_run(n2, current=0.5, steps=10000))
        assert s1 != s2 or n1.y != pytest.approx(n2.y)
