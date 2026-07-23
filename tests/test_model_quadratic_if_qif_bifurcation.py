# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestQIFBifurcation from former test_model_quadratic_if.py

"""Focused suite: TestQIFBifurcation from former test_model_quadratic_if.py."""

from __future__ import annotations

from tests.model_quadratic_if_support import *  # noqa: F403

class TestQIFBifurcation:
    """Saddle-node bifurcation at I=0: the defining property of QIF."""

    def test_negative_current_no_spikes(self):
        """I<0 → stable fixed point at V = -sqrt(-I). No spikes."""
        n = QuadraticIFNeuron()
        spikes = _run(n, current=-0.5, steps=50000)
        assert len(spikes) == 0

    def test_zero_current_no_spikes(self):
        """I=0 → half-stable fixed point at V=0. From V=-1, approaches slowly."""
        n = QuadraticIFNeuron()
        spikes = _run(n, current=0.0, steps=50000)
        assert len(spikes) == 0

    def test_positive_current_fires(self):
        """I>0 → no stable fixed point → periodic spiking (limit cycle)."""
        n = QuadraticIFNeuron()
        spikes = _run(n, current=0.5, steps=50000)
        assert len(spikes) >= 50

    def test_type_i_continuous_fi_onset(self):
        """Type-I: firing rate rises continuously from zero at I=0⁺.

        Near bifurcation, f ∝ sqrt(I). Verify rate at I=0.1 < rate at I=1.0,
        and the ratio is consistent with sqrt scaling.
        """
        n1 = QuadraticIFNeuron()
        n2 = QuadraticIFNeuron()
        s1 = len(_run(n1, current=0.1, steps=50000))
        s2 = len(_run(n2, current=1.0, steps=50000))
        assert s2 > s1
        if s1 > 10:
            ratio = s2 / s1
            # sqrt(1.0/0.1) ≈ 3.16, but reset dynamics modify scaling
            assert 1.5 < ratio < 8.0, f"ratio = {ratio:.2f}"
