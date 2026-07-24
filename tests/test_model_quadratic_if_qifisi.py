# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestQIFISI from former test_model_quadratic_if.py

"""Focused suite: TestQIFISI from former test_model_quadratic_if.py."""

from __future__ import annotations

from tests.model_quadratic_if_support import *  # noqa: F403


class TestQIFISI:
    def test_constant_isi(self):
        """Deterministic → constant ISI at steady state."""
        n = QuadraticIFNeuron()
        spikes = _run(n, current=1.0, steps=50000)
        assert len(spikes) >= 20
        isis = np.diff(spikes[5:]).astype(float)  # skip transient
        cv = np.std(isis) / np.mean(isis)
        assert cv < 0.02, f"CV(ISI) = {cv:.4f}"

    def test_isi_shortens_with_current(self):
        n1 = QuadraticIFNeuron()
        n5 = QuadraticIFNeuron()
        s1 = _run(n1, current=1.0, steps=50000)
        s5 = _run(n5, current=5.0, steps=50000)
        isi1 = np.mean(np.diff(s1[5:])) if len(s1) > 10 else float("inf")
        isi5 = np.mean(np.diff(s5[5:])) if len(s5) > 10 else float("inf")
        assert isi5 < isi1
