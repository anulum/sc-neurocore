# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSRM0FI from former test_model_srm0.py

"""Focused suite: TestSRM0FI from former test_model_srm0.py."""

from __future__ import annotations

from tests.model_srm0_support import *  # noqa: F403

class TestSRM0FI:
    def test_subthreshold_silent(self) -> None:
        n = SRM0Neuron()
        assert len(_run(n, current=0.5, steps=10000)) == 0

    def test_suprathreshold_fires(self) -> None:
        n = SRM0Neuron()
        assert len(_run(n, current=2.0, steps=10000)) >= 20

    def test_monotonic_fi(self) -> None:
        rates = []
        for I in [2.0, 3.0, 5.0, 10.0]:
            n = SRM0Neuron()
            rates.append(len(_run(n, current=I, steps=10000)))
        assert all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))

    def test_isi_regularity(self) -> None:
        n = SRM0Neuron()
        spikes = _run(n, current=5.0, steps=10000)
        if len(spikes) >= 20:
            isis = np.diff(spikes[5:]).astype(float)
            cv = np.std(isis) / np.mean(isis)
            assert cv < 0.1
