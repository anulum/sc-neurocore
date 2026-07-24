# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestKLIFParameters from former test_model_klif.py

"""Focused suite: TestKLIFParameters from former test_model_klif.py."""

from __future__ import annotations

from tests.model_klif_support import *  # noqa: F403


class TestKLIFParameters:
    @pytest.mark.parametrize("k", [0.5, 1.0, 2.0])
    def test_k_sweep(self, k: float):
        n = KLIFNeuron(k=k)
        spikes = len(_run(n, current=1.0, steps=5000))
        assert isinstance(spikes, int)

    def test_higher_k_more_spikes(self):
        s_low = len(_run(KLIFNeuron(k=0.5), 1.0, 5000))
        s_high = len(_run(KLIFNeuron(k=2.0), 1.0, 5000))
        assert s_high >= s_low

    @pytest.mark.parametrize("tau", [5.0, 10.0, 20.0])
    def test_tau_sweep(self, tau: float):
        n = KLIFNeuron(tau=tau)
        for _ in range(5000):
            n.step(1.0)
        assert np.isfinite(n.v)
