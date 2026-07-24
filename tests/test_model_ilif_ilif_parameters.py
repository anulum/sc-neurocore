# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestILIFParameters from former test_model_ilif.py

"""Focused suite: TestILIFParameters from former test_model_ilif.py."""

from __future__ import annotations

from tests.model_ilif_support import *  # noqa: F403


class TestILIFParameters:
    @pytest.mark.parametrize("inh_strength", [0.0, 0.5, 2.0])
    def test_inh_strength_sweep(self, inh_strength: float):
        n = InhibitoryLIFNeuron(inh_strength=inh_strength)
        spikes = len(_run(n, current=5.0, steps=5000))
        assert isinstance(spikes, int)

    def test_stronger_inhibition_fewer_spikes(self):
        s_weak = len(_run(InhibitoryLIFNeuron(inh_strength=0.1), 5.0, 5000))
        s_strong = len(_run(InhibitoryLIFNeuron(inh_strength=2.0), 5.0, 5000))
        assert s_weak >= s_strong

    @pytest.mark.parametrize("tau_inh", [2.0, 5.0, 20.0])
    def test_tau_inh_sweep(self, tau_inh: float):
        n = InhibitoryLIFNeuron(tau_inh=tau_inh)
        for _ in range(5000):
            n.step(5.0)
        assert np.isfinite(n.v)
