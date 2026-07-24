# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPospischilAdaptation from former test_model_pospischil.py

"""Focused suite: TestPospischilAdaptation from former test_model_pospischil.py."""

from __future__ import annotations

from tests.model_pospischil_support import *  # noqa: F403


class TestPospischilAdaptation:
    def test_adaptation_lengthens_later_isis(self):
        """I_M activates slowly → later ISIs should be longer than early ISIs.

        This is the hallmark of RS (regular-spiking) neurons.
        """
        n = PospischilNeuron()
        spikes = _run(n, current=10.0, steps=50000)
        assert len(spikes) >= 20
        isis = np.diff(spikes)
        early_mean = np.mean(isis[:5])
        late_mean = np.mean(isis[-5:])
        assert late_mean > early_mean * 0.9, (
            f"Early ISI={early_mean:.1f}, late ISI={late_mean:.1f} — "
            "expected adaptation (late ≥ early)"
        )

    def test_p_variable_grows_during_spiking(self):
        """Slow K gate p should increase during sustained firing."""
        n = PospischilNeuron()
        p0 = n.p
        for _ in range(50000):
            n.step(10.0)
        assert n.p > p0, f"p didn't grow: {p0} → {n.p}"

    def test_fs_no_adaptation(self):
        """FS type (g_m=0) → no adaptation → higher firing rate than RS."""
        n_fs = PospischilNeuron(g_m=0.0)
        n_rs = PospischilNeuron(g_m=0.07)
        s_fs = len(_run(n_fs, current=5.0, steps=50000))
        s_rs = len(_run(n_rs, current=5.0, steps=50000))
        assert s_fs > s_rs, f"FS: {s_fs} spikes, RS: {s_rs} — expected FS > RS"

    def test_g_m_scales_adaptation(self):
        """Higher g_m → stronger adaptation → fewer spikes."""
        n_weak = PospischilNeuron(g_m=0.03)
        n_strong = PospischilNeuron(g_m=0.1)
        s_weak = len(_run(n_weak, current=5.0, steps=50000))
        s_strong = len(_run(n_strong, current=5.0, steps=50000))
        assert s_weak > s_strong
