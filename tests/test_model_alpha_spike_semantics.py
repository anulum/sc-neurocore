# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeSemantics from former test_model_alpha.py

"""Focused suite: TestSpikeSemantics from former test_model_alpha.py."""

from __future__ import annotations

from tests.model_alpha_support import *  # noqa: F403

class TestSpikeSemantics:
    """Candidate crossing, somatic reset, cascade preservation."""

    def test_step_returns_binary(self) -> None:
        n = AlphaNeuron()
        assert n.step(0.0) in (0, 1)

    def test_spike_resets_only_the_membrane(self) -> None:
        n = AlphaNeuron(v=0.9, a_exc=0.4, i_exc=0.6, a_inh=0.2, i_inh=0.1, v_threshold=0.5)
        before = (n.a_exc, n.i_exc, n.a_inh, n.i_inh)
        assert n.step(0.0) == 1
        assert n.v == n.v_rest
        decay_exc = math.exp(-1.0 / 5.0)
        decay_inh = math.exp(-1.0 / 10.0)
        assert n.a_exc == pytest.approx(before[0] * decay_exc, rel=0.0, abs=1e-14)
        assert n.i_exc == pytest.approx(
            decay_exc * (before[1] + before[0] * 1.0 / 5.0), rel=0.0, abs=1e-14
        )
        assert n.a_inh == pytest.approx(before[2] * decay_inh, rel=0.0, abs=1e-14)
        assert n.i_inh == pytest.approx(
            decay_inh * (before[3] + before[2] * 1.0 / 10.0), rel=0.0, abs=1e-14
        )

    def test_spikes_under_excitatory_drive(self) -> None:
        n = AlphaNeuron()
        spikes = sum(n.step(3.0) for _ in range(2000))
        assert spikes > 0

    def test_inhibition_suppresses_excitatory_drive(self) -> None:
        exc_only = AlphaNeuron()
        dual = AlphaNeuron()
        exc_spikes = sum(exc_only.step(2.5) for _ in range(500))
        dual_spikes = sum(dual.step(2.5, 1.5) for _ in range(500))
        assert dual_spikes < exc_spikes

    def test_state_finite(self) -> None:
        n = AlphaNeuron()
        for index in range(5000):
            n.step(3.0 + math.sin(index * 0.01), 0.5)
        assert np.isfinite(n.v)
        assert np.isfinite(n.a_exc)
        assert np.isfinite(n.i_exc)
        assert np.isfinite(n.a_inh)
        assert np.isfinite(n.i_inh)

    def test_reset_restores_documented_state_preserving_configuration(self) -> None:
        n = AlphaNeuron()
        for _ in range(100):
            n.step(3.0)
        n.reset()
        assert n.v == n.v_rest
        assert (n.a_exc, n.i_exc, n.a_inh, n.i_inh) == (0.0, 0.0, 0.0, 0.0)
        assert (n.tau_v, n.tau_exc, n.tau_inh, n.dt) == (20.0, 5.0, 10.0, 1.0)
