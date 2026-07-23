# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSubstrateCreation from former test_identity_substrate.py

"""Focused suite: TestSubstrateCreation from former test_identity_substrate.py."""

from __future__ import annotations

from tests.identity_substrate_support import *  # noqa: F403

class TestSubstrateCreation:
    def test_populations_created(self):
        sub = _make_substrate()
        assert sub.cortical.n == N_CORTICAL
        assert sub.inhibitory.n == N_INHIBITORY
        assert sub.memory.n == N_MEMORY

    def test_projections_exist(self):
        sub = _make_substrate()
        assert sub.proj_ee.data.size > 0
        assert sub.proj_ei.data.size > 0
        assert sub.proj_ie.data.size > 0
        assert sub.proj_em.data.size > 0
        assert sub.proj_me.data.size > 0

    def test_single_step(self):
        sub = _make_substrate()
        spikes = sub.step()
        assert spikes.shape == (N_CORTICAL,)
        assert spikes.dtype == np.int8
        assert sub._total_steps == 1

    def test_step_zero_pads_short_stimuli(self):
        # Stimuli narrower than the cortical population are zero-padded to the
        # full width rather than truncating the injection.
        sub = _make_substrate()
        spikes = sub.step(stimuli=np.ones(N_CORTICAL // 4))
        assert spikes.shape == (N_CORTICAL,)

    def test_run_returns_correct_shape(self):
        sub = _make_substrate()
        result = sub.run(duration=0.01, dt=0.001)
        assert result.shape == (10, N_CORTICAL)

    def test_run_with_stimuli_sequence(self):
        sub = _make_substrate()
        n_steps = 10
        stim = np.random.default_rng(0).uniform(5, 15, (n_steps, N_CORTICAL))
        result = sub.run(duration=0.01, dt=0.001, stimuli_sequence=stim)
        assert result.shape == (n_steps, N_CORTICAL)
