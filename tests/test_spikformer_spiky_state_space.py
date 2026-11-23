# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikyStateSpace from former test_spikformer.py

"""Focused suite: TestSpikyStateSpace from former test_spikformer.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.transformers.spikformer import (
    SpikyStateSpace,
)


class TestSpikyStateSpace:
    """The spiking state-space sequence model."""

    def test_step(self) -> None:
        """One step gives binary spikes and an output per channel."""
        ssm = SpikyStateSpace(d_model=8, d_state=16)
        x = np.random.rand(8)
        spikes, y = ssm.step(x)
        assert spikes.shape == (8,)
        assert y.shape == (8,)
        assert set(np.unique(spikes)).issubset({0.0, 1.0})

    def test_forward_sequence(self) -> None:
        """A sequence gives binary spikes per step."""
        ssm = SpikyStateSpace(d_model=8, d_state=16)
        x_seq = np.random.rand(20, 8)
        out = ssm.forward(x_seq)
        assert out.shape == (20, 8)
        assert set(np.unique(out)).issubset({0.0, 1.0})

    def test_reset(self) -> None:
        """A reset clears the hidden state and potential."""
        ssm = SpikyStateSpace(d_model=4, d_state=8)
        x = np.ones(4)
        ssm.step(x)
        assert not np.allclose(ssm._h, 0)
        ssm.reset()
        assert np.allclose(ssm._h, 0)
        assert np.allclose(ssm._v, 0)

    def test_state_accumulates(self) -> None:
        """Inputs build up the membrane potential."""
        ssm = SpikyStateSpace(d_model=4, d_state=8, threshold=100.0)
        # With high threshold, no spikes — membrane should accumulate
        for _ in range(10):
            ssm.step(np.ones(4))
        assert not np.allclose(ssm._v, 0)

    def test_different_dt(self) -> None:
        """A different time step changes the discretised dynamics."""
        ssm_fast = SpikyStateSpace(d_model=4, d_state=8, dt=0.1)
        ssm_slow = SpikyStateSpace(d_model=4, d_state=8, dt=0.001)
        # A values should differ (different decay rates)
        assert not np.allclose(ssm_fast.A, ssm_slow.A)

    def test_long_sequence(self) -> None:
        """A long sequence stays finite."""
        ssm = SpikyStateSpace(d_model=4, d_state=8)
        x_seq = np.random.rand(200, 4)
        out = ssm.forward(x_seq)
        assert out.shape == (200, 4)
        assert np.all(np.isfinite(out))
