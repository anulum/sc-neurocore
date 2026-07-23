# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPOSSMDecoder from former test_neural_decoders.py

"""Focused suite: TestPOSSMDecoder from former test_neural_decoders.py."""

from __future__ import annotations

from tests.neural_decoders_support import *  # noqa: F403

class TestPOSSMDecoder:
    def test_defaults(self) -> None:
        dec = POSSMDecoder()
        assert dec.d_model == 64
        assert dec.d_state == 32

    def test_discretise_zoh(self) -> None:
        """A_bar = exp(dt * A) for diagonal SSM."""
        dec = POSSMDecoder(d_model=4, d_state=2, dt=0.1, seed=1)
        a_bar, b_bar = dec.discretise(0.1)
        expected_a = np.exp(0.1 * dec._A)
        np.testing.assert_allclose(a_bar, expected_a)

    def test_step_output_shape(self) -> None:
        dec = POSSMDecoder(d_model=8, d_state=4)
        x = np.ones(8)
        y = dec.step(x)
        assert y.shape == (8,)

    def test_step_state_changes(self) -> None:
        dec = POSSMDecoder(d_model=4, d_state=2, seed=3)
        h_before = dec._h.copy()
        dec.step(np.ones(4))
        assert not np.allclose(dec._h, h_before)

    def test_encode_causal_empty(self) -> None:
        dec = POSSMDecoder(d_model=8)
        out = dec.encode_causal([])
        assert out.shape == (0, 8)

    def test_encode_causal_shape(self) -> None:
        dec = POSSMDecoder(d_model=16, d_state=8)
        trains = [np.zeros(50), np.zeros(50)]
        trains[0][10] = 1
        trains[1][20] = 1
        out = dec.encode_causal(trains)
        assert out.shape == (50, 16)

    def test_causal_no_future_leakage(self) -> None:
        """Output at time t must not depend on spikes at t+k."""
        dec = POSSMDecoder(d_model=8, d_state=4, seed=5)
        t1 = np.zeros(30)
        t1[5] = 1
        out1 = dec.encode_causal([t1])
        # Add future spike at t=25
        t2 = t1.copy()
        t2[25] = 1
        dec.reset()
        out2 = dec.encode_causal([t2])
        # Outputs before t=25 must be identical (causal)
        np.testing.assert_allclose(out1[:25], out2[:25])

    def test_reset_zeros_state(self) -> None:
        dec = POSSMDecoder(d_model=4, d_state=2)
        dec.step(np.ones(4))
        dec.reset()
        assert np.allclose(dec._h, 0.0)

    def test_oscillatory_dynamics(self) -> None:
        """Complex diagonal A produces oscillatory hidden state."""
        dec = POSSMDecoder(d_model=4, d_state=4, dt=0.01, seed=10)
        x = np.array([1.0, 0.0, 0.0, 0.0])
        dec.step(x)
        h1 = dec._h.copy()
        for _ in range(100):
            dec.step(np.zeros(4))
        h2 = dec._h
        # Imaginary parts of A cause oscillation → h should not converge to 0
        # but should decay (real part is -0.5)
        assert np.linalg.norm(h2) < np.linalg.norm(h1)
        assert np.linalg.norm(h2) > 0
