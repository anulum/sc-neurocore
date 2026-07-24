# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSigmaDeltaEncoding from former test_model_sigma_delta.py

"""Focused suite: TestSigmaDeltaEncoding from former test_model_sigma_delta.py."""

from __future__ import annotations

from tests.model_sigma_delta_support import *  # noqa: F403


class TestSigmaDeltaEncoding:
    def test_spike_rate_equals_input_over_threshold(self):
        """For constant I ∈ (0, θ): rate = I/θ spikes per step."""
        n = SigmaDeltaNeuron(v_threshold=1.0)
        I = 0.3
        outputs = [n.step(I) for _ in range(10000)]
        pos = outputs.count(1)
        expected = 10000 * I / 1.0
        assert abs(pos - expected) <= 2, f"pos={pos}, expected={expected}"

    @pytest.mark.parametrize("I", [0.1, 0.25, 0.5, 0.75])
    def test_rate_proportional_to_input(self, I: float):
        """Rate = I/θ for I ∈ (0, θ) — exact for sigma-delta."""
        n = SigmaDeltaNeuron(v_threshold=1.0)
        outputs = [n.step(I) for _ in range(10000)]
        pos = outputs.count(1)
        expected = 10000 * I
        assert abs(pos - expected) <= 2

    def test_negative_input_produces_negative_spikes(self):
        n = SigmaDeltaNeuron()
        outputs = [n.step(-0.5) for _ in range(1000)]
        assert outputs.count(-1) == 500
        assert outputs.count(1) == 0

    def test_signal_reconstruction_bounded(self):
        """Cumulative output × θ tracks cumulative input within ±θ.

        This is the fundamental sigma-delta guarantee: the quantisation
        error (sigma residual) is always bounded by the threshold.
        """
        n = SigmaDeltaNeuron(v_threshold=1.0)
        I_signal = np.sin(np.arange(1000) * 0.05) * 0.4
        outputs = np.array([n.step(float(x)) for x in I_signal])
        cumsum_in = np.cumsum(I_signal)
        cumsum_out = np.cumsum(outputs) * n.v_threshold
        max_error = np.max(np.abs(cumsum_in - cumsum_out))
        assert max_error < n.v_threshold + 0.01, (
            f"Reconstruction error {max_error:.4f} exceeds threshold {n.v_threshold}"
        )

    def test_dc_removal(self):
        """For I=0, no spikes ever — perfect silence."""
        n = SigmaDeltaNeuron()
        outputs = [n.step(0.0) for _ in range(10000)]
        assert all(o == 0 for o in outputs)

    def test_bidirectional_encoding(self):
        """Alternating +/- input produces both +1 and -1 spikes."""
        n = SigmaDeltaNeuron(v_threshold=0.5)
        outputs = []
        for t in range(1000):
            I = 0.6 if t % 2 == 0 else -0.6
            outputs.append(n.step(I))
        assert 1 in outputs and -1 in outputs
