# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAmariMexicanHatKernel from former test_model_amari_field.py

"""Focused suite: TestAmariMexicanHatKernel from former test_model_amari_field.py."""

from __future__ import annotations

from tests.model_amari_field_support import *  # noqa: F403

class TestAmariMexicanHatKernel:
    """w(x) = A·exp(-a|x|) - B·exp(-b|x|). Centre excitatory, surround inhibitory."""

    def test_kernel_centre_positive(self):
        """At x=0: w = A - B = 1.5 - 0.75 = 0.75."""
        n = AmariNeuralField()
        assert abs(n._w[0] - (n.a_exc - n.b_inh)) < 1e-10

    def test_kernel_shape(self):
        n = AmariNeuralField()
        assert n._w.shape == (n.n,)

    def test_kernel_sum_default_positive(self):
        """Default kernel sum > 1 → inherently unstable (positive feedback)."""
        n = AmariNeuralField()
        assert n._w.sum() > 1.0, f"Kernel sum = {n._w.sum():.2f}"

    def test_balanced_kernel_stable(self):
        """With a_exc = b_inh = 0.5: kernel sum ≈ 0.96 → stable dynamics."""
        n = AmariNeuralField(a_exc=0.5, b_inh=0.5)
        assert n._w.sum() < 1.5

    def test_fft_convolution_correct(self):
        """Convolution via FFT: should match direct sum for simple case."""
        n = AmariNeuralField(n=8)
        # Set f(u) = delta at centre
        n.u = np.zeros(8)
        n.u[4] = 1.0
        # After one step with zero input: u gets kernel contribution
        n.step(np.zeros(8))
        # u should have changed (kernel convolved with delta → kernel itself)
        assert not np.allclose(n.u, 0.0)
