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

    def test_kernel_centre_positive(self) -> None:
        """At x=0: w = A - B = 1.5 - 0.75 = 0.75."""
        n = AmariNeuralField()
        assert abs(n._w[0] - (n.a_exc - n.b_inh)) < 1e-10

    def test_kernel_shape(self) -> None:
        n = AmariNeuralField()
        assert n._w.shape == (n.n,)

    def test_kernel_is_distally_inhibitory(self) -> None:
        """The farthest periodic interaction is inhibitory, as the source requires."""
        n = AmariNeuralField()
        assert n._w[n.n // 2] < 0.0

    def test_non_lateral_kernel_is_rejected(self) -> None:
        """Do not silently admit the historical all-excitatory width ordering."""
        with pytest.raises(ValueError, match="distally inhibitory"):
            AmariNeuralField(a_width=1.0, b_width=2.0)

    def test_circular_interaction_correct(self) -> None:
        """The circular interaction responds to one active source site."""
        n = AmariNeuralField(n=8)
        # Set f(u) = delta at centre
        n.u = np.zeros(8)
        n.u[4] = 1.0
        # After one step with zero input: u gets the circular kernel contribution
        n.step(np.zeros(8))
        # u should have changed (kernel convolved with delta → kernel itself)
        assert not np.allclose(n.u, 0.0)
