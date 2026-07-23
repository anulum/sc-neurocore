# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDifferentiableDenseLayer from former test_surrogate_python.py

"""Focused suite: TestDifferentiableDenseLayer from former test_surrogate_python.py."""

from __future__ import annotations

from tests.surrogate_python_support import *  # noqa: F403

class TestDifferentiableDenseLayer:
    def test_train_step(self):
        layer = DifferentiableDenseLayer(
            n_inputs=8,
            n_neurons=4,
            length=1024,
            surrogate="fast_sigmoid",
            k=5.0,
        )
        out1 = np.array(layer.forward([0.5] * 8))

        grad_in, grad_w = layer.backward([1.0] * 4)
        assert grad_in.shape == (8,)
        assert grad_w.shape == (4, 8)
        layer.update_weights(grad_w, lr=0.5)

        out2 = np.array(layer.forward([0.5] * 8))
        assert not np.allclose(out1, out2)
