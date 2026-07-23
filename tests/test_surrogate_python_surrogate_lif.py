# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSurrogateLif from former test_surrogate_python.py

"""Focused suite: TestSurrogateLif from former test_surrogate_python.py."""

from __future__ import annotations

from tests.surrogate_python_support import *  # noqa: F403

class TestSurrogateLif:
    def test_forward_backward_cycle(self):
        lif = SurrogateLif(surrogate="fast_sigmoid", k=25.0)
        _spike, _v = lif.forward(leak_k=20, gain_k=256, i_t=128)
        grad = lif.backward(1.0)
        assert isinstance(grad, float)
        assert grad != 0.0

    def test_clear_trace(self):
        lif = SurrogateLif(surrogate="arctan", k=10.0)
        for _ in range(10):
            lif.forward(20, 256, 128)
        assert lif.trace_len() == 10
        lif.clear_trace()
        assert lif.trace_len() == 0
