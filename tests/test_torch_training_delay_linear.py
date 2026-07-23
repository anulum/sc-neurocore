# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDelayLinear from former test_torch_training.py

"""Focused suite: TestDelayLinear from former test_torch_training.py."""

from __future__ import annotations

from tests.torch_training_support import *  # noqa: F403

class TestDelayLinear:
    def test_forward_shape(self):
        dl = DelayLinear(in_features=16, out_features=8, max_delay=4)
        x = torch.randn(4, 16)
        out = dl.step(x)
        assert out.shape == (4, 8)

    def test_unbatched(self):
        dl = DelayLinear(in_features=8, out_features=4, max_delay=3)
        x = torch.randn(8)
        out = dl.step(x)
        assert out.shape == (4,)

    def test_delays_int(self):
        dl = DelayLinear(in_features=8, out_features=4, max_delay=8)
        d = dl.delays_int
        assert d.shape == (4, 8)
        assert (d >= 0).all()
        assert (d <= 8).all()

    def test_nir_export(self):
        dl = DelayLinear(in_features=4, out_features=2, max_delay=3)
        arr = dl.to_nir_delay_array()
        assert arr.shape == (8,)  # 2 * 4

    def test_delay_gradient(self):
        dl = DelayLinear(in_features=4, out_features=2, max_delay=4, learn_delay=True)
        dl.reset()
        x = torch.randn(4, requires_grad=True)
        dl.step(x)
        out = dl.step(x)
        out.sum().backward()
        assert dl.delay.grad is not None

    def test_reset(self):
        dl = DelayLinear(in_features=4, out_features=2, max_delay=3)
        dl.step(torch.ones(4))
        dl.reset()
        assert dl._t == 0
        assert dl._history.abs().sum().item() == 0.0

    def test_multi_timestep_sequence(self):
        dl = DelayLinear(in_features=4, out_features=2, max_delay=3, learn_delay=False)
        dl.reset()
        outputs = []
        for t in range(10):
            x = torch.randn(4)
            outputs.append(dl.step(x))
        assert len(outputs) == 10
