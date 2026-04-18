# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for learnable delay training modules

"""Tests for DelayLinear learnable-delay SNN training module."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import numpy as np

from sc_neurocore.training.delay_linear import DelayLinear


class TestDelayLinear:
    def test_shapes(self):
        layer = DelayLinear(4, 3, max_delay=8)
        assert layer.weight.shape == (3, 4)
        assert layer.delay.shape == (3, 4)

    def test_step_produces_output(self):
        layer = DelayLinear(4, 3, max_delay=8, init_delay=1.0)
        layer.reset()
        spikes = torch.tensor([1.0, 0.0, 1.0, 0.0])
        # At t=0 inject spikes, at t=1 delayed output should arrive
        layer.step(spikes)
        out = layer.step(torch.zeros(4))
        assert out.shape == (3,)
        assert out.abs().sum() > 0

    def test_delay_postpones_signal(self):
        layer = DelayLinear(2, 1, max_delay=8, init_delay=3.0)
        layer.reset()
        spikes = torch.tensor([1.0, 1.0])
        # Inject at t=0
        outputs = []
        for t in range(6):
            out = layer.step(spikes if t == 0 else torch.zeros(2))
            outputs.append(out.item())
        # Output should be near zero at t=0,1,2 and nonzero at t=3
        assert abs(outputs[0]) < 1e-6
        assert abs(outputs[1]) < 1e-6
        assert abs(outputs[3]) > 1e-6

    def test_gradient_flows_through_delays(self):
        layer = DelayLinear(4, 3, max_delay=8, init_delay=2.5)
        layer.reset()
        spikes = torch.tensor([1.0, 0.0, 1.0, 0.0])
        for t in range(4):
            out = layer.step(spikes if t == 0 else torch.zeros(4))
        loss = out.sum()
        loss.backward()
        assert layer.delay.grad is not None
        assert layer.delay.grad.norm() > 0

    def test_gradient_flows_through_weights(self):
        layer = DelayLinear(4, 3, max_delay=8, init_delay=1.0)
        layer.reset()
        out = layer.step(torch.ones(4))
        out = layer.step(torch.zeros(4))
        loss = out.sum()
        loss.backward()
        assert layer.weight.grad is not None
        assert layer.weight.grad.norm() > 0

    def test_delays_clamp_to_valid_range(self):
        layer = DelayLinear(2, 2, max_delay=4, init_delay=0.0)
        with torch.no_grad():
            layer.delay[0, 0] = -5.0
            layer.delay[1, 1] = 100.0
        layer.reset()
        out = layer.step(torch.ones(2))
        assert not torch.isnan(out).any()

    def test_reset_clears_history(self):
        layer = DelayLinear(3, 2, max_delay=4, init_delay=1.0)
        layer.reset()
        layer.step(torch.ones(3))
        layer.reset()
        out = layer.step(torch.zeros(3))
        # After reset, no history so delayed output should be zero
        assert out.abs().sum() < 1e-6

    def test_batch_input(self):
        layer = DelayLinear(4, 3, max_delay=4, init_delay=1.0)
        layer.reset()
        batch = torch.randn(8, 4)
        out = layer.step(batch)
        assert out.shape == (8, 3)

    def test_delays_int_export(self):
        layer = DelayLinear(3, 2, max_delay=8, init_delay=3.7)
        d = layer.delays_int
        assert d.shape == (2, 3)
        assert (d == 4).all()

    def test_nir_delay_array(self):
        layer = DelayLinear(3, 2, max_delay=8, init_delay=2.0)
        arr = layer.to_nir_delay_array()
        assert arr.shape == (6,)
        assert arr.dtype == np.float64
        assert (arr == 2.0).all()

    def test_no_learn_delay(self):
        layer = DelayLinear(3, 2, max_delay=4, learn_delay=False, init_delay=1.0)
        assert not isinstance(layer.delay, torch.nn.Parameter)
        # Should still work
        layer.reset()
        out = layer.step(torch.ones(3))
        assert out.shape == (2,)

    def test_delay_optimization_changes_value(self):
        """Verify optimizer can modify delay values via gradient."""
        torch.manual_seed(42)
        layer = DelayLinear(2, 1, max_delay=8, init_delay=2.5)
        with torch.no_grad():
            layer.weight.fill_(1.0)
        init_delay = layer.delay.clone().detach()
        optimizer = torch.optim.SGD([layer.delay], lr=1.0)

        for epoch in range(20):
            layer.reset()
            layer.step(torch.ones(2))
            for _ in range(2):
                out = layer.step(torch.zeros(2))
            loss = out.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_delay = layer.delay.detach()
        changed = (final_delay - init_delay).abs().sum().item()
        assert changed > 0.01, f"Delay should change during optimization, delta={changed:.6f}"
