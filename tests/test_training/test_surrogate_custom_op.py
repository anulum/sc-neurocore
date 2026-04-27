# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for surrogate custom-op and legacy execution paths

from __future__ import annotations

import torch

from sc_neurocore.training.snn_modules import LIFCell
from sc_neurocore.training.surrogate import (
    SURROGATE_PATHS,
    atan_surrogate_custom_op,
    atan_surrogate_legacy,
    fast_sigmoid_custom_op,
    fast_sigmoid_legacy,
    sigmoid_surrogate_custom_op,
    sigmoid_surrogate_legacy,
    straight_through_custom_op,
    straight_through_legacy,
    superspike_custom_op,
    superspike_legacy,
    triangular_custom_op,
    triangular_legacy,
)


def _run_grad(
    fn,
    x: torch.Tensor,
    *args: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    probe = x.clone().requires_grad_(True)
    out = fn(probe, *args)
    out.sum().backward()
    return out.detach(), probe.grad.detach()


def test_surrogate_paths_are_explicit():
    assert SURROGATE_PATHS == ("custom_op", "legacy_autograd")


@torch.no_grad()
def test_custom_op_forward_matches_legacy_for_all_surrogates():
    x = torch.tensor([-1.5, -0.1, 0.0, 0.2, 3.0], dtype=torch.float32)

    pairs = [
        (fast_sigmoid_custom_op, fast_sigmoid_legacy, (25.0,)),
        (superspike_custom_op, superspike_legacy, (10.0,)),
        (atan_surrogate_custom_op, atan_surrogate_legacy, (2.0,)),
        (sigmoid_surrogate_custom_op, sigmoid_surrogate_legacy, (5.0,)),
        (straight_through_custom_op, straight_through_legacy, ()),
        (triangular_custom_op, triangular_legacy, (1.0,)),
    ]

    for custom_fn, legacy_fn, args in pairs:
        custom_out = custom_fn(x, *args)
        legacy_out = legacy_fn(x, *args)
        assert torch.equal(custom_out, legacy_out)


def test_custom_op_backward_matches_legacy_for_all_surrogates():
    x = torch.tensor([-0.75, -0.2, 0.1, 0.8], dtype=torch.float32)

    pairs = [
        (fast_sigmoid_custom_op, fast_sigmoid_legacy, (25.0,)),
        (superspike_custom_op, superspike_legacy, (10.0,)),
        (atan_surrogate_custom_op, atan_surrogate_legacy, (2.0,)),
        (sigmoid_surrogate_custom_op, sigmoid_surrogate_legacy, (5.0,)),
        (straight_through_custom_op, straight_through_legacy, ()),
        (triangular_custom_op, triangular_legacy, (1.0,)),
    ]

    for custom_fn, legacy_fn, args in pairs:
        custom_out, custom_grad = _run_grad(custom_fn, x, *args)
        legacy_out, legacy_grad = _run_grad(legacy_fn, x, *args)
        assert torch.equal(custom_out, legacy_out)
        assert torch.allclose(custom_grad, legacy_grad, atol=1e-7, rtol=0.0)


def test_lifcell_survives_torch_compile_with_custom_op_surrogate():
    cell = LIFCell(beta=0.9, threshold=1.0, surrogate_fn=atan_surrogate_custom_op)
    compiled = torch.compile(cell)

    current = torch.tensor([0.2, 0.9, 1.2], dtype=torch.float32, requires_grad=True)
    voltage = torch.zeros_like(current, requires_grad=True)

    eager_spike, eager_v = cell(current, voltage)
    eager_loss = eager_spike.sum() + eager_v.sum()
    eager_loss.backward()
    eager_current_grad = current.grad.detach().clone()
    eager_voltage_grad = voltage.grad.detach().clone()

    current.grad.zero_()
    voltage.grad.zero_()

    compiled_spike, compiled_v = compiled(current, voltage)
    compiled_loss = compiled_spike.sum() + compiled_v.sum()
    compiled_loss.backward()

    assert torch.allclose(compiled_spike, eager_spike)
    assert torch.allclose(compiled_v, eager_v)
    assert torch.allclose(current.grad, eager_current_grad, atol=1e-6, rtol=0.0)
    assert torch.allclose(voltage.grad, eager_voltage_grad, atol=1e-6, rtol=0.0)
