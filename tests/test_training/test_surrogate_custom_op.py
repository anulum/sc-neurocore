# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for surrogate custom-op and compiled execution paths

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import warnings

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

SurrogateCall = Callable[..., torch.Tensor]


REPO_ROOT = Path(__file__).resolve().parents[2]
SURROGATE_SOURCE = REPO_ROOT / "src" / "sc_neurocore" / "training" / "surrogate.py"
TEST_SOURCE = Path(__file__).resolve()
SCRIPT_MODULE_TOKEN = "torch.jit." + "ScriptModule"
SCRIPT_METHOD_TOKEN = "torch.jit." + "script_method"
SCRIPT_METHOD_WARNING = f"`{SCRIPT_METHOD_TOKEN}` is deprecated"


def _checked_grad(tensor: torch.Tensor) -> torch.Tensor:
    """Return a populated gradient tensor after an explicit backward pass."""
    grad = tensor.grad
    assert grad is not None
    return grad


def _run_grad(
    fn: SurrogateCall,
    x: torch.Tensor,
    *args: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return detached forward output and input gradient for one surrogate call."""
    probe = x.clone().requires_grad_(True)
    out = fn(probe, *args)
    torch.autograd.backward(out.sum())
    return out.detach(), _checked_grad(probe).detach()


def test_surrogate_paths_are_explicit() -> None:
    """The training surface keeps modern and legacy paths named explicitly."""
    assert SURROGATE_PATHS == ("custom_op", "legacy_autograd")


@torch.no_grad()
def test_custom_op_forward_matches_legacy_for_all_surrogates() -> None:
    """Custom operators keep the same Heaviside forward pass as legacy autograd."""
    x = torch.tensor([-1.5, -0.1, 0.0, 0.2, 3.0], dtype=torch.float32)

    pairs: tuple[tuple[SurrogateCall, SurrogateCall, tuple[float, ...]], ...] = (
        (fast_sigmoid_custom_op, fast_sigmoid_legacy, (25.0,)),
        (superspike_custom_op, superspike_legacy, (10.0,)),
        (atan_surrogate_custom_op, atan_surrogate_legacy, (2.0,)),
        (sigmoid_surrogate_custom_op, sigmoid_surrogate_legacy, (5.0,)),
        (straight_through_custom_op, straight_through_legacy, ()),
        (triangular_custom_op, triangular_legacy, (1.0,)),
    )

    for custom_fn, legacy_fn, args in pairs:
        custom_out = custom_fn(x, *args)
        legacy_out = legacy_fn(x, *args)
        assert torch.equal(custom_out, legacy_out)


def test_custom_op_backward_matches_legacy_for_all_surrogates() -> None:
    """Custom operators keep the same surrogate gradients as legacy autograd."""
    x = torch.tensor([-0.75, -0.2, 0.1, 0.8], dtype=torch.float32)

    pairs: tuple[tuple[SurrogateCall, SurrogateCall, tuple[float, ...]], ...] = (
        (fast_sigmoid_custom_op, fast_sigmoid_legacy, (25.0,)),
        (superspike_custom_op, superspike_legacy, (10.0,)),
        (atan_surrogate_custom_op, atan_surrogate_legacy, (2.0,)),
        (sigmoid_surrogate_custom_op, sigmoid_surrogate_legacy, (5.0,)),
        (straight_through_custom_op, straight_through_legacy, ()),
        (triangular_custom_op, triangular_legacy, (1.0,)),
    )

    for custom_fn, legacy_fn, args in pairs:
        custom_out, custom_grad = _run_grad(custom_fn, x, *args)
        legacy_out, legacy_grad = _run_grad(legacy_fn, x, *args)
        assert torch.equal(custom_out, legacy_out)
        assert torch.allclose(custom_grad, legacy_grad, atol=1e-7, rtol=0.0)


def test_surrogate_compile_lane_has_no_torchscript_script_method() -> None:
    """The touched training lane no longer depends on deprecated ScriptModule APIs."""
    source_text = "\n".join(
        (
            SURROGATE_SOURCE.read_text(encoding="utf-8"),
            TEST_SOURCE.read_text(encoding="utf-8"),
        )
    )

    assert SCRIPT_MODULE_TOKEN not in source_text
    assert SCRIPT_METHOD_TOKEN not in source_text


def test_lifcell_survives_torch_compile_eager_backend_without_script_method_warning() -> None:
    """Dynamo eager backend preserves custom-op gradients without TorchScript warnings."""
    cell = LIFCell(beta=0.9, threshold=1.0, surrogate_fn=atan_surrogate_custom_op)
    compiled = torch.compile(cell, backend="eager")

    current = torch.tensor([0.2, 0.9, 1.2], dtype=torch.float32, requires_grad=True)
    voltage = torch.zeros_like(current, requires_grad=True)

    eager_spike, eager_v = cell(current, voltage)
    eager_loss = eager_spike.sum() + eager_v.sum()
    eager_loss.backward()
    current_grad = _checked_grad(current)
    voltage_grad = _checked_grad(voltage)
    eager_current_grad = current_grad.detach().clone()
    eager_voltage_grad = voltage_grad.detach().clone()

    current_grad.zero_()
    voltage_grad.zero_()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compiled_spike, compiled_v = compiled(current, voltage)
        compiled_loss = compiled_spike.sum() + compiled_v.sum()
        compiled_loss.backward()

    assert torch.allclose(compiled_spike, eager_spike)
    assert torch.allclose(compiled_v, eager_v)
    assert torch.allclose(_checked_grad(current), eager_current_grad, atol=1e-6, rtol=0.0)
    assert torch.allclose(_checked_grad(voltage), eager_voltage_grad, atol=1e-6, rtol=0.0)
    assert all(SCRIPT_METHOD_WARNING not in str(warning.message) for warning in caught)
