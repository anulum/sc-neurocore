# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public conversion calibration custody

"""Verify calibration preserves the actual Torch model's mode, hooks and RNG."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from sc_neurocore.conversion import convert


def test_calibration_uses_eval_and_restores_mixed_training_modes() -> None:
    """Calibrate inference activations without consuming dropout random numbers."""
    first_layer = torch.nn.Linear(3, 4).double()
    last_layer = torch.nn.Linear(4, 2).double()
    model = torch.nn.Sequential(
        first_layer,
        torch.nn.ReLU(),
        torch.nn.Dropout(0.75),
        last_layer,
        torch.nn.ReLU(),
    ).double()
    with torch.no_grad():
        assert first_layer.bias is not None and last_layer.bias is not None
        first_layer.weight.copy_(
            torch.tensor([[1.0, 0.5, 0.25], [0.25, 1.0, 0.5], [0.5, 0.25, 1.0], [0.75, 0.75, 0.75]])
        )
        first_layer.bias.copy_(torch.tensor([0.0, 0.1, 0.2, 0.3]))
        last_layer.weight.copy_(torch.tensor([[0.5, -0.25, 0.5, 0.25], [0.25, 0.5, 0.25, -0.25]]))
        last_layer.bias.copy_(torch.tensor([0.1, 0.2]))
    model[0].eval()
    model[3].eval()
    modes = [module.training for module in model.modules()]
    reference = deepcopy(model).eval()
    inputs = torch.arange(36, dtype=torch.float64).reshape(12, 3) / 36
    first = reference[:2](inputs).detach().numpy()
    last = reference(inputs).detach().numpy()
    scales = [max(float(np.percentile(values, 99.9)), 1e-6) for values in (first, last)]
    rng_before = torch.random.get_rng_state().clone()
    snn = convert(model, calibration_data=inputs)
    np.testing.assert_allclose(snn.weights[0], first_layer.weight.detach().numpy() / scales[0])
    np.testing.assert_allclose(
        snn.weights[1], last_layer.weight.detach().numpy() * scales[0] / scales[1]
    )
    assert [module.training for module in model.modules()] == modes
    torch.testing.assert_close(torch.random.get_rng_state(), rng_before)


def test_failed_calibration_removes_its_hooks_and_restores_modes() -> None:
    """A real incompatible matmul after ReLU cannot leave calibration hooks installed."""
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(3, 2))
    model[1].eval()
    modes = [module.training for module in model.modules()]
    original_hooks = dict(model[1]._forward_hooks)
    with pytest.raises(RuntimeError, match="mat1 and mat2"):
        convert(model, calibration_data=torch.ones((2, 3)))
    assert dict(model[1]._forward_hooks) == original_hooks
    assert [module.training for module in model.modules()] == modes


def test_successful_calibration_preserves_existing_user_hook() -> None:
    """Remove only converter-owned hooks and retain a real user observer."""
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU())
    observed: list[torch.Tensor] = []

    def observe(
        module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        """Retain outputs from the user's actual registered Torch hook."""
        observed.append(output.detach().clone())

    handle = model[1].register_forward_hook(observe)
    original_hooks = dict(model[1]._forward_hooks)
    convert(model, calibration_data=torch.ones((2, 3)))
    assert dict(model[1]._forward_hooks) == original_hooks
    model(torch.ones((2, 3)))
    assert len(observed) == 2
    handle.remove()


@pytest.mark.parametrize("percentile", [-1.0, 101.0, float("nan"), float("inf")])
def test_invalid_percentile_refused_before_model_changes(percentile: float) -> None:
    """Refuse an invalid statistic while preserving model and observer state."""
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU())
    with pytest.raises(ValueError, match="percentile"):
        convert(model, calibration_data=torch.ones((2, 3)), percentile=percentile)
    assert model.training and not model[1]._forward_hooks


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_calibration_input_refused(value: float) -> None:
    """Reject bad measured data instead of constructing NaN weight scales."""
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU())
    with pytest.raises(ValueError, match="nonempty and finite"):
        convert(model, calibration_data=torch.full((2, 3), value))
    assert model.training and not model[1]._forward_hooks


def test_empty_calibration_input_refused() -> None:
    """Require observations before estimating an activation percentile."""
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU())
    with pytest.raises(ValueError, match="nonempty and finite"):
        convert(model, calibration_data=torch.empty((0, 3)))


def test_nonfinite_activation_refused_after_hook_cleanup() -> None:
    """Actual overflow in a finite-input ANN cannot become a threshold report."""
    linear = torch.nn.Linear(3, 4)
    with torch.no_grad():
        linear.weight.fill_(torch.finfo(torch.float32).max)
    model = torch.nn.Sequential(linear, torch.nn.ReLU())
    with pytest.raises(ValueError, match="activation must be finite"):
        convert(model, calibration_data=torch.ones((2, 3)))
    assert model.training and not model[1]._forward_hooks


def test_converter_rejects_non_module_model() -> None:
    """Require a real Torch model at the public calibration boundary."""
    with pytest.raises(TypeError, match="PyTorch Module"):
        convert(object())


def test_converter_rejects_non_tensor_calibration_data() -> None:
    """Require the declared calibration tensor before invoking the model."""
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU())
    with pytest.raises(TypeError, match="PyTorch Tensor"):
        convert(model, calibration_data=[[1.0, 1.0, 1.0]])


def test_calibration_owns_activation_before_later_inplace_clipping() -> None:
    """Measure the observed ReLU, not storage changed by a later real operator."""
    from sc_neurocore.conversion.calibration import calibrate_activation_thresholds

    linear = torch.nn.Linear(2, 2, bias=False).double()
    with torch.no_grad():
        linear.weight.copy_(torch.eye(2, dtype=torch.float64))
    model = torch.nn.Sequential(linear, torch.nn.ReLU(), torch.nn.Hardtanh(0.0, 0.25, inplace=True))
    inputs = torch.tensor([[0.25, 0.5], [0.75, 1.0]], dtype=torch.float64)
    thresholds = calibrate_activation_thresholds(model, inputs, percentile=100.0)
    assert thresholds == [1.0]
    torch.testing.assert_close(model(inputs), torch.full_like(inputs, 0.25))
    assert model.training


@pytest.mark.parametrize(
    "data,percentile",
    [
        (torch.ones((1, 1)), float("nan")),
        (torch.empty((0, 1)), 100.0),
        (torch.full((1, 1), float("nan")), 100.0),
    ],
)
def test_general_public_calibrator_refuses_invalid_observations(
    data: torch.Tensor, percentile: float
) -> None:
    """General direct calibration preserves modes when no usable statistic exists."""
    from sc_neurocore.conversion.calibration import calibrate_activation_thresholds

    model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.ReLU())
    with pytest.raises(ValueError):
        calibrate_activation_thresholds(model, data, percentile)
    assert model.training


def test_general_public_calibrator_restores_source_after_real_failure() -> None:
    """Concrete-data calibration retains its mode/hook custody on matmul failure."""
    from sc_neurocore.conversion.calibration import calibrate_activation_thresholds

    model = torch.nn.Sequential(torch.nn.Linear(1, 2), torch.nn.ReLU(), torch.nn.Linear(1, 1))
    model[1].eval()
    with pytest.raises(RuntimeError, match="mat1 and mat2"):
        calibrate_activation_thresholds(model, torch.ones((1, 1)))
    assert model.training and not model[1].training and not model[1]._forward_hooks


def test_general_public_calibrator_refuses_real_nonfinite_activation() -> None:
    """Reject actual finite-input source overflow through the general public API."""
    from sc_neurocore.conversion.calibration import calibrate_activation_thresholds

    linear = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        linear.weight.fill_(torch.finfo(torch.float32).max)
    model = torch.nn.Sequential(linear, torch.nn.ReLU())
    with pytest.raises(ValueError, match="activation must be finite"):
        calibrate_activation_thresholds(model, torch.ones((1, 2)))
    assert model.training and not model[1]._forward_hooks
