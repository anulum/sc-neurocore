# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source execution topology through the public converter

"""Export actual forward invocations rather than registered-module traversal."""

import numpy as np
import pytest
import torch
from torch import nn

from sc_neurocore.conversion import QCFSActivation, convert


class ReorderedNetwork(nn.Module):
    """Register unused and reversed layers while executing a simple dense chain."""

    def __init__(self) -> None:
        """Create exact dyadic coefficients with deliberately misleading registration."""
        super().__init__()
        self.unused = nn.Linear(7, 9).double()
        self.last = nn.Linear(2, 1, bias=False).double()
        self.first = nn.Linear(2, 2, bias=False).double()
        self.activation = QCFSActivation(T=8, theta=1.0).double()
        with torch.no_grad():
            self.first.weight.copy_(torch.eye(2, dtype=torch.float64))
            self.last.weight.copy_(torch.tensor([[0.5, -0.25]], dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute only the first, activation and final linear layer."""
        output: torch.Tensor = self.last(self.activation(self.first(x)))
        return output


class ReusedNetwork(nn.Module):
    """Invoke a shared weighted module twice at distinct graph positions."""

    def __init__(self) -> None:
        """Use one identity matrix for two QCFS stages."""
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False).double()
        self.activation = QCFSActivation(T=8, theta=1.0).double()
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(2, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute the same module objects twice instead of deduplicating them."""
        output: torch.Tensor = self.activation(self.linear(self.activation(self.linear(x))))
        return output


class FunctionalReluNetwork(nn.Module):
    """Use a real functional activation absent from model.modules()."""

    def __init__(self) -> None:
        """Create an identity affine map with a Torch functional ReLU."""
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False).double()
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(2, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return activation output, whose scale must come from real execution."""
        return torch.relu(self.linear(x))


def test_forward_order_unused_parameters_and_final_signed_readout() -> None:
    """Export only executed layers and preserve the actual final operation."""
    model = ReorderedNetwork()
    inputs = np.array([[0.25, 0.75], [0.75, 0.25]])
    snn = convert(model)
    assert snn.n_layers == 2 and snn.output_mode == "linear"
    np.testing.assert_array_equal(
        snn.rates(inputs, input_mode="constant"), model(torch.from_numpy(inputs)).detach().numpy()
    )


def test_repeated_modules_preserve_each_invocation() -> None:
    """A shared layer retains both forward positions and complete source behavior."""
    model = ReusedNetwork()
    snn = convert(model)
    assert snn.n_layers == 2
    inputs = np.array([[0.25, 0.75]])
    np.testing.assert_array_equal(
        snn.rates(inputs, input_mode="constant"), model(torch.from_numpy(inputs)).detach().numpy()
    )


def test_functional_activation_calibrates_actual_output_node() -> None:
    """Associate a functional ReLU with its preceding weights and final output."""
    model = FunctionalReluNetwork()
    snn = convert(
        model, calibration_data=torch.tensor([[2.0, 1.0]], dtype=torch.float64), percentile=100.0
    )
    assert snn.output_mode == "spikes" and snn.output_scale == 2.0
    np.testing.assert_array_equal(snn.weights[0], np.eye(2) / 2)


def test_consecutive_affine_layers_preserve_signed_analog_source() -> None:
    """Lower an affine composition without inventing an intermediate IF nonlinearity."""
    first = nn.Linear(2, 2).double()
    last = nn.Linear(2, 1).double()
    with torch.no_grad():
        first.weight.copy_(torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float64))
        assert first.bias is not None and last.bias is not None
        first.bias.copy_(torch.tensor([0.25, -0.25], dtype=torch.float64))
        last.weight.copy_(torch.tensor([[0.5, -0.25]], dtype=torch.float64))
        last.bias.fill_(-0.125)
    source = nn.Sequential(first, last)
    inputs = np.array([[0.0, 1.0], [1.0, 0.0]])
    snn = convert(source, T=8)
    np.testing.assert_array_equal(
        snn.rates(inputs, input_mode="constant"), source(torch.from_numpy(inputs)).detach().numpy()
    )


def test_unsupported_residual_operation_cannot_silently_become_dense_chain() -> None:
    """Refuse a real additive branch that the dense target cannot represent."""

    class Residual(nn.Module):
        """Execute an additive skip alongside a weighted branch."""

        def __init__(self) -> None:
            """Create a weighted identity-size branch."""
            super().__init__()
            self.linear = nn.Linear(2, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Add the original input instead of returning the linear layer alone."""
            branch: torch.Tensor = self.linear(x)
            return branch + x

    with pytest.raises(ValueError, match="dense.*operation"):
        convert(Residual())


def test_mixed_relu_qcfs_maps_scales_and_preloads_by_invocation() -> None:
    """A calibrated ReLU and learned QCFS retain their own scale and IF shift."""
    first = nn.Linear(1, 1, bias=False).double()
    last = nn.Linear(1, 1, bias=False).double()
    with torch.no_grad():
        first.weight.fill_(1.0)
        last.weight.fill_(1.0)
    source = nn.Sequential(first, nn.ReLU(), last, QCFSActivation(T=8, theta=4.0).double())
    snn = convert(
        source, calibration_data=torch.tensor([[2.0]], dtype=torch.float64), percentile=100
    )
    assert snn.layer_membrane_fractions == [0.0, 0.5]
    np.testing.assert_array_equal(snn.weights[0], [[0.5]])
    np.testing.assert_array_equal(snn.weights[1], [[0.5]])
    inputs = np.array([[1.0]])
    np.testing.assert_array_equal(
        snn.rates(inputs, input_mode="constant"), source(torch.from_numpy(inputs)).detach().numpy()
    )


def test_relu_six_retains_its_declared_saturation_scale() -> None:
    """A clipped ReLU cannot silently become unit-scale unbounded ReLU metadata."""
    linear = nn.Linear(1, 1).double()
    with torch.no_grad():
        linear.weight.zero_()
        assert linear.bias is not None
        linear.bias.fill_(7.0)
    source = nn.Sequential(linear, nn.ReLU6())
    snn = convert(source, T=8)
    assert snn.output_scale == 6.0
    np.testing.assert_array_equal(snn.rates([0.0], input_mode="constant"), [6.0])


def test_trace_restores_real_rngs_and_does_not_mutate_source_buffer() -> None:
    """Actual user-forward state changes occur only on the independent source copy."""
    import random

    class Stateful(ReusedNetwork):
        """Consume real CPU random streams and update a buffer during forward."""

        counter: torch.Tensor

        def __init__(self) -> None:
            """Register observable source state before its first invocation."""
            super().__init__()
            self.register_buffer("counter", torch.zeros(1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Use the same output recurrence while exercising trace-side custody."""
            self.counter.add_(1)
            random.random()
            np.random.random()
            torch.rand(1)
            return super().forward(x)

    source = Stateful()
    source.linear.eval()
    modes = [module.training for module in source.modules()]
    cpu_state = torch.random.get_rng_state().clone()
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    convert(source)
    assert [module.training for module in source.modules()] == modes
    torch.testing.assert_close(source.counter, torch.zeros(1))
    torch.testing.assert_close(torch.random.get_rng_state(), cpu_state)
    actual_numpy = np.random.get_state()
    assert actual_numpy[0] == numpy_state[0] and actual_numpy[2:] == numpy_state[2:]
    np.testing.assert_array_equal(actual_numpy[1], numpy_state[1])
    assert random.getstate() == python_state


def test_data_dependent_forward_is_explicitly_untraceable() -> None:
    """Refuse input-dependent Python branching with source state retained."""

    class Dynamic(FunctionalReluNetwork):
        """Choose a branch using actual tensor values."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Expose the unsupported dynamic decision to symbolic tracing."""
            if x.sum() > 0:
                return super().forward(x)
            return -x

    source = Dynamic()
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match="data-dependent"):
        convert(source)
    assert source.training
    torch.testing.assert_close(torch.random.get_rng_state(), rng)


@pytest.mark.parametrize("budgets", [(4, 8), (8, 4)])
def test_mixed_qcfs_budgets_require_explicit_simulation_budget(budgets: tuple[int, int]) -> None:
    """Do not silently pick one trained lattice when source activations differ."""
    source = nn.Sequential(
        nn.Linear(1, 1), QCFSActivation(T=budgets[0]), nn.Linear(1, 1), QCFSActivation(T=budgets[1])
    )
    with pytest.raises(ValueError, match="explicit T"):
        convert(source)
    assert convert(source, T=16).T == 16
