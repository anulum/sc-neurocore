# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — QCFS shared activation replacement fidelity

"""Exercise alias-preserving replacement through actual forward, training and conversion."""

from typing import cast

import pytest
import torch
from torch import nn

from sc_neurocore.conversion import convert, replace_relu_with_qcfs
from sc_neurocore.conversion.qcfs import QCFSActivation


@pytest.mark.parametrize("activation_type", [nn.ReLU, nn.ReLU6])
def test_direct_aliases_keep_one_learned_threshold_and_mode(
    activation_type: type[nn.ReLU] | type[nn.ReLU6],
) -> None:
    """Replace both registered names and train the shared activation exactly once."""
    activation = activation_type().eval()
    model = nn.Sequential(activation, activation).eval()
    returned = replace_relu_with_qcfs(model, T=4, learn_theta=True)
    assert returned is model
    assert model[0] is model[1]
    assert isinstance(model[0], QCFSActivation)
    assert not model.training and not model[0].training
    assert not any(isinstance(module, (nn.ReLU, nn.ReLU6)) for module in model.modules())
    parameters = list(model.parameters())
    assert len(parameters) == 1 and parameters[0] is model[0].theta
    values = torch.tensor([[0.625], [0.625]])
    expected = model[0](model[0](values))
    torch.testing.assert_close(model(values), expected, rtol=0, atol=0)
    optimizer = torch.optim.SGD(parameters, lr=0.1)
    before = parameters[0].detach().clone()
    loss = model(values).square().mean()
    loss.backward()
    assert parameters[0].grad is not None and float(parameters[0].grad) > 0
    optimizer.step()
    assert not torch.equal(parameters[0].detach(), before)
    assert model[0].theta is model[1].theta


class SharedParentNetwork(nn.Module):
    """Dense source with repeated parent and activation registration aliases."""

    def __init__(self) -> None:
        """Register two dense stages sharing one activation, with a parent alias."""
        super().__init__()
        activation = nn.ReLU()
        left = nn.Linear(1, 1, bias=False)
        right = nn.Linear(1, 1, bias=False)
        self.left: nn.Sequential = nn.Sequential(left, activation)
        self.right: nn.Sequential = nn.Sequential(right, activation)
        self.left_alias: nn.Sequential = self.left
        with torch.no_grad():
            left.weight.fill_(1)
            right.weight.fill_(1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Execute the parent alias and both actual shared activation invocations."""
        return cast(torch.Tensor, self.right(self.left_alias(values)))


def test_cross_parent_aliases_preserve_forward_and_conversion_topology() -> None:
    """Convert both actual QCFS invocations after replacing every registration alias."""
    model = SharedParentNetwork().eval()
    replace_relu_with_qcfs(model, T=4, theta=1.0, learn_theta=True)
    assert model.left_alias is model.left
    activation = model.left[1]
    assert isinstance(activation, QCFSActivation) and activation is model.right[1]
    assert not activation.training
    assert len(list(model.parameters())) == 3
    values = torch.tensor([[0.625], [0.875]])
    source = model(values).detach().numpy()
    target = convert(model, T=4)
    assert target.n_layers == 2
    assert target.layer_membrane_fractions == [0.5, 0.5]
    assert (
        target.rates(values.numpy(), input_mode="constant", backend="numpy").tobytes()
        == source.astype("float64").tobytes()
    )
