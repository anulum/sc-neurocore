# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Inference dropout compatibility through public conversion

"""Admit actual inference identities and refuse unrepresented randomness."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from sc_neurocore.conversion import convert


@pytest.mark.parametrize(
    "dropout",
    [nn.Dropout1d(), nn.Dropout2d(), nn.Dropout3d(), nn.AlphaDropout(), nn.FeatureAlphaDropout()],
)
def test_inference_dropout_variants_are_actual_identity_operators(dropout: nn.Module) -> None:
    """Recognize Torch's inference identities without adding a spiking output."""
    source = nn.Sequential(nn.Linear(2, 2), dropout)
    assert convert(source).output_mode == "linear"


@pytest.mark.parametrize(
    "training,probability,accepted", [(False, 0.5, True), (True, 0.0, True), (True, 0.5, False)]
)
def test_functional_dropout_preserves_declared_inference_semantics(
    training: bool, probability: float, accepted: bool
) -> None:
    """Admit actual no-op dropout and refuse an unrepresented random transform."""

    class Functional(nn.Module):
        """Execute functional dropout using fixed source call arguments."""

        def __init__(self) -> None:
            """Create a genuine weighted source branch."""
            super().__init__()
            self.linear = nn.Linear(2, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Apply real Torch functional dropout to the weighted source output."""
            output: torch.Tensor = self.linear(x)
            return F.dropout(output, probability, training)

    source = Functional()
    if accepted:
        assert convert(source).output_mode == "linear"
    else:
        with pytest.raises(ValueError, match="dense.*operation"):
            convert(source)
