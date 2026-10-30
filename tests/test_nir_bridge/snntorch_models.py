# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Shared real snnTorch inference models

"""Construct actual framework models shared by inference and refusal contracts."""

from typing import Any

import snntorch as snn
import torch
from torch import nn


def make_model(
    kind: str = "synaptic", reset: str = "subtract", vector: bool = False, disabled: bool = False
) -> Any:
    """Build a real multilayer model in the qualified inference profile.

    Parameters
    ----------
    kind : str
        Leaky or synaptic cell family.
    reset : str
        Immediate subtract or zero reset convention.
    vector : bool
        Use heterogeneous per-neuron decay and threshold parameters.
    disabled : bool
        Disable the surrogate-gradient calculation.

    Returns
    -------
    torch.nn.Sequential
        Three float64 affine/cell pairs with deterministic independent weights.
    """
    torch.manual_seed(142)
    cell_type = snn.Synaptic if kind == "synaptic" else snn.Leaky
    layers = []
    for before, after in [(4, 8), (8, 5), (5, 3)]:
        layers.append(nn.Linear(before, after, dtype=torch.float64))
        options = {
            "surrogate_disable": disabled,
            "beta": torch.linspace(0.5, 0.9, after, dtype=torch.float64) if vector else 0.8,
            "threshold": torch.linspace(0.6, 1.1, after, dtype=torch.float64) if vector else 0.8,
            "init_hidden": True,
            "reset_delay": False,
            "reset_mechanism": reset,
        }
        if kind == "synaptic":
            options["alpha"] = (
                torch.linspace(0.2, 0.85, after, dtype=torch.float64) if vector else 0.7
            )
        cell = cell_type(**options)
        # Vector unity grading keeps snnTorch binary spike outputs in float64;
        # its default scalar grading leaves float32 spikes even after double().
        cell.graded_spikes_factor = torch.ones(after, dtype=torch.float64)
        layers.append(cell)
    model = nn.Sequential(*layers).double()
    with torch.no_grad():
        for layer in model:
            if isinstance(layer, nn.Linear):
                layer.weight.abs_().mul_(1.3)
                layer.bias.fill_(0.1)
    return model
