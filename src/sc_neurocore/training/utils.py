# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SNN training utilities

"""SNN training utilities: spike monitors, reset, population readout."""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn


def reset_states(monitors: list[Any] | None = None) -> None:
    """Clear SpikeMonitor recorded data.

    Parameters
    ----------
    monitors : list of SpikeMonitor, optional
        Monitors to reset. If None, does nothing.

    Note: this does NOT reset membrane voltages or adaptation variables.
    To reset neuron state, re-initialize the model or call forward()
    with fresh zero-initialized hidden states. To reset a single
    monitor, call ``monitor.reset()`` directly.
    """
    if monitors is None:
        return
    for mon in monitors:
        if hasattr(mon, "reset"):
            mon.reset()


class SpikeMonitor:
    """Record spikes per layer during forward pass.

    Attach to a SpikingNet or ConvSpikingNet to capture spike activity
    at each layer per timestep. Useful for raster plots and diagnostics.

        monitor = SpikeMonitor(model)
        spk, mem = model(x)
        raster = monitor.get("lifs.0")  # (T, batch, n_neurons)
        monitor.reset()
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._hooks: list[Any] = []
        self._records: dict[str, list[torch.Tensor]] = {}
        self._attach()

    def _attach(self) -> None:
        for name, module in self.model.named_modules():
            if hasattr(module, "surrogate_fn"):  # LIF-like cell
                self._records[name] = []
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)

    def _make_hook(self, name: str) -> Callable[..., None]:
        def hook(module, input, output):  # type: ignore[no-untyped-def]
            # output is (spike, v_next) or (spike, v_next, a_next) etc.
            if isinstance(output, tuple) and len(output) >= 1:
                self._records[name].append(output[0].detach())

        return hook

    def get(self, name: str) -> torch.Tensor | None:
        """Get recorded spikes for a named module. Returns (T, *shape) or None."""
        if name in self._records and self._records[name]:
            return torch.stack(self._records[name])
        return None

    @property
    def layer_names(self) -> list[str]:
        """Return names of modules that currently have spike hooks attached."""
        return list(self._records.keys())

    def reset(self) -> None:
        """Clear recorded spike tensors while keeping hooks attached."""
        for v in self._records.values():
            v.clear()

    def remove(self) -> None:
        """Remove forward hooks and clear all recorded spike tensors."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._records.clear()


def model_info(model: nn.Module) -> dict[str, Any]:
    """Quick architecture summary for SNN models."""
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    cell_types = set()
    n_lif_cells = 0
    for m in model.modules():
        if hasattr(m, "surrogate_fn"):
            cell_types.add(type(m).__name__)
            n_lif_cells += 1

    learnable_dynamics = []
    for name, _p in model.named_parameters():
        if "beta_logit" in name or "threshold_log" in name:
            learnable_dynamics.append(name)

    return {
        "total_params": n_params,
        "trainable_params": n_trainable,
        "spiking_cells": n_lif_cells,
        "cell_types": sorted(cell_types),
        "learnable_dynamics": learnable_dynamics,
    }


def population_decode(
    spike_counts: torch.Tensor,
    preferred_values: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode spike counts with a population-vector weighted average.

    Instead of argmax, computes a weighted average of preferred values
    based on spike counts. More informative than winner-take-all.

    Parameters
    ----------
    spike_counts : torch.Tensor
        Shape (batch, n_neurons). Spike counts per neuron.
    preferred_values : torch.Tensor or None
        Shape (n_neurons,) or (n_neurons, d). If None, uses neuron indices.

    Returns
    -------
    torch.Tensor
        Decoded values, shape (batch,) or (batch, d).
    """
    if preferred_values is None:
        preferred_values = torch.arange(
            spike_counts.shape[1], dtype=spike_counts.dtype, device=spike_counts.device
        )

    total = spike_counts.sum(dim=1, keepdim=True).clamp(min=1e-8)
    weights = spike_counts / total

    if preferred_values.dim() == 1:
        return (weights * preferred_values.unsqueeze(0)).sum(dim=1)
    return torch.einsum("bn,nd->bd", weights, preferred_values)
