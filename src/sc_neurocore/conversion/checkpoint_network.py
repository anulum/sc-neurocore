# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Converted network from a trusted dense checkpoint

"""Rebuild the network a checkpoint holds and convert exactly that network.

Two checkpoint forms are read. A plain ``state_dict`` of a dense ReLU chain is
rebuilt in the order its layers were registered, with every trained bias, and
calibrated on caller-supplied samples or, without any, on unit ReLU scales. A
Studio ``qcfs_conversion`` checkpoint is rebuilt as the QCFS classifier its
recorded configuration describes and converted with its learned thresholds and
its own step budget. A Studio spiking checkpoint is already a spiking network
and is refused rather than reinterpreted as an ANN.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from .converted_snn import ConvertedSNN

#: Schema of the checkpoint the Studio seals as ``training/model_state.pt``.
STUDIO_CHECKPOINT_SCHEMA_VERSION = "studio.training.torch-state-dict.v1"


@dataclass(frozen=True)
class CheckpointNetwork:
    """A converted network and how it was obtained from its checkpoint.

    Attributes
    ----------
    snn : ConvertedSNN
        The converted network.
    source : {'state_dict', 'studio_qcfs_conversion'}
        Which checkpoint form was read.
    layer_sizes : list of tuple of int
        ``(inputs, outputs)`` of every dense layer, in forward order.
    calibration : str
        What set the ReLU thresholds: ``samples``, ``unit`` scales or
        ``learned QCFS thresholds``.
    """

    snn: ConvertedSNN
    source: Literal["state_dict", "studio_qcfs_conversion"]
    layer_sizes: list[tuple[int, int]]
    calibration: str


def build_qcfs_classifier(
    n_inputs: int, hidden: tuple[int, ...], n_outputs: int, steps: int
) -> Any:
    """Build the dense classifier with QCFS activations the Studio conversion route trains.

    Parameters
    ----------
    n_inputs, n_outputs : int
        Flattened input width and class count.
    hidden : tuple of int
        Hidden widths in order; empty for a direct input-to-output layer.
    steps : int
        QCFS step budget of every activation.

    Returns
    -------
    torch.nn.Sequential
        ``Flatten`` then alternating ``Linear`` and QCFS layers, ending in ``Linear``.
    """
    import torch.nn as nn

    from .ann_to_snn import replace_relu_with_qcfs

    layers: list[Any] = [nn.Flatten()]
    widths = (n_inputs, *hidden)
    for width_in, width_out in zip(widths, widths[1:]):
        layers += [nn.Linear(width_in, width_out), nn.ReLU()]
    layers.append(nn.Linear(widths[-1], n_outputs))
    return replace_relu_with_qcfs(nn.Sequential(*layers), T=steps, theta=1.0)


def _dense_chain(state: Mapping[str, Any], max_dense_params: int) -> list[str]:
    """Return the dense weight keys in registration order after checking each."""
    import torch

    keys = [
        key
        for key in state
        if (key == "weight" or key.endswith(".weight")) and state[key].dim() == 2
    ]
    if not keys:
        raise ValueError(
            "checkpoint does not contain any 2D dense '.weight' tensors required for deploy."
        )
    total = sum(int(state[key].numel()) for key in keys)
    if total > max_dense_params:
        raise ValueError(
            "deploy checkpoint dense parameter count exceeds safety limit "
            f"({max_dense_params:,}): {total:,}"
        )
    for key in keys:
        weight = state[key]
        if not torch.is_floating_point(weight):
            raise ValueError(f"deploy weight tensor '{key}' must use floating-point dtype.")
        if weight.shape[0] <= 0 or weight.shape[1] <= 0:
            raise ValueError(f"deploy weight tensor '{key}' must have non-zero 2D shape.")
        if not torch.isfinite(weight).all().item():
            raise ValueError(f"deploy weight tensor '{key}' contains non-finite values.")
    for previous, current in zip(keys, keys[1:]):
        if int(state[current].shape[1]) != int(state[previous].shape[0]):
            raise ValueError(
                "dense deploy weights are not composition-compatible between "
                f"'{previous}' (out={int(state[previous].shape[0])}) and "
                f"'{current}' (in={int(state[current].shape[1])})."
            )
    return keys


def _bias_key(key: str) -> str:
    """Return the bias key registered beside a dense weight key."""
    return key[: -len("weight")] + "bias"


def _bias(state: Mapping[str, Any], key: str) -> Any:
    """Return the trained bias belonging to a dense weight, or ``None``."""
    import torch

    bias = state.get(_bias_key(key))
    if bias is None:
        return None
    if (
        bias.dim() != 1
        or int(bias.shape[0]) != int(state[key].shape[0])
        or not torch.is_floating_point(bias)
        or not torch.isfinite(bias).all().item()
    ):
        raise ValueError(f"deploy bias for '{key}' must be a finite float vector of its outputs.")
    return bias


def _from_state_dict(
    state: Mapping[str, Any],
    steps: int,
    calibration: npt.NDArray[np.float64] | None,
    max_dense_params: int,
) -> CheckpointNetwork:
    """Rebuild and convert a plain dense ReLU chain."""
    import torch

    from .ann_to_snn import convert

    keys = _dense_chain(state, max_dense_params)
    rebuilt = set(keys) | {_bias_key(key) for key in keys}
    unread = sorted(set(state) - rebuilt)
    if unread:
        raise ValueError(
            "checkpoint holds parameters a dense ReLU chain does not have, so converting "
            f"it would change the network: {', '.join(unread[:5])}."
        )
    layers: list[Any] = []
    for key in keys:
        weight = state[key]
        bias = _bias(state, key)
        linear = torch.nn.Linear(int(weight.shape[1]), int(weight.shape[0]), bias=bias is not None)
        with torch.no_grad():
            linear.weight.copy_(weight.to(linear.weight.dtype))
            if bias is not None:
                linear.bias.copy_(bias.to(linear.bias.dtype))
        layers += [linear, torch.nn.ReLU()]
    model = torch.nn.Sequential(*layers[:-1])
    samples = None
    if calibration is not None:
        width = int(state[keys[0]].shape[1])
        if calibration.ndim != 2 or calibration.shape[1] != width or len(calibration) == 0:
            raise ValueError(f"calibration samples must be a non-empty (samples, {width}) array.")
        samples = torch.as_tensor(calibration, dtype=torch.float32)
    return CheckpointNetwork(
        snn=convert(model, calibration_data=samples, T=steps),
        source="state_dict",
        layer_sizes=[(int(state[k].shape[1]), int(state[k].shape[0])) for k in keys],
        calibration="unit" if samples is None else "samples",
    )


def _from_studio(payload: Mapping[str, Any], max_dense_params: int) -> CheckpointNetwork:
    """Rebuild and convert a Studio ``qcfs_conversion`` checkpoint."""
    from sc_neurocore.studio.training_contract import resolve_training_config

    from .ann_to_snn import convert

    config = payload.get("config")
    state = payload.get("model_state_dict")
    if not isinstance(config, Mapping) or not isinstance(state, Mapping):
        raise ValueError("Studio checkpoint must carry its config and model_state_dict.")
    resolved = resolve_training_config(config)
    if resolved.model_kind != "qcfs_conversion":
        raise ValueError(
            "a Studio spiking checkpoint is already a spiking network; deploy converts "
            "dense ANN checkpoints and qcfs_conversion runs."
        )
    keys = _dense_chain(state, max_dense_params)
    model = build_qcfs_classifier(
        int(state[keys[0]].shape[1]),
        resolved.hidden_widths,
        int(state[keys[-1]].shape[0]),
        resolved.timesteps,
    )
    try:
        model.load_state_dict(dict(state), strict=True)
    except RuntimeError as exc:
        raise ValueError(
            "Studio checkpoint weights do not match its recorded architecture."
        ) from exc
    return CheckpointNetwork(
        snn=convert(model, T=resolved.timesteps),
        source="studio_qcfs_conversion",
        layer_sizes=[(int(state[k].shape[1]), int(state[k].shape[0])) for k in keys],
        calibration="learned QCFS thresholds",
    )


def network_from_checkpoint(
    payload: object,
    *,
    steps: int,
    calibration: npt.NDArray[np.float64] | None = None,
    max_dense_params: int = 20_000_000,
) -> CheckpointNetwork:
    """Convert the network a trusted, already loaded checkpoint holds.

    Parameters
    ----------
    payload : object
        What ``torch.load(..., weights_only=True)`` returned.
    steps : int
        Timestep budget for a plain state dict; a Studio checkpoint uses its own.
    calibration : ndarray, optional
        ``(samples, inputs)`` source-format samples for ReLU threshold calibration
        of a plain state dict.
    max_dense_params : int
        Largest accepted number of dense weights.

    Returns
    -------
    CheckpointNetwork
        The converted network and its provenance.

    Raises
    ------
    ValueError
        The payload is not a dense checkpoint this function can rebuild exactly.
    """
    import torch

    if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
        raise ValueError("checkpoint must contain a state_dict-like dictionary.")
    if payload.get("schema_version") == STUDIO_CHECKPOINT_SCHEMA_VERSION:
        return _from_studio(payload, max_dense_params)
    if not all(torch.is_tensor(value) for value in payload.values()):
        raise ValueError("checkpoint state_dict entries must be tensors.")
    return _from_state_dict(payload, steps, calibration, max_dense_params)


__all__ = [
    "STUDIO_CHECKPOINT_SCHEMA_VERSION",
    "CheckpointNetwork",
    "build_qcfs_classifier",
    "network_from_checkpoint",
]
