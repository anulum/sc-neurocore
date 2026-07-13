# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Delay, scale, threshold, and shape metadata lowering

"""Validate and compose connection metadata carried by pass-through NIR nodes."""

from __future__ import annotations

from typing import Any

import numpy as np

from sc_neurocore.nir_bridge.neuron_graph_contracts import DelaySteps


def _delay_steps(node: Any, node_name: str) -> DelaySteps:
    """Return scalar or per-source delay metadata from an explicit delay node."""
    raw_steps = getattr(node, "delay_steps", None)
    if raw_steps is None:
        raise ValueError(f"Delay node {node_name!r} does not expose delay_steps")
    steps = np.atleast_1d(np.asarray(raw_steps, dtype=np.int64)).reshape(-1)
    if steps.size == 0:
        raise ValueError(f"Delay node {node_name!r} must contain at least one delay value")
    if np.any(steps < 0):
        raise ValueError(f"Delay node {node_name!r} contains negative delay_steps")
    values = tuple(int(value) for value in steps)
    if len(values) == 1 or all(value == values[0] for value in values):
        return values[0]
    return values


def _delay_steps_array(delay_steps: DelaySteps) -> np.ndarray[Any, Any]:
    """Return delay metadata as a one-dimensional integer array."""
    return np.atleast_1d(np.asarray(delay_steps, dtype=np.int64)).reshape(-1)


def _compose_delay_steps(left: DelaySteps, right: DelaySteps) -> DelaySteps:
    """Compose adjacent delays with scalar or equal-width vector broadcasting."""
    left_steps = _delay_steps_array(left)
    right_steps = _delay_steps_array(right)
    if left_steps.size == 1 and right_steps.size == 1:
        return int(left_steps[0] + right_steps[0])
    if left_steps.size == 1:
        values = right_steps + int(left_steps[0])
    elif right_steps.size == 1:
        values = left_steps + int(right_steps[0])
    elif left_steps.size == right_steps.size:
        values = left_steps + right_steps
    else:
        raise ValueError(
            f"Incompatible delay vector lengths {left_steps.size} and {right_steps.size}; "
            "split delayed channels before FPGA lowering"
        )
    result = tuple(int(value) for value in values)
    if all(value == result[0] for value in result):
        return result[0]
    return result


def _fit_delay_steps_to_width(delay_steps: DelaySteps, width: int, label: str) -> DelaySteps:
    """Validate scalar or vector delay metadata against a source width."""
    steps = _delay_steps_array(delay_steps)
    if steps.size == 1:
        return int(steps[0])
    if steps.size != width:
        raise ValueError(
            f"{label} delay_steps length {steps.size} does not match source width {width}"
        )
    return tuple(int(value) for value in steps)


def _scale_vector(node: Any, node_name: str) -> np.ndarray[Any, Any]:
    """Return a finite one-dimensional scale vector."""
    raw_scale = getattr(node, "scale", None)
    if raw_scale is None:
        raise ValueError(f"Scale node {node_name!r} does not expose scale")
    scale = np.atleast_1d(np.asarray(raw_scale, dtype=np.float32)).reshape(-1)
    if scale.size == 0:
        raise ValueError(f"Scale node {node_name!r} must contain at least one scale value")
    if not np.all(np.isfinite(scale)):
        raise ValueError(f"Scale node {node_name!r} contains non-finite scale values")
    return scale


def _threshold_vector(node: Any, node_name: str) -> np.ndarray[Any, Any]:
    """Return a finite one-dimensional threshold vector."""
    raw_threshold = getattr(node, "threshold", None)
    if raw_threshold is None:
        raise ValueError(f"Threshold node {node_name!r} does not expose threshold")
    threshold = np.atleast_1d(np.asarray(raw_threshold, dtype=np.float32)).reshape(-1)
    if threshold.size == 0:
        raise ValueError(f"Threshold node {node_name!r} must contain at least one threshold")
    if not np.all(np.isfinite(threshold)):
        raise ValueError(f"Threshold node {node_name!r} contains non-finite threshold values")
    return threshold


def _compose_scale(
    left: np.ndarray[Any, Any] | None,
    right: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """Compose adjacent scale vectors under NumPy broadcasting rules."""
    if left is None:
        return right
    try:
        product: np.ndarray[Any, Any] = np.multiply(left, right, dtype=np.float32)
        return product
    except ValueError as exc:
        raise ValueError("Adjacent Scale nodes have incompatible shapes for FPGA lowering") from exc


def _broadcast_scale(
    scale: np.ndarray[Any, Any] | None,
    size: int,
    label: str,
) -> np.ndarray[Any, Any] | None:
    """Broadcast a scalar or exact-width scale vector."""
    if scale is None:
        return None
    if scale.size == 1:
        return np.full(size, float(scale[0]), dtype=np.float32)
    if scale.size == size:
        return scale.astype(np.float32, copy=False)
    raise ValueError(f"{label} scale length {scale.size} does not match required width {size}")


def _broadcast_threshold(
    threshold: np.ndarray[Any, Any] | None,
    size: int,
    label: str,
) -> np.ndarray[Any, Any] | None:
    """Broadcast a scalar or exact-width threshold vector."""
    if threshold is None:
        return None
    if threshold.size == 1:
        return np.full(size, float(threshold[0]), dtype=np.float32)
    if threshold.size == size:
        return threshold.astype(np.float32, copy=False)
    raise ValueError(
        f"{label} threshold length {threshold.size} does not match required width {size}"
    )


def _shape_width(shape: tuple[int, ...] | None, *, node_name: str, label: str) -> int:
    """Return the element count for a finite NIR shape."""
    if shape is None:
        raise ValueError(f"Flatten node {node_name!r} lacks {label} shape metadata")
    if not shape:
        return 1
    width = int(np.prod(np.asarray(shape, dtype=np.int64)))
    if width <= 0:
        raise ValueError(f"Flatten node {node_name!r} has invalid {label} shape {shape}")
    return width


def _flatten_widths(node: Any, node_name: str) -> tuple[int, int]:
    """Return equal input and output widths for a shape-typed flatten node."""
    input_width = _shape_width(
        getattr(node, "input_shape", None),
        node_name=node_name,
        label="input",
    )
    output_width = _shape_width(
        getattr(node, "output_shape", None),
        node_name=node_name,
        label="output",
    )
    if input_width != output_width:
        raise ValueError(
            f"Flatten node {node_name!r} changes element count from {input_width} to {output_width}"
        )
    return input_width, output_width
