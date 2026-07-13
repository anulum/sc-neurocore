# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Dense hardware lowering for NIR weight operators

"""Lower shape-known NIR convolution and pooling operators to dense matrices."""

from __future__ import annotations

from typing import Any

import numpy as np


def _conv1d_to_dense_matrix(
    node: Any,
    node_name: str,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Lower a shape-known NIR Conv1d node to an exact dense matrix."""
    weight = np.asarray(getattr(node, "weight", None), dtype=np.float32)
    if weight.ndim != 3:
        raise ValueError(f"Conv1d node {node_name!r} weight must have shape (C_out, C_in/group, K)")

    input_shape = getattr(node, "input_shape", None)
    if input_shape is None:
        raise ValueError(f"Conv1d node {node_name!r} requires input_shape for FPGA lowering")
    input_length = int(np.asarray(input_shape).reshape(-1)[0])
    if input_length <= 0:
        raise ValueError(f"Conv1d node {node_name!r} input_shape must be positive")

    stride = int(getattr(node, "stride", 1))
    padding = getattr(node, "padding", 0)
    if isinstance(padding, str):
        raise ValueError(f"Conv1d node {node_name!r} string padding requires explicit pre-lowering")
    padding = int(padding)
    dilation = int(getattr(node, "dilation", 1))
    groups = int(getattr(node, "groups", 1))
    if stride <= 0 or padding < 0 or dilation <= 0 or groups <= 0:
        raise ValueError(f"Conv1d node {node_name!r} has invalid stride/padding/dilation/groups")

    out_channels, in_channels_per_group, kernel_size = weight.shape
    if out_channels % groups != 0:
        raise ValueError(f"Conv1d node {node_name!r} output channels must be divisible by groups")
    in_channels = in_channels_per_group * groups
    output_length = (input_length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    if output_length <= 0:
        raise ValueError(f"Conv1d node {node_name!r} output length is not positive")

    dense = np.zeros(
        (out_channels * output_length, in_channels * input_length),
        dtype=np.float32,
    )
    out_channels_per_group = out_channels // groups
    for out_channel in range(out_channels):
        group = out_channel // out_channels_per_group
        in_channel_offset = group * in_channels_per_group
        for out_position in range(output_length):
            row = out_channel * output_length + out_position
            for local_channel in range(in_channels_per_group):
                in_channel = in_channel_offset + local_channel
                for kernel_position in range(kernel_size):
                    in_position = out_position * stride + kernel_position * dilation - padding
                    if 0 <= in_position < input_length:
                        column = in_channel * input_length + in_position
                        dense[row, column] = weight[
                            out_channel,
                            local_channel,
                            kernel_position,
                        ]

    raw_bias = getattr(node, "bias", None)
    if raw_bias is None:
        bias = np.zeros(out_channels, dtype=np.float32)
    else:
        bias = np.asarray(raw_bias, dtype=np.float32).reshape(-1)
    if bias.size != out_channels:
        raise ValueError(f"Conv1d node {node_name!r} bias length must equal output channels")
    return dense, np.repeat(bias, output_length).astype(np.float32, copy=False)


def _conv2d_to_dense_matrix(
    node: Any,
    node_name: str,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Lower a shape-known NIR Conv2d node to an exact dense matrix."""
    weight = np.asarray(getattr(node, "weight", None), dtype=np.float32)
    if weight.ndim != 4:
        raise ValueError(
            f"Conv2d node {node_name!r} weight must have shape (C_out, C_in/group, K_h, K_w)"
        )

    input_shape = getattr(node, "input_shape", None)
    if input_shape is None:
        raise ValueError(f"Conv2d node {node_name!r} requires input_shape for FPGA lowering")
    if len(input_shape) != 2:
        raise ValueError(f"Conv2d node {node_name!r} input_shape must be (height, width)")
    input_height, input_width = (int(value) for value in input_shape)
    if input_height <= 0 or input_width <= 0:
        raise ValueError(f"Conv2d node {node_name!r} input_shape must be positive")

    stride_height, stride_width = (int(value) for value in node.stride)
    pad_height, pad_width = node.padding
    if isinstance(pad_height, str) or isinstance(pad_width, str):
        raise ValueError(f"Conv2d node {node_name!r} string padding requires explicit pre-lowering")
    pad_height = int(pad_height)
    pad_width = int(pad_width)
    dilation_height, dilation_width = (int(value) for value in node.dilation)
    groups = int(node.groups)
    if (
        stride_height <= 0
        or stride_width <= 0
        or pad_height < 0
        or pad_width < 0
        or dilation_height <= 0
        or dilation_width <= 0
        or groups <= 0
    ):
        raise ValueError(f"Conv2d node {node_name!r} has invalid stride/padding/dilation/groups")

    out_channels, in_channels_per_group, kernel_height, kernel_width = weight.shape
    if out_channels % groups != 0:
        raise ValueError(f"Conv2d node {node_name!r} output channels must be divisible by groups")
    in_channels = in_channels_per_group * groups
    output_height = (
        input_height + 2 * pad_height - dilation_height * (kernel_height - 1) - 1
    ) // stride_height + 1
    output_width = (
        input_width + 2 * pad_width - dilation_width * (kernel_width - 1) - 1
    ) // stride_width + 1
    if output_height <= 0 or output_width <= 0:
        raise ValueError(f"Conv2d node {node_name!r} output shape is not positive")

    output_shape = getattr(node, "output_shape", None)
    if output_shape is not None:
        expected_output_shape = (out_channels, output_height, output_width)
        if tuple(int(value) for value in output_shape) != expected_output_shape:
            raise ValueError(
                f"Conv2d node {node_name!r} output_shape {output_shape} does not match "
                f"computed shape {expected_output_shape}"
            )

    dense = np.zeros(
        (out_channels * output_height * output_width, in_channels * input_height * input_width),
        dtype=np.float32,
    )
    out_channels_per_group = out_channels // groups
    for out_channel in range(out_channels):
        group = out_channel // out_channels_per_group
        in_channel_offset = group * in_channels_per_group
        for out_y in range(output_height):
            for out_x in range(output_width):
                row = (out_channel * output_height + out_y) * output_width + out_x
                for local_channel in range(in_channels_per_group):
                    in_channel = in_channel_offset + local_channel
                    for kernel_y in range(kernel_height):
                        in_y = out_y * stride_height + kernel_y * dilation_height - pad_height
                        if not 0 <= in_y < input_height:
                            continue
                        for kernel_x in range(kernel_width):
                            in_x = out_x * stride_width + kernel_x * dilation_width - pad_width
                            if not 0 <= in_x < input_width:
                                continue
                            column = (in_channel * input_height + in_y) * input_width + in_x
                            dense[row, column] += weight[
                                out_channel,
                                local_channel,
                                kernel_y,
                                kernel_x,
                            ]

    raw_bias = getattr(node, "bias", None)
    if raw_bias is None:
        bias = np.zeros(out_channels, dtype=np.float32)
    else:
        bias = np.asarray(raw_bias, dtype=np.float32).reshape(-1)
    if bias.size != out_channels:
        raise ValueError(f"Conv2d node {node_name!r} bias length must equal output channels")
    return dense, np.repeat(bias, output_height * output_width).astype(np.float32, copy=False)


def _pool2d_to_dense_matrix(node: Any, node_name: str) -> tuple[np.ndarray[Any, Any], None]:
    """Lower a shape-known NIR Pool2d node to an exact dense matrix."""
    class_name = type(node).__name__
    input_shape = getattr(node, "input_shape", None)
    output_shape = getattr(node, "output_shape", None)
    if input_shape is None or output_shape is None:
        primitive = class_name.removeprefix("SC").removesuffix("Node")
        raise ValueError(
            f"{primitive} node {node_name!r} requires input/output shape metadata for FPGA lowering"
        )
    if len(input_shape) != 3 or len(output_shape) != 3:
        raise ValueError(f"{class_name} {node_name!r} requires CHW input/output shape metadata")

    channels, input_height, input_width = (int(value) for value in input_shape)
    out_channels, output_height, output_width = (int(value) for value in output_shape)
    if channels <= 0 or input_height <= 0 or input_width <= 0:
        raise ValueError(f"{class_name} {node_name!r} has invalid input shape {input_shape}")
    if out_channels != channels or output_height <= 0 or output_width <= 0:
        raise ValueError(f"{class_name} {node_name!r} has invalid output shape {output_shape}")

    kernel_height, kernel_width = (int(value) for value in node.kernel_size)
    stride_height, stride_width = (int(value) for value in node.stride)
    pad_height, pad_width = (int(value) for value in node.padding)
    if (
        kernel_height <= 0
        or kernel_width <= 0
        or stride_height <= 0
        or stride_width <= 0
        or pad_height < 0
        or pad_width < 0
    ):
        raise ValueError(f"{class_name} {node_name!r} has invalid kernel/stride/padding")

    dense = np.zeros(
        (channels * output_height * output_width, channels * input_height * input_width),
        dtype=np.float32,
    )
    coefficient = 1.0
    if class_name == "SCAvgPool2dNode":
        coefficient = 1.0 / float(kernel_height * kernel_width)

    for channel in range(channels):
        for out_y in range(output_height):
            for out_x in range(output_width):
                row = (channel * output_height + out_y) * output_width + out_x
                for kernel_y in range(kernel_height):
                    in_y = out_y * stride_height + kernel_y - pad_height
                    if not 0 <= in_y < input_height:
                        continue
                    for kernel_x in range(kernel_width):
                        in_x = out_x * stride_width + kernel_x - pad_width
                        if not 0 <= in_x < input_width:
                            continue
                        column = (channel * input_height + in_y) * input_width + in_x
                        dense[row, column] += coefficient
    return dense, None


def _weight_matrix_and_bias(
    node: Any,
    node_name: str,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any] | None]:
    """Return dense weight and bias arrays for a weight-carrying NIR node."""
    class_name = type(node).__name__
    if class_name == "SCConv1dNode":
        return _conv1d_to_dense_matrix(node, node_name)
    if class_name == "SCConv2dNode":
        return _conv2d_to_dense_matrix(node, node_name)
    if class_name in {"SCSumPool2dNode", "SCAvgPool2dNode"}:
        return _pool2d_to_dense_matrix(node, node_name)

    weight = getattr(node, "weight", None)
    bias = getattr(node, "bias", None)
    if weight is None:
        weight = getattr(node, "weights", None)
    if weight is None:
        raise ValueError(f"Weight node {node_name!r} does not expose weights")
    dense_weight = np.asarray(weight, dtype=np.float32)
    dense_bias = None if bias is None else np.asarray(bias, dtype=np.float32)
    return dense_weight, dense_bias
