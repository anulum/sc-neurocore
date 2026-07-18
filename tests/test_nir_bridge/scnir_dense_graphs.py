# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for SC-NIR conversion wiring

"""SC-NIR conversion tests for the NIR/NeuronGraph pipeline."""

from __future__ import annotations

from typing import Any


import numpy as np
import pytest

nir = pytest.importorskip("nir")


def _build_conv1d_lif_graph() -> Any:
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([1, 4])}),
            "conv": nir.Conv1d(
                input_shape=4,
                weight=np.array(
                    [[[1.0, 2.0]], [[-1.0, 0.5]]],
                    dtype=np.float32,
                ),
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=np.array([0.1, -0.2], dtype=np.float32),
            ),
            "flatten": nir.Flatten(input_type={"input": np.array([2, 3])}, start_dim=0),
            "lif": nir.LIF(
                tau=np.full(6, 20.0),
                r=np.ones(6),
                v_leak=np.zeros(6),
                v_threshold=np.ones(6),
            ),
            "output": nir.Output(output_type={"output": np.array([6])}),
        },
        edges=[("input", "conv"), ("conv", "flatten"), ("flatten", "lif"), ("lif", "output")],
    )


def _build_conv1d_without_shape_lif_graph() -> Any:
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([1, 4])}),
            "conv": nir.Conv1d(
                input_shape=None,
                weight=np.array([[[1.0, 2.0]]], dtype=np.float32),
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=np.zeros(1, dtype=np.float32),
            ),
            "lif": nir.LIF(
                tau=np.full(3, 20.0),
                r=np.ones(3),
                v_leak=np.zeros(3),
                v_threshold=np.ones(3),
            ),
            "output": nir.Output(output_type={"output": np.array([3])}),
        },
        edges=[("input", "conv"), ("conv", "lif"), ("lif", "output")],
        type_check=False,
    )


def _build_conv2d_lif_graph() -> Any:
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([1, 3, 3])}),
            "conv": nir.Conv2d(
                input_shape=(3, 3),
                weight=np.array(
                    [[[[1.0, 2.0], [3.0, 4.0]]]],
                    dtype=np.float32,
                ),
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=np.array([0.5], dtype=np.float32),
            ),
            "flatten": nir.Flatten(input_type={"input": np.array([1, 2, 2])}, start_dim=0),
            "lif": nir.LIF(
                tau=np.full(4, 20.0),
                r=np.ones(4),
                v_leak=np.zeros(4),
                v_threshold=np.ones(4),
            ),
            "output": nir.Output(output_type={"output": np.array([4])}),
        },
        edges=[("input", "conv"), ("conv", "flatten"), ("flatten", "lif"), ("lif", "output")],
    )


def _build_conv2d_without_shape_lif_graph() -> Any:
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([1, 3, 3])}),
            "conv": nir.Conv2d(
                input_shape=None,
                weight=np.ones((1, 1, 2, 2), dtype=np.float32),
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=np.zeros(1, dtype=np.float32),
            ),
            "lif": nir.LIF(
                tau=np.full(4, 20.0),
                r=np.ones(4),
                v_leak=np.zeros(4),
                v_threshold=np.ones(4),
            ),
            "output": nir.Output(output_type={"output": np.array([4])}),
        },
        edges=[("input", "conv"), ("conv", "lif"), ("lif", "output")],
        type_check=False,
    )


def _build_sum_pool2d_lif_graph() -> Any:
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([1, 3, 3])}),
            "pool": nir.SumPool2d(
                kernel_size=np.array([2, 2]),
                stride=np.array([1, 1]),
                padding=np.array([0, 0]),
            ),
            "flatten": nir.Flatten(input_type={"input": np.array([1, 2, 2])}, start_dim=0),
            "lif": nir.LIF(
                tau=np.full(4, 20.0),
                r=np.ones(4),
                v_leak=np.zeros(4),
                v_threshold=np.ones(4),
            ),
            "output": nir.Output(output_type={"output": np.array([4])}),
        },
        edges=[("input", "pool"), ("pool", "flatten"), ("flatten", "lif"), ("lif", "output")],
    )


def _build_avg_pool2d_lif_graph() -> Any:
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([1, 3, 3])}),
            "pool": nir.AvgPool2d(
                kernel_size=np.array([2, 2]),
                stride=np.array([1, 1]),
                padding=np.array([0, 0]),
            ),
            "flatten": nir.Flatten(input_type={"input": np.array([1, 2, 2])}, start_dim=0),
            "lif": nir.LIF(
                tau=np.full(4, 20.0),
                r=np.ones(4),
                v_leak=np.zeros(4),
                v_threshold=np.ones(4),
            ),
            "output": nir.Output(output_type={"output": np.array([4])}),
        },
        edges=[("input", "pool"), ("pool", "flatten"), ("flatten", "lif"), ("lif", "output")],
    )


def _build_sum_pool2d_without_shape_lif_graph() -> Any:
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([1, 3, 3])}),
            "pool": nir.SumPool2d(
                kernel_size=np.array([2, 2]),
                stride=np.array([1, 1]),
                padding=np.array([0, 0]),
            ),
            "lif": nir.LIF(
                tau=np.full(4, 20.0),
                r=np.ones(4),
                v_leak=np.zeros(4),
                v_threshold=np.ones(4),
            ),
            "output": nir.Output(output_type={"output": np.array([4])}),
        },
        edges=[("input", "pool"), ("pool", "lif"), ("lif", "output")],
        type_check=False,
    )
