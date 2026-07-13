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


def _build_source_scale_li_lif_graph() -> Any:
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([2])}),
            "aff": nir.Affine(
                weight=np.eye(2, dtype=np.float32),
                bias=np.zeros(2, dtype=np.float32),
            ),
            "li": nir.LI(
                tau=np.full(2, 15.0),
                r=np.ones(2),
                v_leak=np.zeros(2),
            ),
            "scale": nir.Scale(scale=np.array([2.0, 0.5], dtype=np.float32)),
            "readout": nir.Linear(weight=np.array([[0.25, -0.25]], dtype=np.float32)),
            "lif": nir.LIF(
                tau=np.full(1, 20.0),
                r=np.ones(1),
                v_leak=np.zeros(1),
                v_threshold=np.ones(1),
            ),
            "output": nir.Output(output_type={"output": np.array([1])}),
        },
        edges=[
            ("input", "aff"),
            ("aff", "li"),
            ("li", "scale"),
            ("scale", "readout"),
            ("readout", "lif"),
            ("lif", "output"),
        ],
    )


def _build_post_weight_scale_lif_graph() -> Any:
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([2])}),
            "aff": nir.Affine(
                weight=np.array([[0.25, -0.5], [0.125, 0.25]], dtype=np.float32),
                bias=np.array([0.1, -0.2], dtype=np.float32),
            ),
            "scale": nir.Scale(scale=np.array([2.0, 0.5], dtype=np.float32)),
            "lif": nir.LIF(
                tau=np.full(2, 20.0),
                r=np.ones(2),
                v_leak=np.zeros(2),
                v_threshold=np.ones(2),
            ),
            "output": nir.Output(output_type={"output": np.array([2])}),
        },
        edges=[("input", "aff"), ("aff", "scale"), ("scale", "lif"), ("lif", "output")],
    )


def _build_flattened_input_lif_graph() -> Any:
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([2, 2])}),
            "flatten": nir.Flatten(input_type={"input": np.array([2, 2])}, start_dim=0),
            "aff": nir.Affine(
                weight=np.array(
                    [[0.25, -0.5, 0.125, 0.75], [-0.25, 0.5, -0.125, 0.25]],
                    dtype=np.float32,
                ),
                bias=np.array([0.1, -0.2], dtype=np.float32),
            ),
            "lif": nir.LIF(
                tau=np.full(2, 20.0),
                r=np.ones(2),
                v_leak=np.zeros(2),
                v_threshold=np.ones(2),
            ),
            "output": nir.Output(output_type={"output": np.array([2])}),
        },
        edges=[("input", "flatten"), ("flatten", "aff"), ("aff", "lif"), ("lif", "output")],
    )


def _build_incompatible_flattened_input_lif_graph() -> Any:
    graph = _build_flattened_input_lif_graph()
    graph.nodes["aff"] = nir.Affine(
        weight=np.array([[0.25, -0.5, 0.125]], dtype=np.float32),
        bias=np.array([0.1], dtype=np.float32),
    )
    graph.nodes["lif"] = nir.LIF(
        tau=np.full(1, 20.0),
        r=np.ones(1),
        v_leak=np.zeros(1),
        v_threshold=np.ones(1),
    )
    graph.nodes["output"] = nir.Output(output_type={"output": np.array([1])})
    return graph


def _build_post_weight_flatten_lif_graph() -> Any:
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([2])}),
            "aff": nir.Affine(
                weight=np.array(
                    [[0.25, -0.5], [0.125, 0.75], [-0.25, 0.5], [0.375, -0.125]],
                    dtype=np.float32,
                ),
                bias=np.array([0.1, -0.2, 0.0, 0.05], dtype=np.float32),
            ),
            "flatten": nir.Flatten(input_type={"input": np.array([4])}, start_dim=0),
            "lif": nir.LIF(
                tau=np.full(4, 20.0),
                r=np.ones(4),
                v_leak=np.zeros(4),
                v_threshold=np.ones(4),
            ),
            "output": nir.Output(output_type={"output": np.array([4])}),
        },
        edges=[("input", "aff"), ("aff", "flatten"), ("flatten", "lif"), ("lif", "output")],
    )


def _build_incompatible_post_weight_flatten_lif_graph() -> Any:
    graph = _build_post_weight_flatten_lif_graph()
    graph.nodes["lif"] = nir.LIF(
        tau=np.full(3, 20.0),
        r=np.ones(3),
        v_leak=np.zeros(3),
        v_threshold=np.ones(3),
    )
    graph.nodes["output"] = nir.Output(output_type={"output": np.array([3])})
    return graph


def _build_source_threshold_li_lif_graph() -> Any:
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([2])}),
            "aff": nir.Affine(
                weight=np.eye(2, dtype=np.float32),
                bias=np.zeros(2, dtype=np.float32),
            ),
            "li": nir.LI(
                tau=np.full(2, 15.0),
                r=np.ones(2),
                v_leak=np.zeros(2),
            ),
            "threshold": nir.Threshold(threshold=np.array([0.25, 0.5], dtype=np.float32)),
            "readout": nir.Linear(weight=np.array([[0.5, -0.25]], dtype=np.float32)),
            "lif": nir.LIF(
                tau=np.full(1, 20.0),
                r=np.ones(1),
                v_leak=np.zeros(1),
                v_threshold=np.ones(1),
            ),
            "output": nir.Output(output_type={"output": np.array([1])}),
        },
        edges=[
            ("input", "aff"),
            ("aff", "li"),
            ("li", "threshold"),
            ("threshold", "readout"),
            ("readout", "lif"),
            ("lif", "output"),
        ],
    )


def _build_post_weight_threshold_lif_graph() -> Any:
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([2])}),
            "aff": nir.Affine(
                weight=np.array([[0.25, -0.5], [0.125, 0.25]], dtype=np.float32),
                bias=np.array([0.1, -0.2], dtype=np.float32),
            ),
            "threshold": nir.Threshold(threshold=np.array([0.2, -0.1], dtype=np.float32)),
            "lif": nir.LIF(
                tau=np.full(2, 20.0),
                r=np.ones(2),
                v_leak=np.zeros(2),
                v_threshold=np.ones(2),
            ),
            "output": nir.Output(output_type={"output": np.array([2])}),
        },
        edges=[
            ("input", "aff"),
            ("aff", "threshold"),
            ("threshold", "lif"),
            ("lif", "output"),
        ],
    )
