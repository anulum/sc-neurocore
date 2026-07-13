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


def _build_nested_subgraph_lif_graph() -> Any:
    inner = nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([2])}),
            "aff": nir.Affine(
                weight=np.eye(2, dtype=np.float32),
                bias=np.zeros(2, dtype=np.float32),
            ),
            "output": nir.Output(output_type={"output": np.array([2])}),
        },
        edges=[("input", "aff"), ("aff", "output")],
    )
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([2])}),
            "subgraph": inner,
            "lif": nir.LIF(
                tau=np.full(2, 20.0),
                r=np.ones(2),
                v_leak=np.zeros(2),
                v_threshold=np.ones(2),
            ),
            "output": nir.Output(output_type={"output": np.array([2])}),
        },
        edges=[("input", "subgraph"), ("subgraph", "lif"), ("lif", "output")],
    )


def _build_multiport_nested_subgraph_graph() -> Any:
    inner = nir.NIRGraph(
        nodes={
            "a": nir.Input(input_type={"input": np.array([1])}),
            "b": nir.Input(input_type={"input": np.array([1])}),
            "aff": nir.Affine(
                weight=np.array([[1.0, -1.0]], dtype=np.float32),
                bias=np.zeros(1, dtype=np.float32),
            ),
            "output": nir.Output(output_type={"output": np.array([1])}),
        },
        edges=[("a", "aff"), ("b", "aff"), ("aff", "output")],
        type_check=False,
    )
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([1])}),
            "subgraph": inner,
            "lif": nir.LIF(
                tau=np.full(1, 20.0),
                r=np.ones(1),
                v_leak=np.zeros(1),
                v_threshold=np.ones(1),
            ),
            "output": nir.Output(output_type={"output": np.array([1])}),
        },
        edges=[("input", "subgraph"), ("subgraph", "lif"), ("lif", "output")],
        type_check=False,
    )


def _build_exact_multiport_nested_subgraph_graph() -> Any:
    inner = nir.NIRGraph(
        nodes={
            "a": nir.Input(input_type={"input": np.array([1])}),
            "b": nir.Input(input_type={"input": np.array([1])}),
            "aff": nir.Affine(
                weight=np.array([[1.0, -1.0]], dtype=np.float32),
                bias=np.zeros(1, dtype=np.float32),
            ),
            "output": nir.Output(output_type={"output": np.array([1])}),
        },
        edges=[("a", "aff"), ("b", "aff"), ("aff", "output")],
        type_check=False,
    )
    return nir.NIRGraph(
        nodes={
            "left": nir.Input(input_type={"input": np.array([1])}),
            "right": nir.Input(input_type={"input": np.array([1])}),
            "subgraph": inner,
            "lif": nir.LIF(
                tau=np.full(1, 20.0),
                r=np.ones(1),
                v_leak=np.zeros(1),
                v_threshold=np.ones(1),
            ),
            "output": nir.Output(output_type={"output": np.array([1])}),
        },
        edges=[
            ("left", "subgraph"),
            ("right", "subgraph"),
            ("subgraph", "lif"),
            ("lif", "output"),
        ],
        type_check=False,
    )


def _build_exact_multiport_multioutput_nested_subgraph_graph() -> Any:
    inner = nir.NIRGraph(
        nodes={
            "a": nir.Input(input_type={"input": np.array([1])}),
            "b": nir.Input(input_type={"input": np.array([1])}),
            "aff_a": nir.Affine(
                weight=np.array([[0.5]], dtype=np.float32),
                bias=np.zeros(1, dtype=np.float32),
            ),
            "aff_b": nir.Affine(
                weight=np.array([[-0.25]], dtype=np.float32),
                bias=np.zeros(1, dtype=np.float32),
            ),
            "out_a": nir.Output(output_type={"output": np.array([1])}),
            "out_b": nir.Output(output_type={"output": np.array([1])}),
        },
        edges=[
            ("a", "aff_a"),
            ("aff_a", "out_a"),
            ("b", "aff_b"),
            ("aff_b", "out_b"),
        ],
        type_check=False,
    )
    return nir.NIRGraph(
        nodes={
            "left": nir.Input(input_type={"input": np.array([1])}),
            "right": nir.Input(input_type={"input": np.array([1])}),
            "subgraph": inner,
            "lif_a": nir.LIF(
                tau=np.full(1, 20.0),
                r=np.ones(1),
                v_leak=np.zeros(1),
                v_threshold=np.ones(1),
            ),
            "lif_b": nir.LIF(
                tau=np.full(1, 20.0),
                r=np.ones(1),
                v_leak=np.zeros(1),
                v_threshold=np.ones(1),
            ),
            "output": nir.Output(output_type={"output": np.array([2])}),
        },
        edges=[
            ("left", "subgraph"),
            ("right", "subgraph"),
            ("subgraph", "lif_a"),
            ("subgraph", "lif_b"),
            ("lif_a", "output"),
            ("lif_b", "output"),
        ],
        type_check=False,
    )
