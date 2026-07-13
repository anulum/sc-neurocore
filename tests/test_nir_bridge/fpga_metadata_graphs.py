# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for SC-NIR FPGA integration

"""SC-NIR metadata integration tests for FPGA compilation artefacts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

nir = pytest.importorskip("nir")


def _integrator_graph() -> Any:
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([2])}),
            "aff": nir.Affine(
                weight=np.eye(2, dtype=np.float32),
                bias=np.zeros(2, dtype=np.float32),
            ),
            "i": nir.I(r=np.ones(2)),
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
            ("aff", "i"),
            ("i", "readout"),
            ("readout", "lif"),
            ("lif", "output"),
        ],
    )


def _source_scale_graph() -> Any:
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


def _post_weight_scale_graph() -> Any:
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


def _flattened_input_graph() -> Any:
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([2, 2])}),
            "flatten": nir.Flatten(input_type={"input": np.array([2, 2])}, start_dim=0),
            "aff": nir.Affine(
                weight=np.array(
                    [[0.25, -0.5, 0.125, 0.75], [-0.25, 0.5, -0.125, 0.25]],
                    dtype=np.float32,
                ),
                bias=np.zeros(2, dtype=np.float32),
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


def _source_threshold_graph() -> Any:
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


def _post_weight_threshold_graph() -> Any:
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
        edges=[("input", "aff"), ("aff", "threshold"), ("threshold", "lif"), ("lif", "output")],
    )
