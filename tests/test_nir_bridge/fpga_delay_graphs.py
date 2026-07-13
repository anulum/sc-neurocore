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


def _graph() -> Any:
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([2])}),
            "aff": nir.Affine(
                weight=np.array([[0.25, -0.5], [0.75, 0.125]], dtype=np.float32),
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
        edges=[("input", "aff"), ("aff", "lif"), ("lif", "output")],
    )


def _recurrent_graph() -> Any:
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([2])}),
            "aff": nir.Affine(
                weight=np.eye(2, dtype=np.float32),
                bias=np.zeros(2, dtype=np.float32),
            ),
            "lif": nir.LIF(
                tau=np.full(2, 20.0),
                r=np.ones(2),
                v_leak=np.zeros(2),
                v_threshold=np.ones(2),
            ),
            "rec": nir.Linear(weight=np.array([[0.25, 0.0], [0.0, 0.125]], dtype=np.float32)),
            "output": nir.Output(output_type={"output": np.array([2])}),
        },
        edges=[
            ("input", "aff"),
            ("aff", "lif"),
            ("lif", "rec"),
            ("rec", "lif"),
            ("lif", "output"),
        ],
    )


def _explicit_delay_graph() -> Any:
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([1])}),
            "aff": nir.Affine(
                weight=np.ones((1, 1), dtype=np.float32),
                bias=np.zeros(1, dtype=np.float32),
            ),
            "lif0": nir.LIF(
                tau=np.full(1, 20.0),
                r=np.ones(1),
                v_leak=np.zeros(1),
                v_threshold=np.ones(1),
            ),
            "delay": nir.Delay(delay=np.array([2.0])),
            "readout": nir.Linear(weight=np.array([[0.25]], dtype=np.float32)),
            "lif1": nir.LIF(
                tau=np.full(1, 20.0),
                r=np.ones(1),
                v_leak=np.zeros(1),
                v_threshold=np.ones(1),
            ),
            "output": nir.Output(output_type={"output": np.array([1])}),
        },
        edges=[
            ("input", "aff"),
            ("aff", "lif0"),
            ("lif0", "delay"),
            ("delay", "readout"),
            ("readout", "lif1"),
            ("lif1", "output"),
        ],
    )


def _explicit_analogue_delay_graph() -> Any:
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([1])}),
            "aff": nir.Affine(
                weight=np.ones((1, 1), dtype=np.float32),
                bias=np.zeros(1, dtype=np.float32),
            ),
            "li": nir.LI(
                tau=np.full(1, 15.0),
                r=np.ones(1),
                v_leak=np.zeros(1),
            ),
            "delay": nir.Delay(delay=np.array([2.0])),
            "readout": nir.Linear(weight=np.array([[0.5]], dtype=np.float32)),
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
            ("li", "delay"),
            ("delay", "readout"),
            ("readout", "lif"),
            ("lif", "output"),
        ],
    )


def _heterogeneous_delay_graph() -> Any:
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([2])}),
            "aff": nir.Affine(
                weight=np.eye(2, dtype=np.float32),
                bias=np.zeros(2, dtype=np.float32),
            ),
            "lif0": nir.LIF(
                tau=np.full(2, 20.0),
                r=np.ones(2),
                v_leak=np.zeros(2),
                v_threshold=np.ones(2),
            ),
            "delay": nir.Delay(delay=np.array([1.0, 2.0])),
            "readout": nir.Linear(weight=np.array([[0.25, -0.125]], dtype=np.float32)),
            "lif1": nir.LIF(
                tau=np.full(1, 20.0),
                r=np.ones(1),
                v_leak=np.zeros(1),
                v_threshold=np.ones(1),
            ),
            "output": nir.Output(output_type={"output": np.array([1])}),
        },
        edges=[
            ("input", "aff"),
            ("aff", "lif0"),
            ("lif0", "delay"),
            ("delay", "readout"),
            ("readout", "lif1"),
            ("lif1", "output"),
        ],
    )
