# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_nir_fpga_pipeline.py

from __future__ import annotations

"""End-to-end tests for the NIR → FPGA compilation pipeline.

Every test in this module is a full end-to-end pipeline execution:
NIR graph construction → from_nir() → from_scnetwork() →
compile_network_to_fpga() → Verilog artefact verification.
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from uuid import uuid4
import numpy as np
import pytest

nir = pytest.importorskip("nir")
from sc_neurocore.compiler.equation_compiler import Q88
from sc_neurocore.nir_bridge import (
    compile_network_to_fpga,
    from_nir,
    from_scnetwork,
    quantise_graph,
)
from sc_neurocore.nir_bridge.fpga_compiler import _AER_THRESHOLD


@pytest.fixture
def local_tmp_path():
    root = Path(__file__).resolve().parents[1] / ".pytest_tmp"
    root.mkdir(exist_ok=True)
    path = root / uuid4().hex
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


def _build_lif_feedforward(n_in=4, n_hidden=8, n_out=2, seed=42):
    """Build: Input → Affine → LIF → Affine → LIF → Output."""
    rng = np.random.RandomState(seed)
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([n_in])}),
            "aff1": nir.Affine(
                weight=rng.randn(n_hidden, n_in).astype(np.float32),
                bias=np.zeros(n_hidden, dtype=np.float32),
            ),
            "lif1": nir.LIF(
                tau=np.full(n_hidden, 20.0),
                r=np.ones(n_hidden),
                v_leak=np.zeros(n_hidden),
                v_threshold=np.ones(n_hidden),
            ),
            "aff2": nir.Affine(
                weight=rng.randn(n_out, n_hidden).astype(np.float32),
                bias=np.zeros(n_out, dtype=np.float32),
            ),
            "lif2": nir.LIF(
                tau=np.full(n_out, 20.0),
                r=np.ones(n_out),
                v_leak=np.zeros(n_out),
                v_threshold=np.ones(n_out),
            ),
            "output": nir.Output(output_type={"output": np.array([n_out])}),
        },
        edges=[
            ("input", "aff1"),
            ("aff1", "lif1"),
            ("lif1", "aff2"),
            ("aff2", "lif2"),
            ("lif2", "output"),
        ],
    )


def _build_cubalif_network(n_in=3, n_out=4, seed=99):
    """Build: Input → Affine → CubaLIF → Output."""
    rng = np.random.RandomState(seed)
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([n_in])}),
            "aff": nir.Affine(
                weight=rng.randn(n_out, n_in).astype(np.float32),
                bias=np.zeros(n_out, dtype=np.float32),
            ),
            "cuba": nir.CubaLIF(
                tau_syn=np.full(n_out, 5.0),
                tau_mem=np.full(n_out, 20.0),
                r=np.ones(n_out),
                v_leak=np.zeros(n_out),
                v_threshold=np.ones(n_out),
                w_in=np.ones(n_out),
            ),
            "output": nir.Output(output_type={"output": np.array([n_out])}),
        },
        edges=[("input", "aff"), ("aff", "cuba"), ("cuba", "output")],
    )


def _build_mixed_type_network(n_in=4, seed=77):
    """Build: Input → Affine → IF → Affine → LIF → Output."""
    rng = np.random.RandomState(seed)
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([n_in])}),
            "aff1": nir.Affine(
                weight=rng.randn(6, n_in).astype(np.float32),
                bias=np.zeros(6, dtype=np.float32),
            ),
            "if_layer": nir.IF(r=np.ones(6), v_threshold=np.ones(6)),
            "aff2": nir.Affine(
                weight=rng.randn(3, 6).astype(np.float32),
                bias=np.zeros(3, dtype=np.float32),
            ),
            "lif_layer": nir.LIF(
                tau=np.full(3, 15.0),
                r=np.ones(3),
                v_leak=np.zeros(3),
                v_threshold=np.ones(3),
            ),
            "output": nir.Output(output_type={"output": np.array([3])}),
        },
        edges=[
            ("input", "aff1"),
            ("aff1", "if_layer"),
            ("if_layer", "aff2"),
            ("aff2", "lif_layer"),
            ("lif_layer", "output"),
        ],
    )


def _full_pipeline(nir_graph, dt=1.0, data_width=16, fraction=8, module_name="sc_test"):
    """Run the full NIR → FPGA pipeline and return the result."""
    net = from_nir(nir_graph, dt=dt)
    ng = from_scnetwork(net, dt=dt)
    return compile_network_to_fpga(
        ng,
        module_name=module_name,
        data_width=data_width,
        fraction=fraction,
    )


__all__ = [
    "os",
    "re",
    "shutil",
    "subprocess",
    "sys",
    "Path",
    "uuid4",
    "np",
    "pytest",
    "nir",
    "Q88",
    "compile_network_to_fpga",
    "from_nir",
    "from_scnetwork",
    "quantise_graph",
    "_AER_THRESHOLD",
    "local_tmp_path",
    "_build_lif_feedforward",
    "_build_cubalif_network",
    "_build_mixed_type_network",
    "_full_pipeline",
]
