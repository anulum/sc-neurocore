# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_folded_heterogeneous_params.py

"""The folded interconnect streams heterogeneous per-neuron parameters from a parameter ROM.

A population whose neurons carry different quantised parameters cannot bake one shared
value into the per-type processing element, so the folded interconnect exposes those
parameters as PE input ports and drives them from a per-neuron ``case(nidx)`` ROM — the
parameter-space analogue of the state BRAM. These tests pin the parameter-uniformity
predicate, the ``_can_fold`` gate, the generated ROM and PE ports, the resource metric,
and — critically — that the PE bakes the *real* parameter value (a regression guard for
the double-encoding that baked ``tau = 0`` into the folded PE for every real graph).
The bit-exact co-simulation against the direct path lives in ``test_folded_interconnect``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from sc_neurocore.nir_bridge.fpga_compiler import NetworkCompilationResult
    from sc_neurocore.nir_bridge.neuron_graph import NeuronGraph
    from sc_neurocore.nir_bridge.quantise_params import QuantisedGraph

pytest.importorskip("nir")
import nir
from sc_neurocore.compiler.equation_compiler import Q88
from sc_neurocore.nir_bridge import (
    compile_network_to_fpga,
    from_nir,
    from_scnetwork,
    quantise_graph,
)
from sc_neurocore.nir_bridge.fpga_folded_interconnect import can_fold as _can_fold
from sc_neurocore.nir_bridge.fpga_compiler import (
    _dequantised_pop,
    _heterogeneous_param_names,
    _population_params_are_uniform,
)

_Q = Q88(data_width=16, fraction=8)


def _neuron_graph(taus: Sequence[float], *, v_leak: Sequence[float] | None = None) -> NeuronGraph:
    """Build a quantisable LIF network with the given per-neuron ``tau`` values."""
    n = len(taus)
    leak = np.zeros(n) if v_leak is None else np.asarray(v_leak, dtype=float)
    graph = nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([2])}),
            "aff": nir.Affine(weight=np.full((n, 2), 0.5), bias=np.zeros(n)),
            "lif": nir.LIF(
                tau=np.asarray(taus, dtype=float),
                r=np.ones(n),
                v_leak=leak,
                v_threshold=np.ones(n),
            ),
            "output": nir.Output(output_type={"output": np.array([n])}),
        },
        edges=[("input", "aff"), ("aff", "lif"), ("lif", "output")],
    )
    return from_scnetwork(from_nir(graph, dt=1.0), dt=1.0)


def _qgraph(taus: Sequence[float], *, v_leak: Sequence[float] | None = None) -> QuantisedGraph:
    """Quantise the LIF network with the given per-neuron ``tau`` values in Q8.8."""
    return quantise_graph(_neuron_graph(taus, v_leak=v_leak), _Q)


def _compile(
    taus: Sequence[float], interconnect: str, *, v_leak: Sequence[float] | None = None
) -> NetworkCompilationResult:
    """Compile the LIF network with the given per-neuron ``tau`` values to FPGA RTL."""
    return compile_network_to_fpga(
        _neuron_graph(taus, v_leak=v_leak), module_name="m", interconnect=interconnect
    )


__all__ = [
    "np",
    "pytest",
    "nir",
    "Q88",
    "compile_network_to_fpga",
    "from_nir",
    "from_scnetwork",
    "quantise_graph",
    "_can_fold",
    "_dequantised_pop",
    "_heterogeneous_param_names",
    "_population_params_are_uniform",
    "_Q",
    "_neuron_graph",
    "_qgraph",
    "_compile",
]
