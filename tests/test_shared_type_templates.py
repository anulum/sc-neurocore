# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — one neuron type, one definition: shared RTL modules never run another template

"""Populations of one neuron type share one module, so their templates must agree.

Two populations named with the same type but different map equations (0.25 and 0.5)
were accepted and both ran the first population's module, in either order and in
both interconnects (RIGOR NEURO-FPGA-TEMPLATE-CACHE-01). Each differing part of the
definition — equation, initial value, detection, spike value, threshold — is now
refused before any RTL is emitted, in both population orders and by the folded
builder itself; identical templates (also with different random streams, which are
per-instance ports) still share a module, and distinct type names compile and give
each population its own dynamics in direct and folded co-simulation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from sc_neurocore.nir_bridge.fpga_compiler import compile_network_to_fpga
from tests.folded_interconnect_support import (
    _DW,
    _FR,
    _HAVE_IVERILOG,
    Q88,
    ConnectionSpec,
    NeuronGraph,
    NeuronSpec,
    _build_top_folded,
    _parity_rasters,
    quantise_graph,
)

_BASE: dict[str, Any] = {
    "equations": ["dv/dt = 0.25"],
    "threshold": "v > 2.0",
    "reset": "v = 0.0",
    "method": "map",
    "default_params": {},
}


def _variant(**changes: Any) -> dict[str, Any]:
    """Return the base map template with ``changes`` applied."""
    return {**_BASE, **changes}


def _graph(first: dict[str, Any], second: dict[str, Any], types: tuple[str, str]) -> NeuronGraph:
    """Return stim → population ``a`` and stim → population ``b`` with the given templates."""
    a = NeuronSpec(name="a", neuron_type=types[0], n_neurons=1, dt=1.0, template=first)
    b = NeuronSpec(name="b", neuron_type=types[1], n_neurons=1, dt=1.0, template=second)
    conns = [
        ConnectionSpec(src="stim", dst="a", weights=np.ones((1, 1), np.float32)),
        ConnectionSpec(src="a", dst="b", weights=np.ones((1, 1), np.float32)),
    ]
    return NeuronGraph(populations=[a, b], connections=conns, input_pop="stim", output_pop="b")


_DIFFERENCES: dict[str, dict[str, Any]] = {
    "equation": {"equations": ["dv/dt = 0.5"]},
    "init": {"init": {"v": 0.125}},
    "detection": {"detection": "edge"},
    "spike_value": {"spike_value": "v"},
    "threshold": {"threshold": "v > 1.0"},
}


@pytest.mark.parametrize("order", ["base-first", "variant-first"])
@pytest.mark.parametrize("difference", sorted(_DIFFERENCES))
def test_different_definitions_under_one_type_are_refused(difference: str, order: str) -> None:
    """compile_network_to_fpga refuses before emitting RTL, in either population order."""
    templates = (_BASE, _variant(**_DIFFERENCES[difference]))
    if order == "variant-first":
        templates = templates[::-1]
    with pytest.raises(ValueError, match="define different neurons"):
        compile_network_to_fpga(_graph(*templates, ("shared", "shared")), module_name="m")


def test_the_folded_builder_refuses_them_itself() -> None:
    """The folded builder checks the definitions too, not only the public compiler."""
    qgraph = quantise_graph(
        _graph(_BASE, _variant(equations=["dv/dt = 0.5"]), ("shared", "shared")),
        Q88(data_width=_DW, fraction=_FR),
    )
    with pytest.raises(ValueError, match="define different neurons"):
        _build_top_folded("m", qgraph, data_width=_DW, fraction=_FR)


def test_identical_definitions_still_share_one_module() -> None:
    """Same template (random stream aside) under one type: one module, compiles."""
    result = compile_network_to_fpga(
        _graph(_BASE, _variant(random_stream=7), ("shared", "shared")), module_name="m"
    )
    assert list(result.neuron_modules) == ["shared"]


@pytest.mark.skipif(not _HAVE_IVERILOG, reason="iverilog/vvp not installed")
@pytest.mark.parametrize("order", ["quarter-first", "half-first"])
def test_distinct_type_names_run_their_own_dynamics(order: str) -> None:
    """Map v += 0.25 and v += 0.5 under two types fire every fourth and every second step."""
    quarter = _variant(equations=["dv/dt = v + 0.25"], threshold="v > 0.9")
    half = _variant(equations=["dv/dt = v + 0.5"], threshold="v > 0.9")
    first, second = (quarter, half) if order == "quarter-first" else (half, quarter)
    direct, folded = _parity_rasters(_graph(first, second, ("t_first", "t_second")), [0.0], 2)
    assert folded == direct
    # The raster's first column is population b (the second), the last is a.
    second_col, first_col = ([row[k] for row in direct] for k in (0, 1))
    quarter_col, half_col = (
        (first_col, second_col) if order == "quarter-first" else (second_col, first_col)
    )
    assert half_col.count("1") > quarter_col.count("1") > 0
