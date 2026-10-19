# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — equation-defined populations in the network compiler

"""Populations defined by equations compile in the direct and the folded interconnect.

A :class:`NeuronSpec` with a ``template`` carries its own equations, threshold,
reset and parameters (and optionally draws, a seed and a base stream). Both
interconnects build the same neuron; the folded sequencer keeps one step counter
for the network and gives each neuron the stream ``base + global index``, as the
direct path's per-instance ``P_RNG_STREAM`` override does. The spike rasters of
the two (Icarus) must be identical, and neurons with equal input must still spike
differently when they draw. Without Icarus the cosimulation cases skip.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from sc_neurocore.nir_bridge.fpga_folded_interconnect import can_fold
from sc_neurocore.nir_bridge.fpga_neuron_rtl import _population_neuron, population_stream
from tests.folded_interconnect_support import (
    _HAVE_IVERILOG,
    ConnectionSpec,
    NeuronGraph,
    NeuronSpec,
    _parity_rasters,
)

_NOISY = {
    "equations": ["dv/dt = -v / tau + I + sigma * randn()"],
    "threshold": "v > 1.0",
    "reset": "v = 0.0",
    "default_params": {"tau": 4.0, "sigma": 1.5},
    "rng_seed": 2026,
    "random_stream": 100,
}

_QUIET = {
    "equations": ["dv/dt = -v / tau + I"],
    "threshold": "v > 1.0",
    "reset": "v = 0.0",
    "default_params": {"tau": 4.0},
}


def _graph(template: Mapping[str, Any], *, recurrent: bool) -> tuple[NeuronGraph, list[float], int]:
    """Return a five-neuron equation population fed by two equal external currents."""
    n = 5
    pop = NeuronSpec(
        name="pop0", neuron_type="noisy_lif", n_neurons=n, dt=0.25, template=dict(template)
    )
    conns = [ConnectionSpec(src="stim", dst="pop0", weights=np.full((n, 2), 0.5, dtype=np.float32))]
    if recurrent:
        ring = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            ring[(i + 1) % n, i] = 0.25
        conns.append(ConnectionSpec(src="pop0", dst="pop0", weights=ring))
    graph = NeuronGraph(
        populations=[pop], connections=conns, input_pop="stim", output_pop="pop0", dt=0.25
    )
    return graph, [1.0, 1.0], n


def test_a_template_population_folds_and_builds_its_neuron() -> None:
    """can_fold accepts it; the neuron carries the template's seed, base stream and draws."""
    graph, _currents, _n = _graph(_NOISY, recurrent=False)
    pop = graph.populations[0]
    neuron = _population_neuron(pop.neuron_type, pop)
    assert neuron.uses_random_draws and neuron.random_seed == 2026
    assert [population_stream(pop, i) for i in (0, 4)] == [100, 104]


def test_a_template_may_not_shadow_a_canonical_type_or_lack_v() -> None:
    """The type name of a template must be its own, and the state v must exist."""
    pop = NeuronSpec(name="p", neuron_type="lif", n_neurons=1, template=_QUIET)
    with pytest.raises(ValueError, match="canonical"):
        _population_neuron("lif", pop)
    no_v = dict(_QUIET, equations=["du/dt = -u + I"], threshold="u > 1.0", reset="u = 0.0")
    pop = NeuronSpec(name="p", neuron_type="custom", n_neurons=1, template=no_v)
    with pytest.raises(ValueError, match="state 'v'"):
        _population_neuron("custom", pop)


def test_stochastic_detection_and_rising_edge_templates_do_not_fold() -> None:
    """The sequencer keeps no per-neuron LFSR or threshold history; such graphs stay direct."""
    crossing = dict(_QUIET, reset=None, detection="crossing")
    graph, _c, _n = _graph(crossing, recurrent=False)
    from sc_neurocore.compiler.equation_compiler import Q88
    from sc_neurocore.nir_bridge import quantise_graph

    assert not can_fold(quantise_graph(graph, Q88(data_width=16, fraction=8)), data_width=16)


@pytest.mark.skipif(not _HAVE_IVERILOG, reason="iverilog/vvp not installed")
@pytest.mark.parametrize("recurrent", [False, True])
def test_folded_equals_direct_with_draws(recurrent: bool) -> None:
    """Noisy equation neurons: identical rasters, spikes present, equal inputs spike differently."""
    graph, currents, n = _graph(_NOISY, recurrent=recurrent)
    direct, folded = _parity_rasters(graph, currents, n)
    assert len(direct) == len(folded) > 0
    assert direct == folded
    columns = ["".join(row[k] for row in direct) for k in range(n)]
    assert any("1" in column for column in columns)
    assert len(set(columns)) > 1, "each neuron draws from its own stream"


@pytest.mark.skipif(not _HAVE_IVERILOG, reason="iverilog/vvp not installed")
def test_folded_equals_direct_without_draws() -> None:
    """A deterministic equation population: identical rasters and identical neurons."""
    graph, currents, n = _graph(_QUIET, recurrent=False)
    direct, folded = _parity_rasters(graph, currents, n)
    assert direct == folded and any("1" in row for row in direct)
