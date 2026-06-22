# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for SC-NIR conversion config, stochastic sources and hierarchy edges

"""Contracts for SC-NIR conversion config validation, source selection and naming edges."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from sc_neurocore.ir.scnir_convert import (
    SCNIRConversionConfig,
    _hierarchy_from_graph,
    _hierarchy_port_prefix,
    _nth_prime,
    _source,
    _stream_fragment,
    build_scnir_from_neuron_graph,
)
from sc_neurocore.ir.scnir_convert import (
    _MAX_SEED as MAX_SEED,
)


def _config(**overrides: Any) -> SCNIRConversionConfig:
    """A valid conversion config with optional field overrides."""
    base: dict[str, Any] = {"bitstream_length": 1024, "data_width": 16, "fraction": 8}
    base.update(overrides)
    return SCNIRConversionConfig(**base)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"bitstream_length": 0}, "bitstream_length must be a positive integer"),
        ({"data_width": 0}, "data_width must be a positive integer"),
        ({"fraction": -1}, "fraction must be a non-negative integer"),
        ({"fraction": 16}, "fraction must be smaller than data_width"),
        ({"accumulator_bits": 8}, "accumulator_bits must be greater"),
        ({"base_seed": -1}, "base_seed must fit in uint64"),
        ({"max_abs_correlation": 1.5}, "max_abs_correlation must be in"),
        ({"seed_domain": ""}, "seed_domain must be non-empty"),
        ({"producer": ""}, "producer must be non-empty"),
    ],
)
def test_conversion_config_validates_every_field(overrides: dict[str, Any], match: str) -> None:
    """Each SCNIRConversionConfig invariant rejects its out-of-range field."""
    with pytest.raises(ValueError, match=match):
        _config(**overrides)


def test_source_rejects_seed_allocation_beyond_uint64() -> None:
    """Allocating a per-stream seed past the uint64 ceiling is rejected."""
    config = _config(base_seed=MAX_SEED)

    with pytest.raises(ValueError, match="source seed allocation exceeds uint64"):
        _source(config, 1)


def test_source_emits_sobol_and_halton_descriptors() -> None:
    """The Sobol and Halton source kinds emit their dimension and prime-base metadata."""
    sobol = _source(_config(source_kind="sobol"), 2)
    halton = _source(_config(source_kind="halton"), 3)

    assert sobol.kind == "sobol"
    assert sobol.sobol_dimension == 3
    assert halton.kind == "halton"
    assert halton.halton_base == _nth_prime(4)


def test_nth_prime_returns_sequence_and_rejects_non_positive_index() -> None:
    """_nth_prime returns the n-th prime and rejects indices below one."""
    assert [_nth_prime(i) for i in (1, 2, 3, 4, 5)] == [2, 3, 5, 7, 11]

    with pytest.raises(ValueError, match="prime index must be positive"):
        _nth_prime(0)


def test_stream_fragment_falls_back_and_prefixes_non_alpha_start() -> None:
    """A wiped fragment defaults to 'stream' and a numeric start gains an 's_' prefix."""
    assert _stream_fragment("***") == "stream"
    assert _stream_fragment("123abc") == "s_123abc"


def test_hierarchy_port_prefix_maps_analogue_state_to_state() -> None:
    """An analogue-state signal maps to the 'state' port prefix; others pass through."""
    assert _hierarchy_port_prefix("analogue_state") == "state"
    assert _hierarchy_port_prefix("spike") == "spike"


def test_build_rejects_connection_without_destination_stream() -> None:
    """A connection whose destination has no population stream is rejected."""
    graph = SimpleNamespace(
        populations=[],
        connections=[SimpleNamespace(src="a", dst="ghost", weights=np.ones((1, 1)))],
        hierarchy=(),
    )

    with pytest.raises(ValueError, match="has no population stream"):
        build_scnir_from_neuron_graph(graph, config=_config())


def test_hierarchy_from_graph_rejects_instance_without_streams() -> None:
    """A hierarchy instance whose prefix matches no stream is rejected."""
    instance = SimpleNamespace(
        instance_id="inst0",
        module_name="mod",
        node_name_prefix="nomatch__",
    )
    graph = SimpleNamespace(populations=[], connections=[], hierarchy=(instance,))

    with pytest.raises(ValueError, match="did not produce any SC-NIR streams"):
        _hierarchy_from_graph(graph, ())


def test_hierarchy_from_graph_collects_prefixed_population_streams() -> None:
    """Hierarchy stream collection keeps prefixed populations and skips unrelated connections."""
    instance = SimpleNamespace(
        instance_id="sub",
        module_name="submod",
        node_name_prefix="sub__",
    )
    graph = SimpleNamespace(
        populations=[SimpleNamespace(name="sub__lif", neuron_type="lif")],
        connections=[SimpleNamespace(src="other", dst="elsewhere", weights=np.ones((1, 1)))],
        hierarchy=(instance,),
    )
    streams = build_scnir_from_neuron_graph(
        SimpleNamespace(
            populations=graph.populations,
            connections=[],
            hierarchy=(),
        ),
        config=_config(),
    ).streams

    result = _hierarchy_from_graph(graph, tuple(streams))

    assert len(result) == 1
    assert result[0].instance_id == "sub"
    assert result[0].ports
