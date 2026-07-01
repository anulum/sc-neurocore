# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Property-based fuzz tests for HDL source lowering

"""Property-based fuzz tests for stochastic-source HDL lowering inputs."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sc_neurocore.hdl_gen import emit_sources_from_ir
from sc_neurocore.hdl_gen._ident import sanitize_ident
from sc_neurocore.hdl_gen.verilog_generator import (
    _HALTON_SOURCE_TYPES,
    _LFSR_SOURCE_TYPES,
    _SOBOL_SOURCE_TYPES,
    _normalise,
)

# Every alias the lowerer actually accepts (lfsr/sobol/halton and their
# ``*_source`` / ``sc_*_source`` forms). The unknown-kind fuzz test skips any
# generated string that normalises into this set, so its expectation can never
# drift from the emitter's accepted vocabulary again.
_KNOWN_SOURCE_TYPES = _LFSR_SOURCE_TYPES | _SOBOL_SOURCE_TYPES | _HALTON_SOURCE_TYPES

_VALID_IDENT = st.from_regex(r"[A-Za-z_][A-Za-z0-9_]{0,30}", fullmatch=True).filter(
    lambda value: _is_valid_identifier(value)
)
_MALICIOUS_IDENT = st.text(min_size=1, max_size=80).filter(
    lambda value: not _is_valid_identifier(value)
)
_BAD_SEED = (
    st.none()
    | st.booleans()
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(max_size=16)
    | st.lists(st.integers(), max_size=2)
    | st.dictionaries(st.text(max_size=4), st.integers(), max_size=2)
)
_SOURCE_KIND = st.sampled_from(["lfsr", "lfsr16", "sobol", "sobol16"])


def _is_valid_identifier(value: str) -> bool:
    try:
        sanitize_ident(value)
    except ValueError:
        return False
    return True


@given(module_name=_VALID_IDENT, source_kind=_SOURCE_KIND, seed=st.integers())
@settings(max_examples=120, deadline=None)
def test_fuzz_emit_sources_from_ir_accepts_valid_names_and_integer_seeds(
    module_name: str, source_kind: str, seed: int
) -> None:
    verilog = emit_sources_from_ir(
        {
            "nodes": [
                {
                    "type": "StochasticSource",
                    "module_name": module_name,
                    "params": {"source_type": source_kind, "seed": seed},
                }
            ]
        }
    )

    assert f"module {module_name}" in verilog
    assert "endmodule" in verilog


@given(module_name=_MALICIOUS_IDENT, source_kind=_SOURCE_KIND)
@settings(max_examples=120, deadline=None)
def test_fuzz_emit_sources_from_ir_rejects_invalid_module_names(
    module_name: str, source_kind: str
) -> None:
    with pytest.raises(ValueError, match="stochastic source module name"):
        emit_sources_from_ir(
            {
                "nodes": [
                    {
                        "type": "StochasticSource",
                        "module_name": module_name,
                        "params": {"source_type": source_kind},
                    }
                ]
            }
        )


@given(node_id=_MALICIOUS_IDENT, source_kind=_SOURCE_KIND)
@settings(max_examples=120, deadline=None)
def test_fuzz_emit_sources_from_ir_rejects_invalid_mapping_node_ids(
    node_id: str, source_kind: str
) -> None:
    with pytest.raises(ValueError, match="stochastic source module name"):
        emit_sources_from_ir({"nodes": {node_id: {"type": source_kind}}})


@given(source_kind=_SOURCE_KIND, seed=_BAD_SEED)
@settings(max_examples=120, deadline=None)
def test_fuzz_emit_sources_from_ir_rejects_non_integer_seeds(
    source_kind: str, seed: object
) -> None:
    with pytest.raises(ValueError, match="seed"):
        emit_sources_from_ir(
            {
                "nodes": [
                    {
                        "type": "StochasticSource",
                        "module_name": "seed_source",
                        "params": {"source_type": source_kind, "seed": seed},
                    }
                ]
            }
        )


@given(
    source_kind=st.text(min_size=1, max_size=16).filter(
        lambda value: bool(value.strip()) and value.lower() != "lfsr"
    )
)
@settings(max_examples=80, deadline=None)
def test_fuzz_emit_sources_from_ir_rejects_unknown_source_kinds(source_kind: str) -> None:
    if _normalise(source_kind) in _KNOWN_SOURCE_TYPES:
        return

    with pytest.raises(ValueError, match="unsupported stochastic source type"):
        emit_sources_from_ir(
            {
                "nodes": [
                    {
                        "type": "StochasticSource",
                        "module_name": "unknown_source",
                        "params": {"source_type": source_kind},
                    }
                ]
            }
        )


@given(module_name=_VALID_IDENT, seed=st.integers())
@settings(max_examples=60, deadline=None)
def test_fuzz_emit_sources_from_ir_accepts_object_nodes(module_name: str, seed: int) -> None:
    node = SimpleNamespace(
        node_type="StochasticSource",
        module_name=module_name,
        params={"decorrelator": "sobol16", "seed": seed},
    )

    verilog = emit_sources_from_ir(SimpleNamespace(nodes=[node]))

    assert f"module {module_name}" in verilog
    assert "output reg [15:0] value" in verilog
