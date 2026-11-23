# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (ir_routing) from former test_stochastic_source_emitters.py

"""Stochastic source emission routed from an IR."""

from __future__ import annotations

import pytest
from sc_neurocore.hdl_gen import (
    emit_sources_from_ir,
    VerilogGenerator,
)
from types import (
    SimpleNamespace,
)


def test_verilog_generator_exposes_stochastic_source_helpers() -> None:
    """The Verilog generator emits LFSR and Sobol source modules with their seeds."""
    generator = VerilogGenerator()

    lfsr_verilog = generator.emit_lfsr16_source(seed=0xACE1)
    sobol_verilog = generator.emit_sobol16_source(seed=0x0042)

    assert "module sc_lfsr16_source" in lfsr_verilog
    assert "module sc_sobol16_source" in sobol_verilog
    assert "16'h0042" in sobol_verilog


def test_emit_sources_from_ir_accepts_mapping_nodes() -> None:
    """Stochastic sources are emitted from mapping nodes of an IR."""
    verilog = emit_sources_from_ir(
        {
            "nodes": [
                {
                    "name": "rng_lfsr",
                    "type": "StochasticSource",
                    "params": {"source_type": "LFSR", "seed": 0xBEEF},
                },
                {
                    "id": "rng_sobol",
                    "node_type": "StochasticSource",
                    "decorrelator": "Sobol",
                    "seed": 0x0042,
                },
                {"name": "dense0", "type": "Dense"},
            ]
        }
    )

    assert "module rng_lfsr" in verilog
    assert "16'hBEEF" in verilog
    assert "module rng_sobol" in verilog
    assert "16'h0042" in verilog
    assert "dense0" not in verilog


def test_verilog_generator_routes_stochastic_sources_from_ir() -> None:
    """The generator emits a source from an IR whose nodes are keyed by name."""
    generator = VerilogGenerator()

    verilog = generator.emit_sources_from_ir(
        {
            "nodes": {
                "source-a": {
                    "type": "lfsr16",
                    "module_name": "source_a",
                    "seed": 0,
                }
            }
        }
    )

    assert "module source_a" in verilog
    assert "16'hACE1" in verilog


def test_emit_sources_from_ir_accepts_object_nodes() -> None:
    """Stochastic sources are emitted from object nodes of an IR."""
    node = SimpleNamespace(
        module_name="object_sobol",
        node_type="StochasticSource",
        params={"decorrelator": "sobol16", "seed": 0x0017},
    )

    verilog = emit_sources_from_ir(SimpleNamespace(nodes=[node]))

    assert "module object_sobol" in verilog
    assert "16'h0017" in verilog


def test_emit_sources_from_ir_rejects_unknown_explicit_source_kind() -> None:
    """An unknown stochastic source type is refused."""
    with pytest.raises(ValueError, match="unsupported stochastic source type"):
        emit_sources_from_ir(
            {
                "nodes": [
                    {
                        "type": "StochasticSource",
                        "params": {"source_type": "NonexistentSource"},
                    }
                ]
            }
        )


def test_emit_sources_from_ir_rejects_duplicate_module_names() -> None:
    """Two sources with the same module name are refused."""
    with pytest.raises(ValueError, match="duplicate stochastic source module name"):
        emit_sources_from_ir(
            {
                "nodes": [
                    {"type": "lfsr16", "module_name": "shared_source"},
                    {"type": "sobol16", "module_name": "shared_source"},
                ]
            }
        )
