# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for the intelligence HDL/numeric toolbox public API

"""Contracts for the public intelligence HDL/numeric-format toolbox.

These functions are exported from ``sc_neurocore.compiler.intelligence`` as a
reference toolbox (TCL project generation, VHDL wrappers, clock-domain-crossing
synchronisers, posit arithmetic, interconnect-morphology synthesis). They were
previously unexercised; these tests pin their generated-artefact contracts.
"""

from __future__ import annotations

import sc_neurocore.compiler.intelligence.core as intelligence_core
from sc_neurocore.compiler.intelligence import (
    Morphology,
    PositConfig,
    generate_cdc_synchroniser,
    generate_tcl_project,
    posit_decode,
    posit_encode,
    synthesize_morphology,
    verilog_to_vhdl_wrapper,
)


def test_tcl_project_vivado_includes_module_part_and_sources() -> None:
    """The Vivado TCL flow names the part, the module and each Verilog source."""
    tcl = generate_tcl_project(
        "sc_lif",
        tool="vivado",
        part="xc7a35tcpg236-1",
        verilog_files=["sc_lif.v", "sc_top.v"],
        constraint_file="sc_lif.xdc",
    )

    assert "sc_lif" in tcl
    assert "xc7a35tcpg236-1" in tcl
    assert "sc_lif.v" in tcl and "sc_top.v" in tcl
    assert "sc_lif.xdc" in tcl


def test_tcl_project_quartus_differs_from_vivado() -> None:
    """The Quartus flow emits a distinct script that still lists sources and constraints."""
    quartus = generate_tcl_project(
        "sc_lif",
        tool="quartus",
        verilog_files=["sc_lif.v"],
        constraint_file="sc_lif.sdc",
    )
    vivado = generate_tcl_project("sc_lif", tool="vivado")

    assert "sc_lif" in quartus
    assert "sc_lif.v" in quartus
    assert "sc_lif.sdc" in quartus
    assert quartus != vivado


def test_vhdl_wrapper_signed_and_unsigned_entities() -> None:
    """The VHDL wrapper produces an entity/architecture for both signedness modes."""
    signed = verilog_to_vhdl_wrapper("sc_lif", data_width=16, signed=True)
    unsigned = verilog_to_vhdl_wrapper("sc_lif", data_width=8, signed=False)

    for vhdl in (signed, unsigned):
        assert "entity sc_lif" in vhdl
        assert "architecture" in vhdl
    assert "signed" in signed
    assert "unsigned" in unsigned


def test_cdc_synchroniser_single_and_multi_bit() -> None:
    """The CDC synchroniser emits a register chain of the requested depth and width."""
    single = generate_cdc_synchroniser("ready", width=1, stages=2)
    multi = generate_cdc_synchroniser("count", width=8, stages=3, dst_clock="clk_core")

    assert "ready" in single
    assert "module" in single and "endmodule" in single
    assert "count" in multi
    assert "clk_core" in multi


def test_posit8_encodes_standard_anchor_bit_patterns() -> None:
    """posit8 (es=1) matches the canonical bit patterns for 1.0, 2.0 and 0.5."""
    config = PositConfig(nbits=8, es=1)

    assert posit_encode(1.0, config) == 0b01000000  # 64
    assert posit_encode(2.0, config) == 0b01010000  # 80
    assert posit_encode(0.5, config) == 0b00110000  # 48


def test_posit_round_trips_representable_values_and_signs() -> None:
    """Exactly representable magnitudes round-trip losslessly through encode/decode."""
    config = PositConfig(nbits=8, es=1)

    for value in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, -1.0, -2.5):
        assert posit_decode(posit_encode(value, config), config) == value


def test_posit_special_codes_and_saturation() -> None:
    """Zero, NaR and out-of-range magnitudes map to the reserved/saturated codes."""
    config = PositConfig(nbits=8, es=1)

    assert posit_encode(0.0, config) == 0
    assert posit_decode(0, config) == 0.0
    assert posit_encode(float("inf"), config) == 1 << (config.nbits - 1)
    assert posit_decode(1 << (config.nbits - 1), config) == float("inf")
    # A value far above maxpos saturates to the largest finite code, not NaR.
    saturated = posit_encode(1e9, config)
    assert saturated == (1 << (config.nbits - 1)) - 1


def test_posit16_is_accurate_for_irrational_values() -> None:
    """posit16 (es=2) reconstructs irrational/fractional values to small relative error."""
    config = PositConfig(nbits=16, es=2)

    assert config.useed == 16
    assert config.max_value > 0.0
    assert config.min_positive > 0.0
    for value in (0.1, 3.14159, -2.5):
        decoded = posit_decode(posit_encode(value, config), config)
        assert abs(decoded - value) / abs(value) < 1e-3


def test_synthesize_morphology_selects_topology_by_coupling() -> None:
    """Topology scales with inter-equation coupling: mesh when decoupled, denser when coupled."""
    decoupled = synthesize_morphology({"a": "1", "b": "2"})
    coupled = synthesize_morphology({"a": "b + c", "b": "a + c", "c": "a + b", "d": "a + b + c"})

    moderate = synthesize_morphology({"a": "b", "b": "a", "c": "a + b"})

    assert isinstance(decoupled, Morphology)
    assert decoupled.topology == "2D Mesh"
    assert decoupled.dimensions == 2
    assert moderate.topology == "3D Torus"
    assert moderate.dimensions == 3
    assert coupled.topology == "Hypercube"
    assert coupled.dimensions == 4
    assert coupled.bisection_bandwidth_gbps > 0.0


def test_intelligence_core_facade_reexports_toolbox() -> None:
    """The intelligence.core facade bundles the toolbox symbols for convenience imports."""
    assert intelligence_core.generate_tcl_project is generate_tcl_project
    assert intelligence_core.posit_encode is posit_encode
