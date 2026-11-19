# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native common-product frontier admission

"""Reject dependent local atoms after actual native common-product elimination."""

from __future__ import annotations

from pathlib import Path

import pytest

from sc_neurocore.compiler import emit_function_unit_verilog
from tools.function_synthesis_exports import module_json_flow
from tools.function_synthesis_netlist import Netlist
from tools.function_synthesis_polynomial import ExactWordPolynomials
from tools.function_synthesis_toolchain import SynthesisToolchain
from tools.function_synthesis_word_graph import WordGraph, word_dependencies
from tools.function_synthesis_word_groups import prove_grouped_word_composition


def test_native_common_product_is_not_two_independent_consumer_frontiers(tmp_path: Path) -> None:
    """Keep full Sine RTL and its real duplicate product until a recorded native merge."""
    tools = SynthesisToolchain.inspect(tmp_path / "inspection")
    rtl = emit_function_unit_verilog("sin", data_width=16, fraction=8, function_lowering="v2")
    product = "wire signed [63:0] _fu0_t2 = _fu0_t0 * _fu0_t1;"
    assert rtl.count(product) == 1 and rtl.count("assign result =") == 1
    rtl = rtl.replace(product, "(* keep *) " + product).replace(
        "assign result =", "wire signed [15:0] original_sine;\nassign original_sine ="
    )
    rtl = rtl.replace(
        "endmodule",
        "(* keep *) wire signed [63:0] duplicate_product = _fu0_t0 * _fu0_t1;\n"
        "wire signed [31:0] product_pair = "
        "$signed(_fu0_t2[15:0]) * $signed(duplicate_product[15:0]);\n"
        "assign result = original_sine ^ product_pair[15:0];\nendmodule",
    )
    (tmp_path / "unit.v").write_text(rtl)
    (tmp_path / "flow.ys").write_text(
        "read_verilog -sv unit.v\nhierarchy -check -top function_unit\nproc\nopt_clean\n"
        "setattr -set keep 1 t:*\n"
        "synth_xilinx -top function_unit -family xc7 -noiopad -noclkbuf -run begin:map_dsp\n"
        + module_json_flow("original.json")
        + "setattr -unset keep t:$mul\nopt_merge t:$mul\n"
        "synth_xilinx -top function_unit -family xc7 -noiopad -noclkbuf -run map_dsp:coarse\n"
        "rename -enumerate function_unit/t:DSP48E1\n" + module_json_flow("mapped.json")
    )
    log = tmp_path / "yosys.log"
    tools.run([str(tools.yosys), "-Q", "-T", "-s", "flow.ys"], tmp_path, log, timeout=90)
    tools.check_library_log(log)
    original, mapped = (
        Netlist.read(tmp_path / (role + ".json")) for role in ["original", "mapped"]
    )
    widths = {"argument0": 16}
    gold = WordGraph(original, ExactWordPolynomials(), widths, 16, reference_multiplication=True)
    gate = WordGraph(mapped, ExactWordPolynomials(), widths, 16)
    word_dependencies(gold)
    word_dependencies(gate)
    left = tuple(gold.find(bit) for bit in original.names["_fu0_t2"])
    right = tuple(gold.find(bit) for bit in original.names["duplicate_product"])
    assert len(left) == len(right) == 64 and left != right
    assert tuple(gate.find(bit) for bit in mapped.names["_fu0_t2"]) == tuple(
        gate.find(bit) for bit in mapped.names["duplicate_product"]
    )
    directory = tmp_path / "refused"
    with pytest.raises(ValueError, match="Independent grouped frontier wires were aliased"):
        prove_grouped_word_composition(original, mapped, widths, 16, directory)
    assert not (directory / "word-composition.json").exists()
