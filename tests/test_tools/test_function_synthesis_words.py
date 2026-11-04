# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Exact word algebra and actual checkpoint contracts

"""Validate the native word-checkpoint composition contract across the complete original graph."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sc_neurocore.compiler import emit_function_unit_verilog
from tools.function_synthesis_exports import module_json_flow
from tools.function_synthesis_netlist import Netlist, NetlistCell
from tools.function_synthesis_polynomial import ExactWordPolynomials
from tools.function_synthesis_structural_proof import compare_word_structure
from tools.function_synthesis_toolchain import SynthesisToolchain
from tools.function_synthesis_word_graph import WordGraph, configured_dsp
from tools.function_synthesis_word_proof import prove_word_composition

WordCase = tuple[Path, Netlist, Netlist]


@pytest.fixture(scope="module")
def word_case(tmp_path_factory: pytest.TempPathFactory) -> WordCase:
    """Obtain both actual DSP checkpoints from emitted original RTL and native Yosys."""
    root = tmp_path_factory.mktemp("actual-word-checkpoints")
    tools = SynthesisToolchain.inspect(root / "inspection")
    (root / "unit.v").write_text(
        emit_function_unit_verilog("sin", data_width=8, fraction=4, function_lowering="v2")
    )
    for name, stage in [("pre", "map_dsp"), ("post", "coarse")]:
        script = root / (name + ".ys")
        script.write_text(
            "read_verilog -sv unit.v\nhierarchy -check -top function_unit\nproc\nopt_clean\n"
            "synth_xilinx -top function_unit -family xc7 -noiopad -noclkbuf -run begin:"
            + stage
            + "\n"
            + module_json_flow(name + ".json")
        )
        tools.run(
            [str(tools.yosys), "-Q", "-T", script.name], root, root / (name + ".log"), timeout=90
        )
        tools.check_library_log(root / (name + ".log"))
    (root / "normalize.ys").write_text(
        "read_verilog -sv unit.v\nhierarchy -check -top function_unit\n"
        "proc\nopt_clean\nmemory_map\nopt\nwreduce\nopt\n"
        "write_json original-normalized.json\n"
    )
    tools.run(
        [str(tools.yosys), "-Q", "-T", "normalize.ys"],
        root,
        root / "normalize.log",
        timeout=90,
    )
    return root, Netlist.read(root / "pre.json"), Netlist.read(root / "post.json")


def test_actual_word_products_and_all_original_function_edges(
    word_case: WordCase, tmp_path: Path
) -> None:
    """Prove every complete product and all unchanged functions of the real emitted unit."""
    _, original, mapped = word_case
    proof = prove_word_composition(original, mapped, {"argument0": 8}, 8, tmp_path / "proof")
    assert proof.product_count == 4 and proof.non_multiplier_count == 36
    assert proof.covered_product_cells
    assert proof.to_dict()["wire_bindings_checked"] == len(proof.wire_bindings)
    assert len(original.primary_inputs({"argument0": 8}, 8)) == 8


def test_structural_comparison_keeps_every_actual_original_result(
    word_case: WordCase, tmp_path: Path
) -> None:
    """Compare the same real stateless checkpoint without narrowing its full primary domain."""
    root, _, _ = word_case
    original = Netlist.read(root / "original-normalized.json")
    proof = compare_word_structure(original, original, {"argument0": 8}, 8, tmp_path / "same")
    assert proof.matches and len(proof.original_roots) == len(proof.mapped_roots) == 8
    assert proof.to_dict()["all_output_function_keys_identical"] is True


@pytest.mark.parametrize(
    "mutation",
    ["unknown", "undriven", "multiple-drivers", "input-driver", "output-port", "operator", "width"],
)
def test_malformed_actual_structural_dependencies_cannot_be_equal(
    word_case: WordCase, tmp_path: Path, mutation: str
) -> None:
    """Refuse unknown, undriven, ambiguous and uninterpretable actual function dependencies."""
    path = word_case[0] / "original-normalized.json"
    original = Netlist.read(path)
    document = json.loads(path.read_text())
    module = document["modules"]["function_unit"]
    cell_name, cell = next(iter(module["cells"].items()))
    if mutation == "unknown":
        module["ports"]["result"]["bits"][0] = "x"
        module["netnames"]["result"]["bits"][0] = "x"
    elif mutation == "undriven":
        module["ports"]["result"]["bits"][0] = 999999
        module["netnames"]["result"]["bits"][0] = 999999
    elif mutation == "multiple-drivers":
        module["cells"][cell_name + "_duplicate"] = copy.deepcopy(cell)
    elif mutation == "input-driver":
        cell["connections"]["Y"][0] = module["ports"]["argument0"]["bits"][0]
    elif mutation == "output-port":
        cell["connections"]["Z"] = cell["connections"].pop("Y")
        cell["port_directions"]["Z"] = cell["port_directions"].pop("Y")
    elif mutation == "operator":
        cell["type"] = "$unqualified_combinational_operation"
    else:
        width_parameter = "WIDTH" if cell["type"] == "$mux" else "Y_WIDTH"
        cell["parameters"][width_parameter] = "0"
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        compare_word_structure(
            original, Netlist.read(changed), {"argument0": 8}, 8, tmp_path / "proof"
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "kind",
        "registered",
        "missing-parameter",
        "arithmetic",
        "geometry",
        "direction",
        "unknown",
        "carry",
        "mode",
    ],
)
def test_actual_dsp_configuration_guards_are_not_arithmetic_assumptions(
    word_case: WordCase, mutation: str
) -> None:
    """Reject actual primitive faults before treating a DSP as its qualified word function."""
    document = json.loads((word_case[0] / "post.json").read_text())
    cell = next(
        cell
        for cell in document["modules"]["function_unit"]["cells"].values()
        if cell["type"] == "DSP48E1"
    )
    if mutation == "kind":
        cell["type"] = "DSP48E2"
    elif mutation == "registered":
        cell["parameters"]["MREG"] = "1"
    elif mutation == "missing-parameter":
        del cell["parameters"]["MREG"]
    elif mutation == "arithmetic":
        cell["parameters"]["USE_SIMD"] = "TWO24"
    elif mutation == "geometry":
        cell["connections"]["A"].pop()
    elif mutation == "direction":
        cell["port_directions"]["A"] = "output"
    elif mutation == "unknown":
        cell["connections"]["A"][0] = "x"
    elif mutation == "carry":
        cell["connections"]["CARRYIN"][0] = "1"
    else:
        cell["connections"]["OPMODE"] = ["0"] * 7
    with pytest.raises(ValueError):
        configured_dsp(NetlistCell.read(cell))


def test_actual_product_reader_preserves_output_port_and_free_operand_boundaries(
    word_case: WordCase,
) -> None:
    """Reject another output-port identity after reading a real product's complete result."""
    original = word_case[1]
    name, cell = next((name, cell) for name, cell in original.cells.items() if cell.kind == "$mul")
    graph = WordGraph(
        original,
        ExactWordPolynomials(),
        {"argument0": 8},
        8,
        operands={port: cell.connections[port] for port in ["A", "B"]},
        result=cell.connections["Y"],
        reference_multiplication=True,
    )
    graph.read(cell.connections["Y"], False)
    with pytest.raises(ValueError, match="unqualified word operation"):
        graph.raw(name, "Z")
    with pytest.raises(ValueError, match="unknown bit"):
        graph.read(("x",), False)
