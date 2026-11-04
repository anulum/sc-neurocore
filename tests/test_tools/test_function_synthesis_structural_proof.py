# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual structural mux geometry and finite proof admission

"""Exercise structural comparison on native muxes and a real oversized wire interface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.function_synthesis_netlist import Netlist
from tools.function_synthesis_structural_proof import compare_word_structure
from tools.function_synthesis_toolchain import SynthesisToolchain

StructuralCase = tuple[Netlist, SynthesisToolchain]


@pytest.fixture(scope="module")
def structural_case(tmp_path_factory: pytest.TempPathFactory) -> StructuralCase:
    """Acquire an actual conditional eight-bit function without folding its variable selector."""
    root = tmp_path_factory.mktemp("actual-structural-mux")
    tools = SynthesisToolchain.inspect(root / "inspection")
    (root / "unit.v").write_text(
        "module function_unit(input [7:0] argument0, output [7:0] result);\n"
        "assign result = argument0[0] ? argument0 : 8'd3;\nendmodule\n"
    )
    (root / "flow.ys").write_text(
        "read_verilog unit.v\nhierarchy -check -top function_unit\n"
        "proc\nopt_clean\nwrite_json unit.json\n"
    )
    tools.run([str(tools.yosys), "-Q", "-T", "flow.ys"], root, root / "native.log", timeout=90)
    return Netlist.read(root / "unit.json"), tools


def test_actual_mux_keeps_every_original_output_root(
    structural_case: StructuralCase, tmp_path: Path
) -> None:
    """Retain the full independent interface and every output when native structure agrees."""
    module, _ = structural_case
    result = compare_word_structure(module, module, {"argument0": 8}, 8, tmp_path / "same")
    assert result.matches and len(result.original_roots) == len(result.mapped_roots) == 8
    retained = {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in (tmp_path / "same").iterdir()
    }
    assert compare_word_structure(module, module, {"argument0": 8}, 8) == result
    assert retained == {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in (tmp_path / "same").iterdir()
    }
    assert sorted(p.name for p in tmp_path.iterdir()) == ["same"]


@pytest.mark.parametrize("damage", ["width", "unknown-parameter", "constant-driver"])
def test_damaged_actual_mux_geometry_or_driver_is_refused(
    structural_case: StructuralCase, tmp_path: Path, damage: str
) -> None:
    """Reject a malformed real mux or a lost original result driver without a positive record."""
    original, _ = structural_case
    document = json.loads(json.dumps({"modules": {"function_unit": original.raw}}))
    cell = next(
        c for c in document["modules"]["function_unit"]["cells"].values() if c["type"] == "$mux"
    )
    if damage == "width":
        cell["parameters"]["WIDTH"] = "0"
    elif damage == "unknown-parameter":
        cell["parameters"]["UNQUALIFIED"] = "1"
    else:
        cell["connections"]["Y"][0] = "0"
    path = tmp_path / "damaged.json"
    path.write_text(json.dumps(document))
    proof = tmp_path / "refused"
    with pytest.raises(ValueError):
        compare_word_structure(original, Netlist.read(path), {"argument0": 8}, 8, proof)
    assert not (proof / "structural-comparison.json").exists()


def test_actual_constant_one_selector_cannot_equal_the_variable_mux(
    structural_case: StructuralCase, tmp_path: Path
) -> None:
    """Keep a real changed selector visible across all original result roots."""
    original, _ = structural_case
    document = json.loads(json.dumps({"modules": {"function_unit": original.raw}}))
    cell = next(
        c for c in document["modules"]["function_unit"]["cells"].values() if c["type"] == "$mux"
    )
    cell["connections"]["S"] = ["1"]
    path = tmp_path / "constant-selector.json"
    path.write_text(json.dumps(document))
    result = compare_word_structure(
        original, Netlist.read(path), {"argument0": 8}, 8, tmp_path / "different"
    )
    assert not result.matches and len(result.original_roots) == len(result.mapped_roots) == 8


@pytest.mark.parametrize("change", ["none", "selector", "true-data", "false-data"])
def test_native_mux_width_reduction_preserves_only_equal_output_bits(
    structural_case: StructuralCase, tmp_path: Path, change: str
) -> None:
    """Admit native width reduction while retaining selector and both data dependencies.

    Parameters
    ----------
    structural_case : tuple
        Actual native toolchain acquired by the owning fixture.
    tmp_path : pathlib.Path
        Private native RTL, netlists and complete comparison artifacts.
    change : str
        Equal behavior or a changed selector, true input or false input.

    """
    _, tools = structural_case
    for name, selector, true_data, false_data in [
        ("original", "argument1[0]", "argument0", "8'd3"),
        (
            "candidate",
            "argument1[1]" if change == "selector" else "argument1[0]",
            "{argument0[6:0], argument0[7]}" if change == "true-data" else "argument0",
            "8'd2" if change == "false-data" else "8'd3",
        ),
    ]:
        (tmp_path / f"{name}.v").write_text(
            "module function_unit(input [7:0] argument0, input [7:0] argument1, "
            "output [7:0] result);\n"
            f"wire [63:0] wide = {selector} ? {{56'd0, ({true_data})}}"
            f" : {{56'd0, ({false_data})}};\n"
            "assign result = wide[7:0];\nendmodule\n"
        )
        (tmp_path / f"{name}.ys").write_text(
            f"read_verilog {name}.v\nhierarchy -check -top function_unit\nproc\nopt_clean\n"
            + ("opt\nwreduce\nopt\n" if name == "candidate" else "")
            + f"write_json {name}.json\n"
        )
        tools.run(
            [str(tools.yosys), "-Q", "-T", f"{name}.ys"],
            tmp_path,
            tmp_path / f"{name}.log",
            timeout=90,
        )
    original = Netlist.read(tmp_path / "original.json")
    candidate = Netlist.read(tmp_path / "candidate.json")
    original_widths = {
        len(cell.connections["Y"]) for cell in original.cells.values() if cell.kind == "$mux"
    }
    candidate_widths = {
        len(cell.connections["Y"]) for cell in candidate.cells.values() if cell.kind == "$mux"
    }
    assert original_widths == {64} and candidate_widths == {8}
    result = compare_word_structure(
        original, candidate, {"argument0": 8, "argument1": 8}, 8, tmp_path / "comparison"
    )
    assert result.matches is (change == "none")
    assert len(result.original_roots) == len(result.mapped_roots) == 8


def test_actual_oversized_argument_is_refused_before_structural_allocation(
    structural_case: StructuralCase, tmp_path: Path
) -> None:
    """Refuse a native argument outside the supported finite-word geometry."""
    _, tools = structural_case
    width = 200_001
    (tmp_path / "wide.v").write_text(
        f"module function_unit(input [{width - 1}:0] argument0, output [{width - 1}:0] result);\n"
        "assign result = argument0;\nendmodule\n"
    )
    (tmp_path / "wide.ys").write_text(
        "read_verilog wide.v\nhierarchy -check -top function_unit\nwrite_json wide.json\n"
    )
    tools.run(
        [str(tools.yosys), "-Q", "-T", "wide.ys"], tmp_path, tmp_path / "wide.log", timeout=90
    )
    module = Netlist.read(tmp_path / "wide.json")
    proof = tmp_path / "refused-budget"
    with pytest.raises(ValueError, match="positive supported words"):
        compare_word_structure(module, module, {"argument0": width}, width, proof)
    assert not proof.exists()


def test_actual_many_word_interface_cannot_exceed_the_structural_node_budget(
    structural_case: StructuralCase, tmp_path: Path
) -> None:
    """Bound independent expression identities while every native port has supported width."""
    _, tools = structural_case
    widths = {f"argument{index}": 64 for index in range(3_126)}
    ports = [f"input [63:0] {name}" for name in widths]
    ports.append("output [7:0] result")
    (tmp_path / "many-inputs.v").write_text(
        "module function_unit(" + ", ".join(ports) + ");\n"
        "assign result = argument0[7:0];\nendmodule\n"
    )
    (tmp_path / "many-inputs.ys").write_text(
        "read_verilog many-inputs.v\nhierarchy -check -top function_unit\n"
        "write_json many-inputs.json\n"
    )
    tools.run(
        [str(tools.yosys), "-Q", "-T", "many-inputs.ys"],
        tmp_path,
        tmp_path / "many-inputs.log",
        timeout=90,
    )
    module = Netlist.read(tmp_path / "many-inputs.json")
    proof = tmp_path / "refused-capacity"
    with pytest.raises(ValueError, match="node budget"):
        compare_word_structure(module, module, widths, 8, proof)
    assert not proof.exists()
