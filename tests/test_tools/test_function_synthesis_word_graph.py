# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual word operand, alias and arithmetic admission

"""Validate independent operands and exact arithmetic against actual native graph faults."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sc_neurocore.compiler import emit_function_unit_verilog
from tools.function_synthesis_dsp_contracts import prove_dsp_contract
from tools.function_synthesis_exports import module_json_flow
from tools.function_synthesis_netlist import Bit, Netlist
from tools.function_synthesis_polynomial import ExactWordPolynomials
from tools.function_synthesis_toolchain import SynthesisToolchain
from tools.function_synthesis_word_graph import WordGraph

GraphCase = tuple[Netlist, Netlist]


@pytest.mark.parametrize("mode", ["0000101", "0010101", "1010101", "0110101"])
@pytest.mark.parametrize("a_input", ["DIRECT", "CASCADE"])
@pytest.mark.parametrize("b_input", ["DIRECT", "CASCADE"])
def test_python_dsp_words_match_the_actual_full_native_contract(
    tmp_path: Path, mode: str, a_input: str, b_input: str
) -> None:
    """Bind admitted Python DSP profiles and refusals to the retained full native contract."""
    tools = SynthesisToolchain.inspect(tmp_path / "inspection")
    native = tmp_path / "contract"
    contract = prove_dsp_contract((mode, a_input, b_input), tools, native, timeout=90)
    assert contract.to_dict()["independent_input_bits"] == 218
    document = json.loads((native / "paired-word.json").read_text())
    inputs = {
        name: len(port["bits"])
        for name, port in document["modules"]["contract_miter"]["ports"].items()
        if port["direction"] == "input"
    }
    widths = {name: width for name, width in inputs.items() if not name.startswith("control_")}
    assert sum(widths.values()) == 192 and len(inputs) - len(widths) == 26
    before = (native / "contract.v").read_bytes()
    modules: list[Netlist] = []
    for role in ("reference", "candidate"):
        root = tmp_path / role
        root.mkdir()
        ports = ", ".join(f"input [{width - 1}:0] {name}" for name, width in widths.items())
        connections = ", ".join(
            f".{name}(1'b0)" if name.startswith("control_") else f".{name}({name})"
            for name in inputs
        )
        (root / "wrapper.v").write_text(
            f"module function_unit({ports}, output [47:0] result);\n"
            f"{role} actual_contract({connections}, .result(result));\nendmodule\n"
        )
        (root / "flow.ys").write_text(
            "read_verilog -lib +/xilinx/cells_sim.v\n"
            "read_verilog -sv ../contract/contract.v wrapper.v\n"
            "hierarchy -check -top function_unit\nflatten\ndelete t:$scopeinfo\nproc\nopt_clean\n"
            + module_json_flow("word.json")
        )
        tools.run([str(tools.yosys), "-Q", "-T", "flow.ys"], root, root / "native.log", timeout=90)
        modules.append(Netlist.read(root / "word.json"))
    algebra = ExactWordPolynomials()
    reference = WordGraph(modules[0], algebra, widths, 48, reference_multiplication=True)
    if (mode, a_input, b_input) not in {
        ("0000101", "DIRECT", "DIRECT"),
        ("0010101", "DIRECT", "DIRECT"),
        ("1010101", "CASCADE", "DIRECT"),
        ("0110101", "DIRECT", "DIRECT"),
        ("1010101", "DIRECT", "CASCADE"),
    }:
        with pytest.raises(ValueError, match="no qualified arithmetic contract"):
            WordGraph(modules[1], algebra, widths, 48)
        assert (native / "contract.v").read_bytes() == before
        return
    candidate = WordGraph(modules[1], algebra, widths, 48)
    difference = algebra.residue(
        algebra.add(
            reference.read(modules[0].ports["result"].bits, False),
            algebra.scale(candidate.read(modules[1].ports["result"].bits, False), -1),
        ),
        48,
    )
    assert not difference
    assert (native / "contract.v").read_bytes() == before


@pytest.fixture(scope="module")
def graph_case(tmp_path_factory: pytest.TempPathFactory) -> GraphCase:
    """Acquire actual original multiplication and qualified DSP cascade checkpoints."""
    root = tmp_path_factory.mktemp("actual-word-graph-admission")
    tools = SynthesisToolchain.inspect(root / "inspection")
    (root / "unit.v").write_text(
        emit_function_unit_verilog("sin", data_width=8, fraction=4, function_lowering="v2")
    )
    for name, stage in [("original", "map_dsp"), ("mapped", "coarse")]:
        (root / (name + ".ys")).write_text(
            "read_verilog -sv unit.v\nhierarchy -check -top function_unit\nproc\nopt_clean\n"
            "synth_xilinx -top function_unit -family xc7 -noiopad -noclkbuf -run begin:"
            + stage
            + "\n"
            + module_json_flow(name + ".json")
        )
        log = root / (name + ".log")
        tools.run([str(tools.yosys), "-Q", "-T", name + ".ys"], root, log, timeout=90)
        tools.check_library_log(log)
    return Netlist.read(root / "original.json"), Netlist.read(root / "mapped.json")


def test_real_word_driver_cannot_disappear_into_a_constant_cascade_alias(
    graph_case: GraphCase, tmp_path: Path
) -> None:
    """Reject an actual word output shorted onto a constant native cascade connection."""
    document = json.loads(json.dumps({"modules": {"function_unit": graph_case[1].raw}}))
    module = document["modules"]["function_unit"]
    cascade = next(
        cell
        for cell in module["cells"].values()
        if cell["type"] == "DSP48E1"
        and cell["parameters"]["A_INPUT"] == "DIRECT"
        and any(bit in {"0", "1"} for bit in cell["connections"]["A"])
        and "ACOUT" in cell["connections"]
    )
    driver = next(cell for cell in module["cells"].values() if cell["type"] == "$add")
    assert isinstance(driver["connections"]["Y"][0], int)
    index = next(i for i, bit in enumerate(cascade["connections"]["A"]) if bit in {"0", "1"})
    cascade["connections"]["ACOUT"][index] = driver["connections"]["Y"][0]
    path = tmp_path / "constant-short.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="aliased constant or unknown"):
        WordGraph(Netlist.read(path), ExactWordPolynomials(), {"argument0": 8}, 8)


def test_actual_identity_cascade_keeps_the_independent_input_word(
    graph_case: GraphCase, tmp_path: Path
) -> None:
    """Preserve a valid direct B cascade when its output wire is the original primary itself."""
    mapped = graph_case[1]
    document = json.loads(json.dumps({"modules": {"function_unit": mapped.raw}}))
    module = document["modules"]["function_unit"]
    bits = tuple(module["ports"]["argument0"]["bits"])
    cell = next(
        c
        for c in module["cells"].values()
        if c["type"] == "DSP48E1"
        and c["parameters"]["B_INPUT"] == "DIRECT"
        and "BCOUT" in c["connections"]
        and set(c["connections"]["B"]) <= set(bits)
    )
    original_cascade = tuple(cell["connections"]["BCOUT"])
    aliases = dict(zip(original_cascade, cell["connections"]["B"], strict=True))
    for port in module["ports"].values():
        port["bits"] = [aliases.get(bit, bit) for bit in port["bits"]]
    for net in module["netnames"].values():
        net["bits"] = [aliases.get(bit, bit) for bit in net["bits"]]
    for instance in module["cells"].values():
        instance["connections"] = {
            port: [aliases.get(bit, bit) for bit in values]
            for port, values in instance["connections"].items()
        }
    path = tmp_path / "identity-cascade.json"
    path.write_text(json.dumps(document))
    algebra = ExactWordPolynomials()
    expected = WordGraph(mapped, algebra, {"argument0": 8}, 8).read(original_cascade, True)
    actual = WordGraph(Netlist.read(path), algebra, {"argument0": 8}, 8).read(
        tuple(aliases.get(bit, bit) for bit in original_cascade), True
    )
    assert actual == expected


@pytest.mark.parametrize(
    "damage",
    ["primary-driver", "aliased-inputs", "constant-conflict", "input-driver", "duplicate-driver"],
)
def test_actual_driver_or_cascade_faults_are_refused_by_graph_construction(
    graph_case: GraphCase, tmp_path: Path, damage: str
) -> None:
    """Reject ambiguous actual drivers and aliases which constrain an independent input."""
    module = graph_case[0] if damage in {"input-driver", "duplicate-driver"} else graph_case[1]
    document = json.loads(json.dumps({"modules": {"function_unit": module.raw}}))
    raw = document["modules"]["function_unit"]
    primary = raw["ports"]["argument0"]["bits"]
    if damage in {"input-driver", "duplicate-driver"}:
        name, cell = next((n, c) for n, c in raw["cells"].items() if c["type"] == "$add")
        if damage == "input-driver":
            cell["connections"]["Y"][0] = primary[0]
        else:
            raw["cells"][name + "_duplicate"] = json.loads(json.dumps(cell))
    elif damage in {"primary-driver", "aliased-inputs"}:
        cells = [
            c
            for c in raw["cells"].values()
            if c["type"] == "DSP48E1"
            and c["parameters"]["B_INPUT"] == "DIRECT"
            and "BCOUT" in c["connections"]
            and set(c["connections"]["B"]) <= set(primary)
        ]
        assert len(cells) >= 2
        if damage == "primary-driver":
            cells[0]["connections"]["BCOUT"][0] = primary[1]
            cells[0]["connections"]["B"][0] = primary[0]
        else:
            for index, cell in enumerate(cells[:2]):
                cell["connections"]["BCOUT"][0] = 999999
                cell["connections"]["B"][0] = primary[index]
    else:
        cells = [
            c
            for c in raw["cells"].values()
            if c["type"] == "DSP48E1"
            and c["parameters"]["A_INPUT"] == "DIRECT"
            and c["connections"]["A"][0] in {"0", "1"}
        ]
        first, second = next(
            (a, b)
            for a in cells
            for b in cells
            if a["connections"]["A"][0] != b["connections"]["A"][0]
        )
        for cell in [first, second]:
            cell["connections"]["ACOUT"] = list(cell["connections"]["A"])
            cell["connections"]["ACOUT"][0] = 999999
            cell["port_directions"]["ACOUT"] = "output"
    path = tmp_path / "damaged.json"
    path.write_text(json.dumps(document))
    expected = {
        "primary-driver": "constrains an original primary input",
        "aliased-inputs": "independent primary input bits were constrained",
        "constant-conflict": "conflicting constant aliases",
        "input-driver": "drives an original primary input",
        "duplicate-driver": "multiple non-alias drivers",
    }
    with pytest.raises(ValueError, match=expected[damage]):
        WordGraph(Netlist.read(path), ExactWordPolynomials(), {"argument0": 8}, 8)


@pytest.mark.parametrize("damage", ["empty", "crossed", "unknown", "undriven"])
def test_actual_local_product_boundaries_must_be_independent_and_driven(
    graph_case: GraphCase, damage: str
) -> None:
    """Refuse incomplete or self-dependent local domains before proving an actual product."""
    original = graph_case[0]
    cell = next(c for c in original.cells.values() if c.kind == "$mul")
    result = cell.connections["Y"]
    bits: tuple[Bit, ...] = {
        "empty": (),
        "crossed": (result[0],),
        "unknown": ("x",),
        "undriven": (999999,),
    }[damage]
    with pytest.raises(ValueError):
        WordGraph(
            original,
            ExactWordPolynomials(),
            {"argument0": 8},
            8,
            operands={"A": bits},
            result=result,
            reference_multiplication=True,
        )


@pytest.mark.parametrize(
    "damage", ["unmapped", "width", "signedness", "mixed", "cycle", "undriven"]
)
def test_actual_arithmetic_cones_require_qualified_widths_signs_and_dependencies(
    graph_case: GraphCase, tmp_path: Path, damage: str
) -> None:
    """Reject actual product faults rather than changing finite-word arithmetic assumptions."""
    original = graph_case[0]
    document = json.loads(json.dumps({"modules": {"function_unit": original.raw}}))
    raw = document["modules"]["function_unit"]
    name, cell = next(
        (n, c)
        for n, c in raw["cells"].items()
        if c["type"] == ("$add" if damage == "cycle" else "$mul")
    )
    if damage == "width":
        cell["parameters"]["A_WIDTH"] = format(len(cell["connections"]["A"]) + 1, "b")
    elif damage == "signedness":
        cell["parameters"]["A_SIGNED"] = "10"
    elif damage == "mixed":
        cell["parameters"]["A_SIGNED"], cell["parameters"]["B_SIGNED"] = "1", "0"
    elif damage == "cycle":
        cell["connections"]["A"][0] = cell["connections"]["Y"][0]
    path = tmp_path / "arithmetic.json"
    path.write_text(json.dumps(document))
    graph = WordGraph(
        Netlist.read(path),
        ExactWordPolynomials(),
        {"argument0": 8},
        8,
        reference_multiplication=damage != "unmapped",
    )
    bits = (999999,) if damage == "undriven" else tuple(cell["connections"]["Y"])
    expected = {
        "unmapped": "unmapped multiplication",
        "width": "parameter and port widths",
        "signedness": "single Boolean",
        "mixed": "mixed-signedness",
        "cycle": "cyclic product cone",
        "undriven": "undriven internal bit",
    }
    with pytest.raises(ValueError, match=expected[damage]):
        graph.read(bits, False)
