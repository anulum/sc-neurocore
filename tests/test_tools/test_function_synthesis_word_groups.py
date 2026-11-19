# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual grouped arithmetic proof admission

"""Exercise complete native arithmetic groups, structural frontiers and global faults."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pytest

from sc_neurocore.compiler import emit_function_unit_verilog
from tools.function_synthesis_dsp_ports import configured_dsp
from tools.function_synthesis_exports import module_json_flow
from tools.function_synthesis_netlist import Netlist
from tools.function_synthesis_polynomial import ExactWordPolynomials
from tools.function_synthesis_toolchain import SynthesisToolchain
from tools.function_synthesis_word_graph import WordGraph, active_word_inputs, word_dependencies
from tools.function_synthesis_word_groups import (
    GroupedWordComposition,
    prove_grouped_word_composition,
)
from tools.function_synthesis_word_proof import WordComposition, prove_word_composition

GroupCase = tuple[Netlist, Netlist, dict[str, int], int]


@pytest.fixture(scope="module", params=[(16, 8), (18, 9), (32, 16)])
def group_case(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> GroupCase:
    """Acquire fresh actual complete pre/post DSP graphs for each distinct word geometry."""
    width, fraction = request.param
    root = tmp_path_factory.mktemp(f"actual-grouped-words-{width}")
    tools = SynthesisToolchain.inspect(root / "inspection")
    (root / "unit.v").write_text(
        emit_function_unit_verilog(
            "hypot", data_width=width, fraction=fraction, function_lowering="v2"
        )
    )
    for role, stage in [("original", "map_dsp"), ("mapped", "coarse")]:
        (root / (role + ".ys")).write_text(
            "read_verilog -sv unit.v\nhierarchy -check -top function_unit\nproc\nopt_clean\n"
            "synth_xilinx -top function_unit -family xc7 -noiopad -noclkbuf -run begin:"
            + stage
            + "\n"
            + module_json_flow(role + ".json")
        )
        log = root / (role + ".log")
        tools.run([str(tools.yosys), "-Q", "-T", role + ".ys"], root, log, timeout=90)
        tools.check_library_log(log)
    return (
        Netlist.read(root / "original.json"),
        Netlist.read(root / "mapped.json"),
        {"argument0": width, "argument1": width},
        width,
    )


@pytest.fixture(scope="module")
def mixed_group_case(tmp_path_factory: pytest.TempPathFactory) -> GroupCase:
    """Compose actual production Hypot and sine RTL into one independently driven native graph."""
    root = tmp_path_factory.mktemp("actual-mixed-function-words")
    tools = SynthesisToolchain.inspect(root / "inspection")
    for function in ["hypot", "sin"]:
        (root / (function + ".v")).write_text(
            emit_function_unit_verilog(function, data_width=16, fraction=8, function_lowering="v2")
        )
    (root / "mixed.v").write_text(
        "module function_unit(input [15:0] argument0, argument1, argument2, "
        "output [15:0] result);\nwire [15:0] geometric, periodic;\n"
        "hypot_unit h(.argument0(argument0), .argument1(argument1), .result(geometric));\n"
        "sin_unit s(.argument0(argument2), .result(periodic));\n"
        "assign result = geometric ^ periodic;\nendmodule\n"
    )
    for role, stage in [("original", "map_dsp"), ("mapped", "coarse")]:
        flow = role + ".ys"
        (root / flow).write_text(
            "read_verilog -sv hypot.v\nrename function_unit hypot_unit\n"
            "read_verilog -sv sin.v\nrename function_unit sin_unit\n"
            "read_verilog -sv mixed.v\nhierarchy -check -top function_unit\nflatten\n"
            "delete t:$scopeinfo\n"
            "proc\nopt_clean\nsynth_xilinx -top function_unit -family xc7 -noiopad "
            "-noclkbuf -run begin:" + stage + "\n" + module_json_flow(role + ".json")
        )
        log = root / (role + ".log")
        tools.run([str(tools.yosys), "-Q", "-T", flow], root, log, timeout=90)
        tools.check_library_log(log)
    return (
        Netlist.read(root / "original.json"),
        Netlist.read(root / "mapped.json"),
        {"argument0": 16, "argument1": 16, "argument2": 16},
        16,
    )


def test_multiple_production_functions_preserve_every_arithmetic_component(
    mixed_group_case: GroupCase, tmp_path: Path
) -> None:
    """Prove absorbed Hypot arithmetic and all disconnected sine products in the same real module."""
    proof = prove_grouped_word_composition(*mixed_group_case, tmp_path / "mixed")
    assert isinstance(proof, GroupedWordComposition)
    record = json.loads(json.dumps(proof.to_dict()))
    assert record["products_proven"] == 0 and record["products_covered_jointly"] == 6
    assert len(record["arithmetic_groups"]) > 1
    missing = mixed_group_case[0].cells.keys() - mixed_group_case[1].cells.keys()
    absorbed_addition = any(mixed_group_case[0].cells[name].kind == "$add" for name in missing)
    assert sum(len(g["source_members"]) == 1 for g in record["arithmetic_groups"]) == (
        4 if absorbed_addition else 6
    )
    members = [n for g in record["arithmetic_groups"] for n in g["source_members"]]
    mapped = [n for g in record["arithmetic_groups"] for n in g["mapped_members"]]
    assert len(members) == len(set(members)) and len(mapped) == len(set(mapped))
    assert len(members) + record["unchanged_functions_checked"] == len(mixed_group_case[0].cells)
    assert record["post_dsp_cell_count"] == len(mixed_group_case[1].cells)
    assert all(not r["difference"] for g in record["arithmetic_groups"] for r in g["roots"])


def test_actual_public_output_can_anchor_an_otherwise_unconsumed_complete_root_bit(
    mixed_group_case: GroupCase, tmp_path: Path
) -> None:
    """Expose an actual discarded product bit with its public port as the sole anchor."""
    original, mapped, widths, width = mixed_group_case
    certificate = json.loads(
        json.dumps(prove_grouped_word_composition(*mixed_group_case).to_dict())
    )
    used = {bit for cell in original.cells.values() for bit in active_word_inputs(cell)} | {
        bit for port in original.ports.values() for bit in port.bits
    }
    bit, actual = next(
        (a, b)
        for group in certificate["arithmetic_groups"]
        for root in group["roots"]
        for a, b in zip(root["source_bits"], root["mapped_bits"], strict=True)
        if isinstance(a, int) and a not in used
    )
    left, right = json.loads(json.dumps(original.raw)), json.loads(json.dumps(mapped.raw))
    anchor = 0
    for name in sorted(original.names.keys() & mapped.names.keys()):
        if name != "result" and bit in original.names[name]:
            for a, b in zip(original.names[name], mapped.names[name], strict=True):
                if a != bit:
                    key = f"complete_root_anchor_{anchor:05d}"
                    left["netnames"][key] = {"bits": [a]}
                    right["netnames"][key] = {"bits": [b]}
                    anchor += 1
            del right["netnames"][name]
    variants = []
    for role, module, endpoint in [("original", left, bit), ("mapped", right, actual)]:
        module["ports"]["result"]["bits"][-1] = endpoint
        module["netnames"]["result"]["bits"][-1] = endpoint
        path = tmp_path / (role + ".json")
        path.write_text(json.dumps({"modules": {"function_unit": module}}))
        variants.append(Netlist.read(path))
    proof = prove_grouped_word_composition(
        variants[0], variants[1], widths, width, tmp_path / "proved"
    )
    assert isinstance(proof, GroupedWordComposition)
    record = json.loads(json.dumps(proof.to_dict()))
    assert [bit, actual] in record["wire_bindings"]
    assert record["global_output_bits"][-1] == actual
    assert any(
        bit in root["source_bits"] for g in record["arithmetic_groups"] for root in g["roots"]
    )


def test_public_output_constant_keeps_complete_native_group_proofs(
    mixed_group_case: GroupCase, tmp_path: Path
) -> None:
    """Prove an identical constant endpoint alongside the complete actual nonlinear circuit."""
    variants = []
    for role, original in zip(["original", "mapped"], mixed_group_case[:2], strict=True):
        module = json.loads(json.dumps(original.raw))
        module["ports"]["result"]["bits"][-1] = "0"
        module["netnames"]["result"]["bits"][-1] = "0"
        path = tmp_path / (role + ".json")
        path.write_text(json.dumps({"modules": {"function_unit": module}}))
        variants.append(Netlist.read(path))
    proof = prove_grouped_word_composition(
        variants[0], variants[1], mixed_group_case[2], mixed_group_case[3], tmp_path / "proved"
    )
    assert isinstance(proof, GroupedWordComposition)
    record = json.loads(json.dumps(proof.to_dict()))
    assert record["global_output_bits"][-1] == "0"
    assert record["products_covered_jointly"] == 6
    assert all(isinstance(a, int) and isinstance(b, int) for a, b in record["wire_bindings"])


@pytest.mark.parametrize("damage", ["missing-anchor", "unknown-complete-root"])
def test_unconsumed_root_bits_still_require_a_complete_known_correspondence(
    mixed_group_case: GroupCase, tmp_path: Path, damage: str
) -> None:
    """Reject omitted or unknown discarded bits of a complete actual product word."""
    original, mapped, widths, width = mixed_group_case
    record = json.loads(json.dumps(prove_grouped_word_composition(*mixed_group_case).to_dict()))
    used = {bit for cell in original.cells.values() for bit in active_word_inputs(cell)} | {
        bit for port in original.ports.values() for bit in port.bits
    }
    name, port, index, bit = next(
        (root["cell"], root["port"], i, raw)
        for group in record["arithmetic_groups"]
        if len(group["source_members"]) == 1
        for root in group["roots"]
        for i, raw in reversed(list(enumerate(root["source_bits"])))
        if isinstance(raw, int) and raw not in used
    )
    left, right = json.loads(json.dumps(original.raw)), json.loads(json.dumps(mapped.raw))
    if damage == "missing-anchor":
        right["netnames"] = {
            n: d
            for n, d in right["netnames"].items()
            if n not in original.names or bit not in original.names[n]
        }
    else:
        left["cells"][name]["connections"][port][index] = "x"
        for value in left["netnames"].values():
            value["bits"] = ["x" if raw == bit else raw for raw in value["bits"]]
        for n in original.names.keys() & mapped.names.keys():
            for i, raw in enumerate(original.names[n]):
                if raw == bit:
                    right["netnames"][n]["bits"][i] = "x"
    variants = []
    for role, module in [("original", left), ("mapped", right)]:
        path = tmp_path / (role + ".json")
        path.write_text(json.dumps({"modules": {"function_unit": module}}))
        variants.append(Netlist.read(path))
    directory = tmp_path / "refused"
    with pytest.raises(ValueError, match="complete root lacks a correspondence"):
        prove_grouped_word_composition(variants[0], variants[1], widths, width, directory)
    assert not (directory / "word-composition.json").exists()


@pytest.mark.parametrize("fault", ["joint-root", "partition-cycle"])
def test_native_operand_fault_merges_overlapping_cones_before_refusing_the_full_word(
    mixed_group_case: GroupCase, tmp_path: Path, fault: str
) -> None:
    """Recompute actual overlap after a real cascade input is connected across arithmetic groups."""
    original, mapped, widths, width = mixed_group_case
    record = json.loads(json.dumps(prove_grouped_word_composition(*mixed_group_case).to_dict()))
    periodic = [
        group
        for group in record["arithmetic_groups"]
        if all("sin.v:" in name for name in group["source_members"])
    ]
    assert len(periodic) == 4
    left, right = (
        (record["arithmetic_groups"][0], periodic[0])
        if fault == "joint-root"
        else (periodic[0], periodic[2])
    )
    frontier = {b for _, b in right["frontier"]}
    wire = next(
        b
        for root in left["roots"]
        for b in root["mapped_bits"]
        if isinstance(b, int) and b not in frontier
    )
    graph = WordGraph(mapped, ExactWordPolynomials(), widths, width)
    module = json.loads(json.dumps(mapped.raw))
    name, port, index = next(
        (name, port, i)
        for name in right["mapped_members"]
        if mapped.cells[name].kind == "DSP48E1"
        for port in (
            {"DIRECT": "A", "CASCADE": "ACIN"}[configured_dsp(mapped.cells[name])[1]],
            {"DIRECT": "B", "CASCADE": "BCIN"}[configured_dsp(mapped.cells[name])[2]],
        )
        for i, bit in enumerate(mapped.cells[name].connections[port])
        if graph.find(bit) in frontier
    )
    module["cells"][name]["connections"][port][index] = wire
    path = tmp_path / "crossed-cascade.json"
    path.write_text(json.dumps({"modules": {"function_unit": module}}))
    changed = Netlist.read(path)
    assert word_dependencies(WordGraph(changed, ExactWordPolynomials(), widths, width))
    directory = tmp_path / "refused"
    message = {"joint-root": "equality was not established", "partition-cycle": "contains a cycle"}
    with pytest.raises(ValueError, match=message[fault]):
        prove_grouped_word_composition(original, changed, widths, width, directory)
    if fault == "joint-root":
        refusal = json.loads((directory / "group-refusal.json").read_text())
        assert set(refusal["source_members"]) == set(left["source_members"]) | set(
            right["source_members"]
        )
        assert refusal["difference"] and len(refusal["source_bits"]) == refusal["width"]
    assert not (directory / "word-composition.json").exists()


def test_complete_native_group_proves_joint_roots_and_keeps_physical_cell_count(
    group_case: GroupCase,
    tmp_path: Path,
) -> None:
    """Prove full removed arithmetic without relabelling two joint products as individual proofs."""
    proof = prove_grouped_word_composition(*group_case, tmp_path / "complete")
    assert isinstance(proof, GroupedWordComposition)
    record = json.loads(json.dumps(proof.to_dict()))
    assert json.loads((tmp_path / "complete/word-composition.json").read_text()) == proof.to_dict()
    assert record["schema_version"] == "sc-neurocore.word-composition.v2"
    assert record["products_proven"] == 0 and record["products_covered_jointly"] == 2
    assert record["post_dsp_cell_count"] == len(group_case[1].cells)
    assert len(record["globally_unobserved_mapped_cells"]) == 12
    members = [name for group in record["arithmetic_groups"] for name in group["source_members"]]
    assert len(members) == len(set(members))
    assert len(members) + record["unchanged_functions_checked"] == len(group_case[0].cells)
    roots = [root for group in record["arithmetic_groups"] for root in group["roots"]]
    assert any(root["width"] == 64 for root in roots)
    assert all(
        len(root["source_bits"]) == len(root["mapped_bits"]) == root["width"] for root in roots
    )
    assert all(root["difference"] == [] for root in roots)
    ledger = proof.work.to_dict()
    assert ledger["expression_term_limit"] == 100000
    assert 0 < proof.work.peak <= 100000 and sum(proof.work.terms.values()) > 100000
    assert {"add", "scale", "floor", "slice", "multiply", "signed_wrap"} <= proof.work.calls.keys()
    before = {
        p: (p.read_bytes(), p.stat().st_mode, p.stat().st_mtime_ns)
        for p in (tmp_path / "complete").iterdir()
    }
    assert prove_grouped_word_composition(*group_case).to_dict() == proof.to_dict()
    assert before == {p: (p.read_bytes(), p.stat().st_mode, p.stat().st_mtime_ns) for p in before}


def test_production_word_dispatch_matches_the_actual_removed_native_functions(
    group_case: GroupCase, tmp_path: Path
) -> None:
    """Select joint or individual proofs from real removed functions without assuming an SDK layout."""
    original, mapped, widths, width = group_case
    missing = original.cells.keys() - mapped.cells.keys()
    absorbed = any(original.cells[name].kind != "$mul" for name in missing)
    proof = prove_word_composition(original, mapped, widths, width, tmp_path / "dispatched")
    if absorbed:
        assert isinstance(proof, GroupedWordComposition)
        assert proof.to_dict()["products_covered_jointly"] == 2
        assert proof.to_dict()["products_proven"] == 0
    else:
        assert isinstance(proof, WordComposition)
        assert proof.product_count == 2
        assert proof.non_multiplier_count + proof.product_count == len(original.cells)


def test_equivalent_native_input_ordering_has_identical_semantic_certificate(
    group_case: GroupCase,
    tmp_path: Path,
) -> None:
    """Reordering JSON mappings cannot alter components, alias representatives or frontier variables."""
    original, mapped, widths, width = group_case
    expected = prove_grouped_word_composition(*group_case).to_dict()
    variants = []
    for index, graph in enumerate((original, mapped)):
        module = json.loads(json.dumps(graph.raw))
        for key in ["cells", "ports", "netnames"]:
            module[key] = dict(reversed(list(module[key].items())))
        for cell in module["cells"].values():
            for key in ["parameters", "connections", "port_directions"]:
                cell[key] = dict(reversed(list(cell[key].items())))
        path = tmp_path / f"permuted-{index}.json"
        path.write_text(json.dumps({"modules": {"function_unit": module}}))
        variants.append(Netlist.read(path))
    assert (
        prove_grouped_word_composition(variants[0], variants[1], widths, width).to_dict()
        == expected
    )


def test_globally_unobserved_native_dsp_cannot_feed_an_unchecked_register(
    group_case: GroupCase, tmp_path: Path
) -> None:
    """A state consumer of otherwise unobserved arithmetic invalidates complete combinational sinks."""
    original, mapped, widths, width = group_case
    certificate = json.loads(json.dumps(prove_grouped_word_composition(*group_case).to_dict()))
    module = json.loads(json.dumps(mapped.raw))
    cells = module["cells"]
    extra = cells[certificate["globally_unobserved_mapped_cells"][0]]
    assert extra["type"] == "DSP48E1"
    signal = next(bit for bit in extra["connections"]["P"] if isinstance(bit, int))
    fresh = 1 + max(
        bit
        for cell in cells.values()
        for bits in cell["connections"].values()
        for bit in bits
        if isinstance(bit, int)
    )
    cells["unexpected_product_register"] = {
        "type": "$dff",
        "parameters": {"WIDTH": "1", "CLK_POLARITY": "1"},
        "port_directions": {"CLK": "input", "D": "input", "Q": "output"},
        "connections": {
            "CLK": [module["ports"]["argument0"]["bits"][0]],
            "D": [signal],
            "Q": [fresh],
        },
    }
    path = tmp_path / "unobserved-product-state.json"
    path.write_text(json.dumps({"modules": {"function_unit": module}}))
    directory = tmp_path / "refused"
    with pytest.raises(ValueError, match="contains state"):
        prove_grouped_word_composition(original, Netlist.read(path), widths, width, directory)
    assert not (directory / "word-composition.json").exists()


@pytest.mark.parametrize(
    "damage",
    [
        "state",
        "extra-output",
        "changed-output-driver",
        "changed-output-name",
        "productive-dsp",
        "partial-operand-swap",
        "same-name-driver",
        "postadder",
        "unknown",
        "cycle",
        "changed-function",
        "internal-escape",
        "primary-alias",
        "primary-constant",
        "memory-declaration",
        "constant-anchor",
        "conflicting-anchor",
        "vector-geometry",
        "renamed-nonarithmetic",
        "unobserved-nonarithmetic",
        "unobserved-source-product",
    ],
)
def test_actual_grouped_graph_faults_never_publish_a_positive_certificate(
    group_case: GroupCase,
    tmp_path: Path,
    damage: str,
) -> None:
    """Reject real mapped arithmetic, frontier, state and escaped-output mutations."""
    original, mapped, widths, width = group_case
    proof = prove_grouped_word_composition(*group_case)
    certificate = json.loads(json.dumps(proof.to_dict()))
    module = json.loads(json.dumps(mapped.raw))
    cells = module["cells"]
    group = certificate["arithmetic_groups"][0]
    if damage == "unobserved-source-product":
        source_module = json.loads(json.dumps(original.raw))
        extra = json.loads(
            json.dumps(next(c for c in source_module["cells"].values() if c["type"] == "$mul"))
        )
        fresh = 1 + max(
            bit
            for cell in source_module["cells"].values()
            for bits in cell["connections"].values()
            for bit in bits
            if isinstance(bit, int)
        )
        extra["connections"]["Y"] = list(range(fresh, fresh + len(extra["connections"]["Y"])))
        source_module["cells"]["unobserved_source_product"] = extra
        source_path = tmp_path / "unobserved-original-product.json"
        source_path.write_text(json.dumps({"modules": {"function_unit": source_module}}))
        original = Netlist.read(source_path)
    elif damage == "memory-declaration":
        module["memories"] = {"unexpected_memory": {"width": 1, "size": 1, "start_offset": 0}}
    elif damage in {"constant-anchor", "conflicting-anchor"}:
        source_module = json.loads(json.dumps(original.raw))
        before = (
            "0" if damage == "constant-anchor" else source_module["ports"]["argument0"]["bits"][0]
        )
        after = "1" if damage == "constant-anchor" else module["ports"]["argument1"]["bits"][0]
        source_module["netnames"]["inconsistent_anchor"] = {"bits": [before]}
        module["netnames"]["inconsistent_anchor"] = {"bits": [after]}
        source_path = tmp_path / "original-anchor.json"
        source_path.write_text(json.dumps({"modules": {"function_unit": source_module}}))
        original = Netlist.read(source_path)
    elif damage == "vector-geometry":
        name = next(
            n for n in original.names.keys() & mapped.names.keys() if n not in original.ports
        )
        module["netnames"][name]["offset"] = original.offsets[name] + 1
    elif damage == "renamed-nonarithmetic":
        name = next(n for n, c in original.cells.items() if c.kind == "$lt" and n in cells)
        cells[name + "_renamed"] = cells.pop(name)
    elif damage == "unobserved-nonarithmetic":
        extra = json.loads(json.dumps(next(c for c in cells.values() if c["type"] == "$neg")))
        fresh = 1 + max(
            bit
            for cell in cells.values()
            for bits in cell["connections"].values()
            for bit in bits
            if isinstance(bit, int)
        )
        extra["connections"]["Y"] = list(range(fresh, fresh + len(extra["connections"]["Y"])))
        cells["unaccounted_negation"] = extra
    elif damage == "state":
        fresh = 1 + max(
            bit
            for cell in cells.values()
            for bits in cell["connections"].values()
            for bit in bits
            if isinstance(bit, int)
        )
        cells["unexpected_state"] = {
            "type": "$dff",
            "parameters": {"WIDTH": "1", "CLK_POLARITY": "1"},
            "port_directions": {"CLK": "input", "D": "input", "Q": "output"},
            "connections": {
                "CLK": ["0"],
                "D": [module["ports"]["result"]["bits"][0]],
                "Q": [fresh],
            },
        }
    elif damage in {"changed-output-driver", "changed-output-name"}:
        bit = module["ports"]["argument0"]["bits"][0]
        module["netnames"]["result"]["bits"][-1] = bit
        if damage == "changed-output-driver":
            module["ports"]["result"]["bits"][-1] = bit
    elif damage in {"extra-output", "internal-escape"}:
        extra = cells[certificate["globally_unobserved_mapped_cells"][0]]
        bit = next(bit for bit in extra["connections"]["P"] if isinstance(bit, int))
        if damage == "extra-output":
            module["ports"]["result"]["bits"][-1] = bit
            module["netnames"]["result"]["bits"][-1] = bit
        else:
            cell = next(
                cells[name]
                for name in original.cells
                if name in cells and cells[name]["type"] == "$mux"
            )
            cell["connections"]["A"][0] = bit
    elif damage in {"productive-dsp", "partial-operand-swap"}:
        frontier = {right for _, right in group["frontier"]}
        graph = WordGraph(mapped, ExactWordPolynomials(), widths, width)
        name, port, index = next(
            (name, port, index)
            for name in group["mapped_members"]
            if cells[name]["type"] == "DSP48E1"
            for port in [
                {"DIRECT": "A", "CASCADE": "ACIN"}[configured_dsp(mapped.cells[name])[1]],
                {"DIRECT": "B", "CASCADE": "BCIN"}[configured_dsp(mapped.cells[name])[2]],
            ]
            for index, bit in enumerate(cells[name]["connections"][port])
            if graph.find(bit) in frontier
        )
        if damage == "productive-dsp":
            cells[name]["connections"][port][index] = "0"
        else:
            values = cells[name]["connections"][port]
            second = next(
                i
                for i, bit in enumerate(values)
                if graph.find(bit) in frontier and graph.find(bit) != graph.find(values[index])
            )
            values[index], values[second] = values[second], values[index]
    elif damage == "same-name-driver":
        frontier = {left for left, _ in group["frontier"]}
        name, cell = next(
            (name, cells[name])
            for name, original_cell in original.cells.items()
            if name in cells
            and any(
                bit in frontier
                for port, values in original_cell.connections.items()
                if original_cell.directions[port] == "output"
                for bit in values
            )
        )
        port = next(
            port
            for port, values in cell["connections"].items()
            if cell["port_directions"][port] == "input" and values
        )
        cell["connections"][port][0] = module["ports"]["argument1"]["bits"][-1]
    elif damage == "postadder":
        cell = next(
            cells[name]
            for name in group["mapped_members"]
            if cells[name]["type"] == "DSP48E1"
            and "".join(reversed(cells[name]["connections"]["OPMODE"])) != "0000101"
        )
        for parameter, cascade, direct in [
            ("A_INPUT", "ACIN", "A"),
            ("B_INPUT", "BCIN", "B"),
        ]:
            if cell["parameters"][parameter] == "CASCADE":
                cell["connections"][direct] = list(cell["connections"][cascade])
                cell["parameters"][parameter] = "DIRECT"
        cell["connections"]["OPMODE"] = list(reversed("0000101"))
    elif damage in {"unknown", "cycle", "changed-function"}:
        cell = next(c for c in cells.values() if c["type"] == "$add")
        if damage == "changed-function":
            cell["type"] = "$sub"
        else:
            cell["connections"]["A"][0] = (
                "x" if damage == "unknown" else cell["connections"]["Y"][0]
            )
    else:
        bits = module["ports"]["argument0"]["bits"]
        bits[0] = bits[1] if damage == "primary-alias" else "0"
        module["netnames"]["argument0"]["bits"] = list(bits)
    path = tmp_path / "actual-fault.json"
    path.write_text(json.dumps({"modules": {"function_unit": module}}))
    directory = tmp_path / "refused"
    messages = {
        "memory-declaration": "memory declarations changed",
        "constant-anchor": "constant or unknown correspondence changed",
        "conflicting-anchor": "wire has conflicting",
        "vector-geometry": "different actual geometry",
        "renamed-nonarithmetic": "disappeared arithmetic functions only",
        "unobserved-nonarithmetic": "no arithmetic disposition",
        "unobserved-source-product": "lacks a complete observable boundary",
        "changed-output-driver": "original output lacks a complete proved grouped driver",
        "changed-output-name": "named original port differs from its declared wire connections",
    }
    with pytest.raises(ValueError, match=messages.get(damage, ".+")):
        prove_grouped_word_composition(original, Netlist.read(path), widths, width, directory)
    assert not (directory / "word-composition.json").exists()
    if damage in {"productive-dsp", "partial-operand-swap", "postadder"}:
        with pytest.raises(ValueError, match="Complete grouped arithmetic word equality"):
            prove_grouped_word_composition(original, Netlist.read(path), widths, width)
        refusal = json.loads((directory / "group-refusal.json").read_text())
        assert refusal["status"] == "equality-not-established" and refusal["difference"]
        assert refusal["witness_scope"] == "independent-local-frontier-not-primary-reachability"
        witness = refusal["witness"]
        if witness is not None:
            assert all(
                atom[0] == "bit" for term in refusal["difference"] for atom in term["factors"]
            )
            value = sum(
                term["coefficient"] * math.prod(witness[atom[1]] for atom in term["factors"])
                for term in refusal["difference"]
            )
            assert value % (1 << refusal["width"]) != 0


def test_reader_cannot_reset_an_exhausted_enclosing_group_budget(group_case: GroupCase) -> None:
    """An inherited expired deadline remains exhaustion on read-only recomputation."""
    with pytest.raises(ValueError, match="exhausted the shared case budget"):
        prove_grouped_word_composition(*group_case, deadline=time.monotonic() - 1)


@pytest.mark.parametrize("deadline", [True, float("nan"), float("inf")])
def test_invalid_group_deadline_is_refused_before_retained_output(
    group_case: GroupCase,
    tmp_path: Path,
    deadline: float,
) -> None:
    """Invalid budget controls cannot create a successful retained certificate."""
    directory = tmp_path / "invalid"
    with pytest.raises(ValueError, match="finite inherited deadline"):
        prove_grouped_word_composition(*group_case, directory, deadline=deadline)
    assert not directory.exists()
