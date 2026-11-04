# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual primitive residual domains and invalid graphs

"""Prove complete native residuals and refuse faults in actual DSP checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sc_neurocore.compiler import emit_function_unit_verilog
from tools.function_synthesis_dsp_proof import prove_primitive_residual
from tools.function_synthesis_exports import MAPPING_FLOW, module_json_flow
from tools.function_synthesis_netlist import Netlist
from tools.function_synthesis_toolchain import SynthesisToolchain

ResidualCase = tuple[Netlist, Netlist, SynthesisToolchain]


@pytest.fixture(scope="module")
def residual_case(tmp_path_factory: pytest.TempPathFactory) -> ResidualCase:
    """Acquire the real post-DSP and physical graphs from the same emitted sine unit."""
    root = tmp_path_factory.mktemp("actual-primitive-residual")
    tools = SynthesisToolchain.inspect(root / "inspection")
    (root / "unit.v").write_text(
        emit_function_unit_verilog("sin", data_width=8, fraction=4, function_lowering="v2")
    )
    flows = {
        "word": (
            "read_verilog -sv unit.v\nhierarchy -check -top function_unit\nproc\nopt_clean\n"
            "synth_xilinx -top function_unit -family xc7 -noiopad -noclkbuf -run begin:coarse\n"
            + module_json_flow("word.json")
        ),
        "physical": MAPPING_FLOW,
    }
    for name, flow in flows.items():
        (root / (name + ".ys")).write_text(flow)
        log = root / (name + ".log")
        tools.run(
            [str(tools.yosys), "-Q", "-T", name + ".ys"],
            root,
            log,
            timeout=240 if name == "physical" else 90,
        )
        tools.check_library_log(log)
    return Netlist.read(root / "word.json"), Netlist.read(root / "mapped.json"), tools


def test_actual_complete_residual_clamps_every_native_deadline(
    residual_case: ResidualCase, tmp_path: Path
) -> None:
    """Preserve all original and primitive bits while bounding a larger caller budget."""
    word, physical, tools = residual_case
    directory = tmp_path / "complete"
    proof = prove_primitive_residual(
        word,
        physical,
        {"argument0": 8},
        8,
        tools,
        directory,
        timeout=190,
    )
    assert proof.primitive_count == 18 and proof.original_input_bits == 8
    additional_inputs = sum(
        len(bits)
        for name, cell in physical.cells.items()
        if cell.kind == "DSP48E1"
        for port, bits in cell.connections.items()
        if cell.directions[port] == "input" and port not in word.cells[name].connections
    )
    additional_outputs = sum(
        len(bits)
        for name, cell in physical.cells.items()
        if cell.kind == "DSP48E1"
        for port, bits in cell.connections.items()
        if cell.directions[port] == "output" and port not in word.cells[name].connections
    )
    assert (additional_inputs, additional_outputs) in [(0, 0), (468, 1308)]
    assert proof.original_output_bits == 8
    assert proof.primitive_input_bits == 4266 + additional_inputs
    assert proof.free_output_values == 1464 + additional_outputs
    record = json.loads((directory / "gia-proof.json").read_text())
    assert len(record["input_order"]) == 1472 + additional_outputs
    assert len(record["output_order"]) == 4274 + additional_inputs
    receipts = list(directory.glob("*.command.json"))
    assert receipts
    for receipt in receipts:
        argv = json.loads(receipt.read_text())
        assert 0 < float(argv[argv.index("--verbose") + 1]) <= 90


@pytest.mark.parametrize("timeout", [True, 0.0, -1.0, float("inf"), float("nan")])
def test_invalid_direct_residual_budget_creates_no_native_artifacts(
    residual_case: ResidualCase, tmp_path: Path, timeout: float
) -> None:
    """Refuse invalid public helper deadlines before creating a proof or running tools."""
    word, physical, tools = residual_case
    directory = tmp_path / "invalid-budget"
    with pytest.raises(ValueError, match="finite positive native deadline"):
        prove_primitive_residual(
            word, physical, {"argument0": 8}, 8, tools, directory, timeout=timeout
        )
    assert not directory.exists()


@pytest.mark.parametrize(
    "damage",
    [
        "no-primitives",
        "renamed-primitive",
        "parameter",
        "output-alias",
        "unknown",
        "undriven",
        "cycle",
        "stateful-memory",
        "observable-unmatched",
        "primitive-cycle",
        "non-generic",
        "multiple-cascade-drivers",
    ],
)
def test_actual_residual_graph_faults_cannot_reach_a_success_record(
    residual_case: ResidualCase, tmp_path: Path, damage: str
) -> None:
    """Reject actual graph damage before native equality can admit an invalid composition."""
    word, physical, tools = residual_case
    a = json.loads(json.dumps({"modules": {"function_unit": word.raw}}))
    b = json.loads(json.dumps({"modules": {"function_unit": physical.raw}}))
    left = a["modules"]["function_unit"]
    right = b["modules"]["function_unit"]
    names = sorted(name for name, cell in right["cells"].items() if cell["type"] == "DSP48E1")
    name = names[0]
    if damage == "no-primitives":
        right["cells"] = {n: c for n, c in right["cells"].items() if c["type"] != "DSP48E1"}
    elif damage == "renamed-primitive":
        right["cells"][name + "_changed"] = right["cells"].pop(name)
    elif damage == "parameter":
        assert int(right["cells"][name]["parameters"]["MREG"], 2) == 0
        right["cells"][name]["parameters"]["MREG"] = "0"
    elif damage == "output-alias":
        right["cells"][name]["connections"]["P"][0] = "0"
    elif damage == "stateful-memory":
        cell = next(c for c in left["cells"].values() if c["type"] == "$memrd_v2")
        cell["parameters"]["CLK_ENABLE"] = "1"
    elif damage == "observable-unmatched":
        unmatched = next(
            n for n, c in left["cells"].items() if c["type"] == "DSP48E1" and n not in names
        )
        bit = left["cells"][unmatched]["connections"]["P"][0]
        left["ports"]["result"]["bits"][0] = bit
        left["netnames"]["result"]["bits"][0] = bit
    elif damage == "primitive-cycle":
        left["cells"][name]["connections"]["A"][0] = left["cells"][name]["connections"]["P"][0]
    elif damage == "non-generic":
        cell = next(c for c in left["cells"].values() if c["type"].startswith("$"))
        cell["type"] = "INV"
    elif damage == "multiple-cascade-drivers":
        candidates = [
            n
            for n in names
            if left["cells"][n]["parameters"]["A_INPUT"] == "DIRECT"
            and left["cells"][n]["connections"]["A"][0] in {"0", "1"}
        ]
        pair = next(
            [a, b]
            for a in candidates
            for b in candidates
            if a != b
            and left["cells"][a]["connections"]["A"][0] == left["cells"][b]["connections"]["A"][0]
        )
        for module in [left, right]:
            for instance in pair:
                cell = module["cells"][instance]
                cell["connections"]["ACOUT"] = list(cell["connections"]["A"])
                cell["connections"]["ACOUT"][0] = 999999
                cell["port_directions"]["ACOUT"] = "output"
    else:
        bit = left["ports"]["result"]["bits"][0]
        cell = next(
            c
            for c in left["cells"].values()
            if any(
                bit in bits
                for port, bits in c["connections"].items()
                if c["port_directions"][port] == "output"
            )
        )
        port = next(p for p, d in cell["port_directions"].items() if d == "input")
        cell["connections"][port][0] = {"unknown": "x", "undriven": 999999, "cycle": bit}[damage]
    path_a, path_b = tmp_path / "word.json", tmp_path / "physical.json"
    path_a.write_text(json.dumps(a))
    path_b.write_text(json.dumps(b))
    directory = tmp_path / "refused"
    expected = {
        "no-primitives": "physical DSP instances",
        "renamed-primitive": "physical DSP instances",
        "parameter": "existing parameter or port",
        "output-alias": "output aliases, constants or primary identities differ",
        "unknown": "unknown actual data bit",
        "undriven": "undriven edge or combinational cycle",
        "cycle": "undriven edge or combinational cycle",
        "stateful-memory": "stateful memory read",
        "observable-unmatched": "unmatched DSP affects",
        "primitive-cycle": "primitive dependency graph is cyclic",
        "non-generic": "non-generic actual cell",
        "multiple-cascade-drivers": "multiple actual bit drivers",
    }
    with pytest.raises(ValueError, match=expected[damage]):
        prove_primitive_residual(
            Netlist.read(path_a),
            Netlist.read(path_b),
            {"argument0": 8},
            8,
            tools,
            directory,
            timeout=90,
        )
    assert not (directory / "gia-proof.json").exists()


def test_actual_primary_cascade_aliases_preserve_the_complete_native_residual(
    residual_case: ResidualCase, tmp_path: Path
) -> None:
    """Keep valid direct B cascade identities instead of treating primary aliases as free values."""
    word, physical, tools = residual_case
    a = json.loads(json.dumps({"modules": {"function_unit": word.raw}}))
    b = json.loads(json.dumps({"modules": {"function_unit": physical.raw}}))
    left, right = a["modules"]["function_unit"], b["modules"]["function_unit"]
    primary = set(left["ports"]["argument0"]["bits"])
    name = next(
        n
        for n, c in right["cells"].items()
        if c["type"] == "DSP48E1"
        and c["parameters"]["B_INPUT"] == "DIRECT"
        and "BCOUT" in c["connections"]
        and set(left["cells"][n]["connections"]["B"]) <= primary
    )
    for module in [left, right]:
        cell = module["cells"][name]
        aliases = dict(zip(cell["connections"]["BCOUT"], cell["connections"]["B"], strict=True))
        for port in module["ports"].values():
            port["bits"] = [aliases.get(bit, bit) for bit in port["bits"]]
        for net in module["netnames"].values():
            net["bits"] = [aliases.get(bit, bit) for bit in net["bits"]]
        for instance in module["cells"].values():
            instance["connections"] = {
                port: [aliases.get(bit, bit) for bit in bits]
                for port, bits in instance["connections"].items()
            }
    path_a, path_b = tmp_path / "word.json", tmp_path / "physical.json"
    path_a.write_text(json.dumps(a))
    path_b.write_text(json.dumps(b))
    directory = tmp_path / "primary-aliases"
    proof = prove_primitive_residual(
        Netlist.read(path_a),
        Netlist.read(path_b),
        {"argument0": 8},
        8,
        tools,
        directory,
        timeout=90,
    )
    assert proof.original_input_bits == proof.original_output_bits == 8
    changed_word, changed_physical = Netlist.read(path_a), Netlist.read(path_b)
    additional_inputs = sum(
        len(bits)
        for name, cell in changed_physical.cells.items()
        if cell.kind == "DSP48E1"
        for port, bits in cell.connections.items()
        if cell.directions[port] == "input" and port not in changed_word.cells[name].connections
    )
    additional_outputs = sum(
        len(bits)
        for name, cell in changed_physical.cells.items()
        if cell.kind == "DSP48E1"
        for port, bits in cell.connections.items()
        if cell.directions[port] == "output" and port not in changed_word.cells[name].connections
    )
    assert (additional_inputs, additional_outputs) in [(0, 0), (468, 1308)]
    assert proof.primitive_input_bits == 4266 + additional_inputs
    assert proof.free_output_values < 1464 + additional_outputs
    record = json.loads((directory / "gia-proof.json").read_text())
    assert len(record["output_order"]) == 4274 + additional_inputs
    assert len(record["input_order"]) == proof.free_output_values + 8
