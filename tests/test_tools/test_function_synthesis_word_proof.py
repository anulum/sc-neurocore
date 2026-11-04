# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual complete word correspondence admission

"""Refuse corrupted correspondences and arithmetic in real native DSP checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sc_neurocore.compiler import emit_function_unit_verilog
from tools.function_synthesis_dsp_ports import configured_dsp
from tools.function_synthesis_exports import module_json_flow
from tools.function_synthesis_netlist import Netlist
from tools.function_synthesis_toolchain import SynthesisToolchain
from tools.function_synthesis_word_proof import prove_word_composition

WordProofCase = tuple[Netlist, Netlist]


@pytest.fixture(scope="module")
def word_proof_case(tmp_path_factory: pytest.TempPathFactory) -> WordProofCase:
    """Acquire both original multiplication and DSP graphs from the same actual unit."""
    root = tmp_path_factory.mktemp("actual-word-proof-admission")
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


def test_actual_word_correspondence_proves_every_complete_product(
    word_proof_case: WordProofCase, tmp_path: Path
) -> None:
    """Preserve all four products and every unchanged function in the native checkpoint."""
    proof = prove_word_composition(*word_proof_case, {"argument0": 8}, 8, tmp_path / "complete")
    assert proof.product_count == 4 and proof.non_multiplier_count == 36
    before = {
        p: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in (tmp_path / "complete").rglob("*")
        if p.is_file()
    }
    assert (
        prove_word_composition(*word_proof_case, {"argument0": 8}, 8).to_dict() == proof.to_dict()
    )
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}
    assert len(list((tmp_path / "complete").glob("product-*.json"))) == 4


def test_original_checkpoint_with_duplicate_primitive_drivers_is_refused(
    word_proof_case: WordProofCase, tmp_path: Path
) -> None:
    """Refuse ambiguous original drivers even when their constant cascade aliases agree."""
    document = json.loads(json.dumps({"modules": {"function_unit": word_proof_case[1].raw}}))
    module = document["modules"]["function_unit"]
    cascades = [
        cell
        for cell in module["cells"].values()
        if cell["type"] == "DSP48E1"
        and cell["parameters"]["A_INPUT"] == "DIRECT"
        and cell["connections"]["A"][0] in {"0", "1"}
    ]
    first, second = next(
        (a, b)
        for a in cascades
        for b in cascades
        if a is not b and a["connections"]["A"][0] == b["connections"]["A"][0]
    )
    fresh = 1 + max(
        bit
        for cell in module["cells"].values()
        for bits in cell["connections"].values()
        for bit in bits
        if isinstance(bit, int)
    )
    for cell in [first, second]:
        cell["connections"]["ACOUT"] = list(cell["connections"]["A"])
        cell["connections"]["ACOUT"][0] = fresh
        cell["port_directions"]["ACOUT"] = "output"
    path = tmp_path / "ambiguous-original.json"
    path.write_text(json.dumps(document))
    original = Netlist.read(path)
    proof = tmp_path / "refused-original"
    with pytest.raises(ValueError, match="original word graph constrains an input or has multiple"):
        prove_word_composition(original, original, {"argument0": 8}, 8, proof)
    assert not proof.exists()


@pytest.mark.parametrize(
    "damage",
    [
        "constant",
        "conflicting-alias",
        "shared-offset",
        "operator",
        "connection-width",
        "memory",
        "unknown",
        "undriven",
        "cycle",
        "stateful-memory",
        "constant-driver",
        "product",
        "missing-product-boundary",
    ],
)
def test_corrupted_actual_word_correspondences_cannot_produce_a_positive_record(
    word_proof_case: WordProofCase, tmp_path: Path, damage: str
) -> None:
    """Keep native function identity, complete boundaries and acyclic known dependencies mandatory."""
    original, mapped = word_proof_case
    a = json.loads(json.dumps({"modules": {"function_unit": original.raw}}))
    b = json.loads(json.dumps({"modules": {"function_unit": mapped.raw}}))
    left, right = a["modules"]["function_unit"], b["modules"]["function_unit"]
    if damage == "constant":
        right["ports"]["result"]["bits"][0] = "0"
        right["netnames"]["result"]["bits"][0] = "0"
    elif damage == "conflicting-alias":
        left["netnames"]["original_input_alias"] = {"bits": [left["ports"]["argument0"]["bits"][0]]}
        right["netnames"]["original_input_alias"] = {"bits": [999999]}
    elif damage == "shared-offset":
        name = next(
            n for n in left["netnames"].keys() & right["netnames"].keys() if n not in left["ports"]
        )
        right["netnames"][name]["offset"] = left["netnames"][name].get("offset", 0) + 1
    elif damage in {"operator", "connection-width", "cycle"}:
        name = next(n for n, c in left["cells"].items() if c["type"] == "$add")
        if damage == "operator":
            right["cells"][name]["type"] = "$sub"
        elif damage == "connection-width":
            right["cells"][name]["connections"]["A"].pop()
        else:
            for module in [left, right]:
                cell = module["cells"][name]
                cell["connections"]["A"][0] = cell["connections"]["Y"][0]
    elif damage == "memory":
        memory = next(iter(right["memories"].values()))
        memory["width"] += 1
    elif damage in {"unknown", "undriven"}:
        for module in [left, right]:
            bit = "x" if damage == "unknown" else 999999
            module["ports"]["result"]["bits"][0] = bit
            module["netnames"]["result"]["bits"][0] = bit
    elif damage == "stateful-memory":
        name = next(n for n, c in left["cells"].items() if c["type"] == "$memrd_v2")
        for module in [left, right]:
            module["cells"][name]["parameters"]["CLK_ENABLE"] = "1"
    elif damage == "constant-driver":
        bit = left["ports"]["result"]["bits"][0]
        name = next(
            n
            for n, c in left["cells"].items()
            if c["type"] != "$mul" and bit in c["connections"].get("Y", [])
        )
        index = left["cells"][name]["connections"]["Y"].index(bit)
        for module in [left, right]:
            module["cells"][name]["connections"]["Y"][index] = "0"
    elif damage == "product":
        cell = next(
            c
            for c in right["cells"].values()
            if c["type"] == "DSP48E1"
            and c["parameters"]["A_INPUT"] == "DIRECT"
            and c["connections"]["A"][0] in {"0", "1"}
        )
        cell["connections"]["A"][0] = "1" if cell["connections"]["A"][0] == "0" else "0"
    else:
        checked = {bit for p in left["ports"].values() for bit in p["bits"]}
        checked.update(
            bit
            for c in left["cells"].values()
            if c["type"] != "$mul"
            for bits in c["connections"].values()
            for bit in bits
        )
        bit = next(
            bit
            for c in left["cells"].values()
            if c["type"] == "$mul"
            for bit in c["connections"]["Y"]
            if isinstance(bit, int) and bit not in checked
        )
        right["netnames"] = {
            n: d
            for n, d in right["netnames"].items()
            if n not in left["netnames"] or bit not in left["netnames"][n]["bits"]
        }
    path_a, path_b = tmp_path / "original.json", tmp_path / "mapped.json"
    path_a.write_text(json.dumps(a))
    path_b.write_text(json.dumps(b))
    directory = tmp_path / "refused"
    with pytest.raises(ValueError):
        prove_word_composition(
            Netlist.read(path_a), Netlist.read(path_b), {"argument0": 8}, 8, directory
        )
    assert not (directory / "word-composition.json").exists()


def test_actual_arccos_full_word_products_include_unshifted_cascade(tmp_path: Path) -> None:
    """Prove every original arccos product with the actual native full-PCIN DSP profile."""
    tools = SynthesisToolchain.inspect(tmp_path / "inspection")
    (tmp_path / "unit.v").write_text(
        emit_function_unit_verilog("arccos", data_width=16, fraction=8, function_lowering="v2")
    )
    for name, stage in [("original", "map_dsp"), ("mapped", "coarse")]:
        (tmp_path / (name + ".ys")).write_text(
            "read_verilog -sv unit.v\nhierarchy -check -top function_unit\nproc\nopt_clean\n"
            "synth_xilinx -top function_unit -family xc7 -noiopad -noclkbuf -run begin:"
            + stage
            + "\n"
            + module_json_flow(name + ".json")
        )
        log = tmp_path / (name + ".log")
        tools.run([str(tools.yosys), "-Q", "-T", name + ".ys"], tmp_path, log, timeout=90)
        tools.check_library_log(log)
    original = Netlist.read(tmp_path / "original.json")
    mapped = Netlist.read(tmp_path / "mapped.json")
    cascades = {
        name
        for name, cell in mapped.cells.items()
        if cell.kind == "DSP48E1" and configured_dsp(cell) == ("0010101", "DIRECT", "DIRECT")
    }
    assert cascades
    proof = prove_word_composition(original, mapped, {"argument0": 16}, 16, tmp_path / "proof")
    assert proof.product_count == 5
    assert cascades <= set(proof.covered_product_cells)
    assert all(
        json.loads(path.read_text())["difference"] == []
        for path in (tmp_path / "proof").glob("product-*.json")
    )
