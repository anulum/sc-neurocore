# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native DSP interface alignment and retained graph readback

"""Check real native DSP interface additions without rewriting original graphs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sc_neurocore.compiler import emit_function_unit_verilog
from tools.function_synthesis_dsp_ports import (
    DSP_AUXILIARY_INPUTS,
    align_dsp_ports,
    configured_dsp,
    read_dsp_port_alignment,
)
from tools.function_synthesis_exports import MAPPING_FLOW, module_json_flow
from tools.function_synthesis_json import MAX_NETLIST_BYTES
from tools.function_synthesis_netlist import Netlist, NetlistCell, object_mapping
from tools.function_synthesis_toolchain import SynthesisToolchain

NativeGraphs = tuple[Netlist, Netlist]


@pytest.fixture(scope="module")
def native_graphs(tmp_path_factory: pytest.TempPathFactory) -> NativeGraphs:
    """Obtain actual coarse and full-port ABC9 physical graphs of the public sine unit."""
    root = tmp_path_factory.mktemp("native-dsp-port-graphs")
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
        "physical": MAPPING_FLOW.replace("synth_xilinx -top", "synth_xilinx -abc9 -top"),
    }
    for name, flow in flows.items():
        (root / (name + ".ys")).write_text(flow)
        log = root / (name + ".log")
        tools.run([str(tools.yosys), "-Q", "-T", name + ".ys"], root, log, timeout=90)
        tools.check_library_log(log)
    word, physical = Netlist.read(root / "word.json"), Netlist.read(root / "mapped.json")
    actual = [cell for cell in physical.cells.values() if cell.kind == "DSP48E1"]
    assert len(actual) == 18
    assert all(set(DSP_AUXILIARY_INPUTS) <= cell.connections.keys() for cell in actual)
    return word, physical


def _read_module(module: dict[str, object], path: Path) -> Netlist:
    """Read an independently retained native module through the public JSON reader."""
    path.write_text(json.dumps({"modules": {"function_unit": module}}))
    return Netlist.read(path)


def _retain_alignment(word: Netlist, physical: Netlist, directory: Path) -> Netlist:
    """Store an actual public derivation and its complete derived module."""
    aligned, record = align_dsp_ports(word, physical)
    directory.mkdir()
    (directory / "native-port-alignment.json").write_text(json.dumps(record))
    (directory / "aligned-word.json").write_text(
        json.dumps({"modules": {"function_unit": aligned.raw}})
    )
    return aligned


def test_native_extension_preserves_every_original_field(native_graphs: NativeGraphs) -> None:
    """Add only known zero controls and fresh unobserved outputs to a separate source clone."""
    word, physical = native_graphs
    original = json.dumps([word.raw, physical.raw], sort_keys=True)
    aligned, record = align_dsp_ports(word, physical)
    assert json.dumps([word.raw, physical.raw], sort_keys=True) == original
    assert aligned.ports == word.ports and aligned.names == word.names
    assert aligned.offsets == word.offsets and aligned.memory_identity == word.memory_identity
    assert aligned.cells.keys() == word.cells.keys()
    assert {k: v for k, v in aligned.raw.items() if k != "cells"} == {
        k: v for k, v in word.raw.items() if k != "cells"
    }
    original_bits = {
        bit for cell in word.cells.values() for bits in cell.connections.values() for bit in bits
    }
    new_outputs: list[int | str] = []
    expanded = 0
    for name, original_cell in word.cells.items():
        cell = aligned.cells[name]
        assert cell.parameters == original_cell.parameters and cell.kind == original_cell.kind
        for port, bits in original_cell.connections.items():
            assert cell.connections[port] == bits
            assert cell.directions[port] == original_cell.directions[port]
        additions = cell.connections.keys() - original_cell.connections.keys()
        if additions:
            expanded += 1
            assert set(DSP_AUXILIARY_INPUTS) <= additions
            for port in additions:
                if cell.directions[port] == "input":
                    assert port in DSP_AUXILIARY_INPUTS and cell.connections[port] == ("0",)
                else:
                    new_outputs.extend(cell.connections[port])
            assert configured_dsp(cell) == configured_dsp(physical.cells[name])
    assert expanded == 18 and len(new_outputs) == 1308
    assert len(set(new_outputs)) == len(new_outputs) and not set(new_outputs) & original_bits
    assert len(object_mapping(record["added_ports"])) == expanded
    for key, graph in [
        ("source_module_sha256", word),
        ("physical_module_sha256", physical),
        ("aligned_source_module_sha256", aligned),
    ]:
        canonical = json.dumps(graph.raw, sort_keys=True, separators=(",", ":")).encode()
        assert record[key] == hashlib.sha256(canonical).hexdigest()


def test_native_minimal_and_complete_profiles_need_no_rewrite(
    native_graphs: NativeGraphs, tmp_path: Path
) -> None:
    """Keep both exact native interface profiles unchanged when no extension is needed."""
    for index, graph in enumerate(native_graphs):
        aligned, record = align_dsp_ports(graph, graph)
        assert aligned == graph and record["added_ports"] == {}
        retained = _retain_alignment(graph, graph, tmp_path / str(index))
        assert read_dsp_port_alignment(graph, graph, tmp_path / str(index)) == retained


def test_native_status_outputs_can_be_unconnected(native_graphs: NativeGraphs) -> None:
    """Keep every real control while omitting unobserved native status outputs."""
    _, physical = native_graphs
    original = next(cell for cell in physical.cells.values() if cell.kind == "DSP48E1")
    raw = next(
        object_mapping(value)
        for value in object_mapping(physical.clone_module()["cells"]).values()
        if object_mapping(value)["type"] == "DSP48E1"
    )
    connections = object_mapping(raw["connections"])
    directions = object_mapping(raw["port_directions"])
    status = {
        port
        for port, direction in directions.items()
        if direction == "output" and port not in {"P", "ACOUT", "BCOUT", "PCOUT"}
    }
    assert len(status) == 7
    for port in status:
        del connections[port], directions[port]
    raw.update(connections=connections, port_directions=directions)
    actual = NetlistCell.read(raw)
    assert configured_dsp(actual) == configured_dsp(original)
    assert actual.parameters == original.parameters
    assert all(
        actual.connections[port] == original.connections[port] for port in actual.connections
    )
    assert set(DSP_AUXILIARY_INPUTS) <= actual.connections.keys()
    assert status <= original.connections.keys()


@pytest.mark.parametrize("port", DSP_AUXILIARY_INPUTS)
def test_native_auxiliary_inputs_must_be_known_zero(native_graphs: NativeGraphs, port: str) -> None:
    """Refuse each actual auxiliary control becoming variable, unknown or active."""
    _, physical = native_graphs
    raw = next(
        value
        for value in object_mapping(physical.clone_module()["cells"]).values()
        if object_mapping(value)["type"] == "DSP48E1"
    )
    cell = object_mapping(raw)
    connections = object_mapping(cell["connections"])
    for value, error in [
        ("1", "auxiliary controls"),
        ("x", "unknown value"),
        (999999, "auxiliary controls"),
    ]:
        cell["connections"] = {**connections, port: [value]}
        with pytest.raises(ValueError, match=error):
            configured_dsp(NetlistCell.read(cell))


@pytest.mark.parametrize(
    "damage,error",
    [
        ("kind", "zero-register"),
        ("register", "zero-register"),
        ("parameter-name", "zero-register"),
        ("arithmetic", "arithmetic configuration"),
        ("missing-control", "port geometry"),
        ("missing-core", "port geometry"),
        ("unknown-port", "port geometry"),
        ("width", "port geometry"),
        ("direction", "different direction"),
        ("control", "connected DSP control"),
        ("mode", "no qualified arithmetic contract"),
    ],
)
def test_native_configuration_damage_is_refused(
    native_graphs: NativeGraphs, damage: str, error: str
) -> None:
    """Validate complete native parameters, geometry, directions and arithmetic controls."""
    _, physical = native_graphs
    cell = next(
        object_mapping(value)
        for value in object_mapping(physical.clone_module()["cells"]).values()
        if object_mapping(value)["type"] == "DSP48E1"
    )
    parameters = object_mapping(cell["parameters"])
    connections = object_mapping(cell["connections"])
    directions = object_mapping(cell["port_directions"])
    if damage == "kind":
        cell["type"] = "DSP48E2"
    elif damage == "register":
        parameters["MREG"] = "1"
    elif damage == "parameter-name":
        parameters["EXTRA_REGISTER"] = "0"
    elif damage == "arithmetic":
        parameters["USE_MULT"] = "NONE"
    elif damage in {"missing-control", "missing-core"}:
        port = "CLK" if damage == "missing-control" else "A"
        del connections[port], directions[port]
    elif damage == "unknown-port":
        connections["UNKNOWN"] = ["0"]
        directions["UNKNOWN"] = "input"
    elif damage == "width":
        connections["CLK"] = ["0", "0"]
    elif damage == "direction":
        directions["CLK"] = "output"
    elif damage == "control":
        connections["CARRYIN"] = ["1"]
    else:
        connections["OPMODE"] = ["1"] * 7
    cell.update(parameters=parameters, connections=connections, port_directions=directions)
    with pytest.raises(ValueError, match=error):
        configured_dsp(NetlistCell.read(cell))


@pytest.mark.parametrize("damage", ["missing", "kind", "parameter", "dropped-port"])
def test_alignment_requires_the_actual_matching_source(
    native_graphs: NativeGraphs, tmp_path: Path, damage: str
) -> None:
    """Refuse renamed or incompatible source primitives before deriving any clone."""
    word, physical = native_graphs
    raw = word.clone_module()
    cells = object_mapping(raw["cells"])
    name = next(name for name, cell in physical.cells.items() if cell.kind == "DSP48E1")
    cell = object_mapping(cells[name])
    if damage == "missing":
        del cells[name]
    elif damage == "kind":
        cell["type"] = "$mul"
        cells[name] = cell
    elif damage == "parameter":
        parameters = object_mapping(cell["parameters"])
        parameters["PREG"] = "0"
        cell["parameters"] = parameters
        cells[name] = cell
    else:
        aligned, _ = align_dsp_ports(word, physical)
        with pytest.raises(ValueError, match="existing parameter or port"):
            align_dsp_ports(aligned, word)
        return
    raw["cells"] = cells
    changed = _read_module(raw, tmp_path / "changed-word.json")
    with pytest.raises(ValueError, match="actual word checkpoint|existing parameter or port"):
        align_dsp_ports(changed, physical)


@pytest.mark.parametrize("damage", ["primary", "input", "constant", "alias", "second-driver"])
def test_added_native_outputs_must_remain_unobserved_and_independent(
    native_graphs: NativeGraphs, tmp_path: Path, damage: str
) -> None:
    """Reject observable, fixed and multiply driven native status outputs."""
    word, physical = native_graphs
    raw = json.loads(json.dumps(physical.raw))
    name = next(name for name, cell in physical.cells.items() if cell.kind == "DSP48E1")
    cell = raw["cells"][name]
    bit = cell["connections"]["CARRYCASCOUT"][0]
    if damage == "primary":
        raw["ports"]["result"]["bits"][0] = bit
        raw["netnames"]["result"]["bits"][0] = bit
    elif damage == "input":
        cell["connections"]["A"][0] = bit
    elif damage == "constant":
        cell["connections"]["CARRYCASCOUT"][0] = "0"
    elif damage == "alias":
        cell["connections"]["OVERFLOW"][0] = bit
    else:
        raw["cells"][name + "_second_driver"] = json.loads(json.dumps(cell))
    changed = _read_module(raw, tmp_path / "changed-physical.json")
    with pytest.raises(ValueError, match="output is observed, fixed or aliased"):
        align_dsp_ports(word, changed)


def test_retained_alignment_recomputes_original_and_derived_identity(
    native_graphs: NativeGraphs, tmp_path: Path
) -> None:
    """Read actual graph derivations and reject a changed original or retained clone."""
    word, physical = native_graphs
    directory = tmp_path / "retained"
    aligned = _retain_alignment(word, physical, directory)
    assert read_dsp_port_alignment(word, physical, directory) == aligned
    raw = word.clone_module()
    raw["attributes"] = {**object_mapping(raw["attributes"]), "changed": "1"}
    changed = _read_module(raw, tmp_path / "changed-original.json")
    with pytest.raises(ValueError, match="differs from its actual original modules"):
        read_dsp_port_alignment(changed, physical, directory)
    clone = directory / "aligned-word.json"
    data = json.loads(clone.read_text())
    data["modules"]["function_unit"]["attributes"]["changed"] = "1"
    clone.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="differs from its actual original modules"):
        read_dsp_port_alignment(word, physical, directory)


@pytest.mark.parametrize("artifact", ["native-port-alignment.json", "aligned-word.json"])
@pytest.mark.parametrize("damage", ["missing", "symlink", "directory", "oversized", "corrupt"])
def test_retained_alignment_requires_bounded_regular_native_artifacts(
    native_graphs: NativeGraphs, tmp_path: Path, artifact: str, damage: str
) -> None:
    """Refuse unavailable, linked, malformed and oversized derivation evidence."""
    word, physical = native_graphs
    directory = tmp_path / "retained"
    _retain_alignment(word, physical, directory)
    path = directory / artifact
    if damage == "missing":
        path.unlink()
    elif damage == "symlink":
        original = path.with_suffix(".original")
        path.rename(original)
        path.symlink_to(original)
    elif damage == "directory":
        path.unlink()
        path.mkdir()
    elif damage == "oversized":
        with path.open("r+b") as stream:
            stream.truncate(MAX_NETLIST_BYTES + 1)
    else:
        path.write_text("{")
    with pytest.raises(ValueError):
        read_dsp_port_alignment(word, physical, directory)


def test_retained_alignment_refuses_directory_links_and_changed_manifest(
    native_graphs: NativeGraphs, tmp_path: Path
) -> None:
    """Bind the manifest schema and prevent redirection through a retained directory link."""
    word, physical = native_graphs
    directory = tmp_path / "retained"
    _retain_alignment(word, physical, directory)
    linked = tmp_path / "linked"
    linked.symlink_to(directory, target_is_directory=True)
    with pytest.raises(ValueError, match="bounded regular artifacts"):
        read_dsp_port_alignment(word, physical, linked)
    path = directory / "native-port-alignment.json"
    record = json.loads(path.read_text())
    record["schema_version"] = "changed"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="differs from its actual original modules"):
        read_dsp_port_alignment(word, physical, directory)
