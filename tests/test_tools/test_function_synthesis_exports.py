# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Independent complete native export contracts

"""Exercise complete native exports while preserving the original physical mapping."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from pathlib import Path

import pytest

from tools.function_synthesis_exports import (
    EXPORT_ARTIFACTS,
    MAPPING_FLOW,
    SYNTHESIS_FLOW,
    export_complete_aigers,
    module_json_flow,
    validate_complete_aiger_exports,
)
from tools.function_synthesis_records import SynthesisCase
from tools.function_synthesis_toolchain import SynthesisToolchain
from tools.function_synthesis_yosys import (
    check_mapped_equivalence,
    read_function_synthesis_measurement,
    run_function_synthesis,
)

ExportCase = tuple[SynthesisCase, Path, SynthesisToolchain]


@pytest.fixture(scope="module")
def export_case(tmp_path_factory: pytest.TempPathFactory) -> ExportCase:
    """Acquire an actual complete zero-DSP measurement through the integrated runner."""
    root = tmp_path_factory.mktemp("actual-independent-export")
    tools = SynthesisToolchain.inspect(root / "inspection")
    case = SynthesisCase("floor", 8, 0)
    measurement = run_function_synthesis(case, root / "measurement", tools)
    assert measurement.proof_method == "complete-aiger"
    assert set(EXPORT_ARTIFACTS) <= dict(measurement.artifact_sha256).keys()
    return case, root / "measurement", tools


def _snapshot(root: Path) -> dict[str, tuple[str, int]]:
    """Read every nested export identity and modification timestamp."""
    snapshot: dict[str, tuple[str, int]] = {}
    for parent, _, names in os.walk(root):
        for name in names:
            path = Path(parent) / name
            snapshot[path.relative_to(root).as_posix()] = (
                hashlib.sha256(path.read_bytes()).hexdigest(),
                path.stat().st_mtime_ns,
            )
    return snapshot


def test_original_physical_mapping_prefix_is_byte_exact() -> None:
    """Keep the original synthesis commands rather than silently changing measured hardware."""
    assert hashlib.sha256(SYNTHESIS_FLOW.encode()).hexdigest() == (
        "ff9a4c8e7edf6994615bc77218d1fa37f2acdfef5791672204b000de7a3bbe67"
    )
    assert hashlib.sha256(MAPPING_FLOW.encode()).hexdigest() == (
        "66f6221d0a28d5e2e1ca853db130558d02a95a389d75f7613675041a4a996483"
    )
    assert SYNTHESIS_FLOW.split("design -stash candidate\n", 1)[0] == MAPPING_FLOW.replace(
        module_json_flow("mapped.json"), "write_json mapped.json\n"
    )


@pytest.mark.parametrize("binary", [False, True])
def test_native_module_export_preserves_the_complete_original_mapping(
    export_case: ExportCase, tmp_path: Path, binary: bool
) -> None:
    """Compare full native graphs and proofs from the old and selected same-case exports."""
    case, selected, tools = export_case
    if binary:
        case = SynthesisCase("heaviside", 8, 4)
        selected = tmp_path / "selected-export"
        run_function_synthesis(case, selected, tools)
    baseline = tmp_path / "original-export"
    baseline.mkdir()
    (baseline / "unit.v").write_bytes((selected / "unit.v").read_bytes())
    (baseline / "flow.ys").write_text(SYNTHESIS_FLOW)
    tools.run(
        [str(tools.yosys), "-Q", "-T", "-s", "flow.ys"],
        baseline,
        baseline / "yosys.log",
        timeout=30,
    )
    check_mapped_equivalence(case, baseline, tools, timeout=30)
    old = read_function_synthesis_measurement(case, baseline, tools)
    new = read_function_synthesis_measurement(case, selected, tools)
    assert old.resources == new.resources
    assert old.to_dict()["proof"] == new.to_dict()["proof"]
    original = json.loads((baseline / "mapped.json").read_bytes())
    exported = json.loads((selected / "mapped.json").read_bytes())
    assert original["modules"]["function_unit"] == exported["modules"]["function_unit"]
    assert len(original["modules"]) > len(exported["modules"])
    assert (selected / "mapped.json").stat().st_size < (baseline / "mapped.json").stat().st_size
    for cell in exported["modules"]["function_unit"]["cells"].values():
        assert exported["modules"][cell["type"]] == original["modules"][cell["type"]]


@pytest.mark.parametrize("filename", ["../out.json", "out.json; clean", "out.txt", "out.json\n"])
def test_module_export_rejects_nonlocal_command_tokens(filename: str) -> None:
    """Refuse unsafe output names before constructing or executing a native command."""
    with pytest.raises(ValueError, match="local JSON filename"):
        module_json_flow(filename)


@pytest.mark.parametrize("module", ["function_unit; clean", "../module", "module\n", ""])
def test_module_export_rejects_nonidentifier_module_names(module: str) -> None:
    """Keep the native design selection to one complete explicitly named module."""
    with pytest.raises(ValueError, match="identifier"):
        module_json_flow("mapped.json", module)


def test_actual_complete_native_exports_retain_full_original_geometry(
    export_case: ExportCase,
) -> None:
    """Retain both real AIG models with all original primary bits and a successful native proof."""
    case, root, tools = export_case
    before = _snapshot(root)
    validate_complete_aiger_exports(case, root, tools)
    for name in ["reference.aig", "mapped.aig"]:
        with (root / name).open("rb") as stream:
            fields = stream.readline().split()
        assert tuple(int(fields[index]) for index in [2, 3, 4]) == (8, 0, 8)
    assert _snapshot(root) == before


@pytest.mark.parametrize("name", ["export.ys", "physical-module.json", "unit.v"])
def test_changed_native_export_inputs_are_preserved_and_refused(
    export_case: ExportCase, tmp_path: Path, name: str
) -> None:
    """Refuse changed flow, physical graph or canonical original function evidence."""
    case, original, tools = export_case
    root = tmp_path / "changed"
    shutil.copytree(original, root)
    changed = root / name
    changed.write_bytes(changed.read_bytes() + b"\n")
    before = _snapshot(root)
    with pytest.raises(ValueError):
        validate_complete_aiger_exports(case, root, tools)
    assert _snapshot(root) == before


@pytest.mark.parametrize("name", ["directory", *EXPORT_ARTIFACTS, "unit.v"])
def test_linked_export_acquisition_is_refused(
    export_case: ExportCase, tmp_path: Path, name: str
) -> None:
    """Require original regular export custody even when a link has genuine target bytes."""
    case, original, tools = export_case
    root = tmp_path / "export"
    shutil.copytree(original, root)
    path = root if name == "directory" else root / name
    target = tmp_path / "retained-target"
    path.rename(target)
    path.symlink_to(target, target_is_directory=target.is_dir())
    with pytest.raises(ValueError, match="symbolic links"):
        validate_complete_aiger_exports(case, root, tools)
    assert path.is_symlink() and target.exists()


@pytest.mark.parametrize("seconds", [True, 0.0, -1.0, float("nan"), float("inf")])
def test_invalid_export_budget_preserves_the_actual_original_acquisition(
    export_case: ExportCase, seconds: float
) -> None:
    """Reject invalid native deadlines before opening any export or source artifact."""
    case, root, tools = export_case
    before = _snapshot(root)
    with pytest.raises(ValueError, match="finite positive"):
        export_complete_aigers(case, root, tools, timeout=seconds)
    assert _snapshot(root) == before


def test_existing_export_acquisition_is_not_overwritten(export_case: ExportCase) -> None:
    """Keep the completed actual native comparison instead of rebuilding over its inputs."""
    case, root, tools = export_case
    before = _snapshot(root)
    with pytest.raises(FileExistsError):
        export_complete_aigers(case, root, tools, timeout=90)
    assert _snapshot(root) == before


def test_actual_physical_export_preserves_both_independent_binary_operands(
    tmp_path: Path,
) -> None:
    """Qualify the complete sixteen-input/eight-output native binary export and immutable reader."""
    tools = SynthesisToolchain.inspect(tmp_path / "inspection")
    case = SynthesisCase("heaviside", 8, 4)
    root = tmp_path / "binary"
    result = run_function_synthesis(case, root, tools)
    assert result.to_dict()["proof"] == {
        "status": "equivalent",
        "method": "complete-aiger",
        "input_bits": 16,
        "output_bits": 8,
        "latches": 0,
    }
    validate_complete_aiger_exports(case, root, tools)
    module = json.loads((root / "physical-module.json").read_text())["modules"]["function_unit"]
    assert set(module["ports"]) == {"argument0", "argument1", "result"}


@pytest.mark.parametrize("existing", [False, True])
def test_real_native_export_input_change_is_refused_after_actual_execution(
    export_case: ExportCase, tmp_path: Path, existing: bool
) -> None:
    """Preserve and refuse an input changed after the actual native export command starts."""
    case, original, tools = export_case
    for name in ["unit.v", "mapped.json"]:
        shutil.copyfile(original / name, tmp_path / name)
    if existing:
        (tmp_path / "export.ys").write_text("Existing owner acquisition\n")
    stop, changed = threading.Event(), threading.Event()

    def change_original() -> None:
        """Append a semantically harmless source comment during the actual acquisition."""
        while not stop.wait(0.001):
            if (tmp_path / "export.command.json").exists():
                with (tmp_path / "unit.v").open("a") as stream:
                    stream.write("\n// actual original input changed during native acquisition\n")
                changed.set()
                return

    worker = threading.Thread(target=change_original)
    worker.start()
    try:
        error = FileExistsError if existing else ValueError
        message = None if existing else "inputs changed during their acquisition"
        with pytest.raises(error, match=message):
            export_complete_aigers(case, tmp_path, tools, timeout=90)
    finally:
        stop.set()
        worker.join(timeout=2)
    assert not worker.is_alive()
    if existing:
        assert not changed.is_set() and not (tmp_path / "export.command.json").exists()
        assert (tmp_path / "export.ys").read_text() == "Existing owner acquisition\n"
        assert (tmp_path / "unit.v").read_bytes() == (original / "unit.v").read_bytes()
    else:
        assert changed.is_set()
        assert (tmp_path / "reference.aig").is_file() and (tmp_path / "mapped.aig").is_file()


def test_actual_export_keeps_direct_blackbox_models_and_native_port_directions(
    tmp_path: Path,
) -> None:
    """Retain the exact LUT model and top graph while excluding unrelated native modules."""
    tools = SynthesisToolchain.inspect(tmp_path / "inspection")
    (tmp_path / "unit.v").write_text(
        "module function_unit(input [5:0] argument0, output result);\n"
        "LUT6 #(.INIT(64'h123456789abcdef0)) mapped_cell(\n"
        ".I0(argument0[0]),.I1(argument0[1]),.I2(argument0[2]),\n"
        ".I3(argument0[3]),.I4(argument0[4]),.I5(argument0[5]),.O(result));\n"
        "endmodule\n"
    )
    (tmp_path / "flow.ys").write_text(
        "read_verilog -lib +/xilinx/cells_sim.v\nread_verilog unit.v\n"
        "hierarchy -check -top function_unit\nwrite_rtlil before.il\n"
        "write_verilog -noattr before.v\n"
        + module_json_flow("selected.json")
        + "write_rtlil after.il\nwrite_verilog -noattr after.v\nblackbox =A:whitebox\n"
        "write_json restored.json\n"
    )
    tools.run(
        [str(tools.yosys), "-Q", "-T", "flow.ys"],
        tmp_path,
        tmp_path / "native.log",
        timeout=10,
    )
    selected = json.loads((tmp_path / "selected.json").read_text())["modules"]
    restored = json.loads((tmp_path / "restored.json").read_text())["modules"]
    assert (tmp_path / "before.v").read_bytes() == (tmp_path / "after.v").read_bytes()
    assert set(selected) == {"function_unit", "LUT6"}
    assert selected["function_unit"] == restored["function_unit"]
    assert selected["LUT6"] == restored["LUT6"]
    directions = {**{f"I{index}": "input" for index in range(6)}, "O": "output"}
    assert selected["function_unit"]["cells"]["mapped_cell"]["port_directions"] == directions
    model = selected["LUT6"]
    assert not model["cells"] and set(model["ports"]) == set(directions)
    assert all(
        port["direction"] == directions[name] and len(port["bits"]) == 1
        for name, port in model["ports"].items()
    )
    tools.verify()
