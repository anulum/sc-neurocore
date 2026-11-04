# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual function mapping and mapped-equivalence tests

"""Exercise real mapped RTL, complete port bindings and corrupted export refusal."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import pytest

from sc_neurocore.compiler import emit_function_unit_verilog
from tools.function_synthesis_aiger import AigerInterface, read_aiger_port_order
from tools.function_synthesis_exports import SYNTHESIS_FLOW
from tools.function_synthesis_records import SynthesisCase
from tools.function_synthesis_toolchain import SynthesisToolchain
from tools.function_synthesis_yosys import (
    FunctionSynthesisMeasurement,
    check_mapped_equivalence,
    read_function_synthesis_measurement,
    run_function_synthesis,
)

WireCase = tuple[SynthesisCase, Path, SynthesisToolchain, FunctionSynthesisMeasurement]


def test_retained_reader_reconstructs_real_measurement(wire_case: WireCase) -> None:
    """Reconstruct the original complete successful artifacts without restamping their proof."""
    case, work, tools, result = wire_case
    assert read_function_synthesis_measurement(case, work, tools) == result


@pytest.mark.parametrize(
    "verdict",
    [
        "Networks are NOT EQUIVALENT.\n",
        "No conclusive proof was produced.\n",
        "Networks are equivalent. Time = 0.00\nERROR: damaged proof\n",
        "Networks are equivalent. Time = 0.00\nNetworks are equivalent. Time = 0.01\n",
    ],
)
def test_retained_reader_refuses_tampered_verdict(
    wire_case: WireCase, tmp_path: Path, verdict: str
) -> None:
    """Refuse negative, absent, conflicting and errored retained proof claims."""
    case, original, tools, result = wire_case
    work = tmp_path / "retained"
    work.mkdir()
    for name, _ in result.artifact_sha256:
        shutil.copyfile(original / name, work / name)
    (work / "equivalence.log").write_text(verdict, encoding="utf-8")
    with pytest.raises(ValueError, match="not proved"):
        read_function_synthesis_measurement(case, work, tools)


@pytest.fixture(scope="module")
def inspected_tools(tmp_path_factory: pytest.TempPathFactory) -> SynthesisToolchain:
    """Inspect the real installed synthesis and equivalence executables."""
    return SynthesisToolchain.inspect(tmp_path_factory.mktemp("mapping-tools") / "receipt")


@pytest.fixture(scope="module")
def wire_case(
    inspected_tools: SynthesisToolchain, tmp_path_factory: pytest.TempPathFactory
) -> WireCase:
    """Generate and prove the real zero-cell function used by fast negative controls."""
    case = SynthesisCase("floor", 8, 0)
    work = tmp_path_factory.mktemp("wire-mapping") / "attempt"
    result = run_function_synthesis(case, work, inspected_tools)
    return case, work, inspected_tools, result


def _export_copy(wire_case: WireCase, destination: Path) -> Path:
    source = wire_case[1]
    work = destination / "exports"
    work.mkdir()
    for name in ("unit.v", "flow.ys", "reference.aig", "mapped.aig"):
        shutil.copyfile(source / name, work / name)
    return work


def test_actual_wire_measurement_binds_all_artifacts(wire_case: WireCase) -> None:
    """Retain actual zero-cell area and every source, netlist, tool and proof digest."""
    case, work, tools, result = wire_case
    row = result.to_dict()
    assert result.case == case
    assert result.resources.total_cells == 0 and result.table_bits == 0
    assert row["latency_cycles"] == 0
    assert row["proof"] == {
        "status": "equivalent",
        "method": "complete-aiger",
        "input_bits": 8,
        "output_bits": 8,
        "latches": 0,
    }
    assert row["toolchain_sha256"] == tools.identity
    assert len(result.artifact_sha256) == 19
    for name, expected in result.artifact_sha256:
        with (work / name).open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == expected


def test_actual_binary_mapping_preserves_both_independent_operands(
    inspected_tools: SynthesisToolchain, tmp_path: Path
) -> None:
    """Prove all sixteen input bits and eight result bits of the binary unit."""
    result = run_function_synthesis(
        SynthesisCase("heaviside", 8, 4), tmp_path / "binary", inspected_tools
    )
    assert result.to_dict()["proof"] == {
        "status": "equivalent",
        "method": "complete-aiger",
        "input_bits": 16,
        "output_bits": 8,
        "latches": 0,
    }


def test_actual_dsp_mapping_uses_the_bound_library_model(
    inspected_tools: SynthesisToolchain, tmp_path: Path
) -> None:
    """Prove a real DSP-bearing polynomial through the production mapping runner."""
    result = run_function_synthesis(
        SynthesisCase("sin", 8, 4), tmp_path / "polynomial", inspected_tools
    )
    assert result.resources.dsp_blocks > 0
    assert result.table_bits > 0
    assert result.to_dict()["resources"] == result.resources.to_dict()


def test_physical_result_fault_cannot_escape_through_complete_bdd_fallback(
    inspected_tools: SynthesisToolchain, tmp_path: Path
) -> None:
    """Reject a real physical output inversion through both independent native proof paths."""
    case = SynthesisCase("sin", 8, 4)
    original = tmp_path / "original"
    measurement = run_function_synthesis(case, original, inspected_tools)
    assert measurement.resources.dsp_blocks == 18
    root = tmp_path / "fault"
    root.mkdir()
    for path in original.iterdir():
        if path.is_file():
            shutil.copyfile(path, root / path.name)
    document = json.loads((root / "mapped.json").read_text())
    module = document["modules"]["function_unit"]
    old = module["ports"]["result"]["bits"][0]
    fresh = 1 + max(
        bit
        for cell in module["cells"].values()
        for bits in cell["connections"].values()
        for bit in bits
        if isinstance(bit, int)
    )
    module["ports"]["result"]["bits"][0] = fresh
    module["netnames"]["result"]["bits"][0] = fresh
    module["cells"]["connected_output_fault"] = {
        "type": "INV",
        "parameters": {},
        "port_directions": {"I": "input", "O": "output"},
        "connections": {"I": [old], "O": [fresh]},
    }
    (root / "mapped.json").write_text(json.dumps({"modules": {"function_unit": module}}))
    (root / "fault-export.ys").write_text(
        "read_verilog -lib -specify +/xilinx/cells_sim.v\n"
        "read_json mapped.json\nhierarchy -check -top function_unit\n"
        "write_verilog -noattr mapped.v\n"
        "tee -o stat.json stat -json -top function_unit\n"
    )
    inspected_tools.run(
        [str(inspected_tools.yosys), "-Q", "-T", "fault-export.ys"],
        root,
        root / "fault-export.log",
        timeout=90,
    )
    with pytest.raises(ValueError, match="all-output difference zero"):
        check_mapped_equivalence(case, root, inspected_tools)
    assert (
        "NOT EQUIVALENT" in (root / "composition-proof/primitive-proof/equivalence.log").read_text()
    )
    assert not (root / "composition-proof/proof-record.json").exists()
    assert not (root / "complete-bdd-proof/proof-record.json").exists()
    header, output, _ = (root / "complete-bdd-proof/collapsed.aig").read_bytes().split(b"\n", 2)
    assert header == b"aig 8 8 0 1 0" and output == b"1"


def test_existing_measurement_directory_is_not_reused(wire_case: WireCase) -> None:
    """Preserve a successful measurement instead of overwriting its source or logs."""
    case, work, tools, result = wire_case
    with pytest.raises(FileExistsError):
        run_function_synthesis(case, work, tools)
    assert (
        hashlib.sha256((work / "unit.v").read_bytes()).hexdigest()
        == dict(result.artifact_sha256)["unit.v"]
    )


def test_unsupported_legacy_case_has_no_zero_cost_measurement(
    inspected_tools: SynthesisToolchain, tmp_path: Path
) -> None:
    """Refuse an unsupported function before creating any mapping attempt."""
    with pytest.raises(ValueError, match="Unsupported function synthesis"):
        run_function_synthesis(
            SynthesisCase("tan", 16, 8, "lut-v1"), tmp_path / "unsupported", inspected_tools
        )
    assert not (tmp_path / "unsupported").exists()


@pytest.mark.parametrize("name", ["unit.v", "flow.ys"])
def test_changed_rtl_or_flow_is_not_a_function_proof(
    wire_case: WireCase, tmp_path: Path, name: str
) -> None:
    """Refuse a source or script differing from the actual production generation."""
    case, _, tools, _ = wire_case
    work = _export_copy(wire_case, tmp_path)
    with (work / name).open("a") as stream:
        stream.write("\n// source changed\n")
    with pytest.raises(ValueError, match="differs"):
        check_mapped_equivalence(case, work, tools)


@pytest.mark.parametrize(
    "header",
    [
        b"aag 8 8 0 8 0\n",
        b"aig 8 8 0 8\n",
        b"aig -8 8 0 8 0\n",
        b"aig 8 8 0 8 0",
        b"aig 8 7 0 8 0\n",
        b"aig 8 8 1 8 0\n",
        b"aig 8 8 0 7 0\n",
        b"aig 9 8 0 8 0\n",
    ],
)
def test_actual_export_with_corrupt_geometry_is_refused(
    wire_case: WireCase, tmp_path: Path, header: bytes
) -> None:
    """Keep the complete primary inputs and outputs without admitting state."""
    case, _, tools, _ = wire_case
    work = _export_copy(wire_case, tmp_path)
    path = work / "reference.aig"
    _, body = path.read_bytes().split(b"\n", 1)
    path.write_bytes(header + body)
    with pytest.raises(ValueError, match="AIGER"):
        check_mapped_equivalence(case, work, tools)


@pytest.mark.parametrize("symbol", [b"argument0[0]", b"result[0]"])
def test_actual_export_with_different_port_names_is_refused(
    wire_case: WireCase, tmp_path: Path, symbol: bytes
) -> None:
    """Reject substituted primary-port identities even when the bit counts match."""
    case, _, tools, _ = wire_case
    work = _export_copy(wire_case, tmp_path)
    path = work / "mapped.aig"
    assert symbol in path.read_bytes()
    path.write_bytes(path.read_bytes().replace(symbol, b"other_port[0]", 1))
    with pytest.raises(ValueError, match="named function interface"):
        check_mapped_equivalence(case, work, tools)


def test_real_nonequivalence_is_refused_even_when_abc_exits_zero(
    wire_case: WireCase, tmp_path: Path
) -> None:
    """Invert one actual output literal while retaining all full-domain port names."""
    case, _, tools, _ = wire_case
    work = _export_copy(wire_case, tmp_path)
    path = work / "mapped.aig"
    header, first, body = path.read_bytes().split(b"\n", 2)
    path.write_bytes(header + b"\n" + str(int(first) ^ 1).encode() + b"\n" + body)
    with pytest.raises(ValueError, match="not proved"):
        check_mapped_equivalence(case, work, tools)
    assert "Networks are NOT EQUIVALENT" in (work / "equivalence.log").read_text()


def test_export_changed_during_real_native_checks_is_not_accepted(
    wire_case: WireCase, tmp_path: Path
) -> None:
    """Detect an actual export change at the native interface-inspection boundary."""
    case, _, tools, _ = wire_case
    work = _export_copy(wire_case, tmp_path)
    changed = False

    def change_export(event: str, args: tuple[object, ...]) -> None:
        """Append to the real export after its digest and native command are retained."""
        nonlocal changed
        if (
            event == "subprocess.Popen"
            and args[2] in (work, str(work), str(work).encode())
            and (work / "reference-io.command.json").is_file()
            and not changed
        ):
            changed = True
            with (work / "mapped.aig").open("ab") as stream:
                stream.write(b"\nactual export changed at its native subprocess boundary\n")

    change_export.__dict__["__cantrace__"] = True
    sys.addaudithook(change_export)
    with pytest.raises(ValueError, match="changed during"):
        check_mapped_equivalence(case, work, tools)
    assert changed
    assert "Networks are equivalent" in (work / "equivalence.log").read_text()


@pytest.mark.parametrize("timeout", [True, 0.0, -1.0, float("inf"), float("nan")])
def test_invalid_native_deadline_preserves_the_actual_measurement(
    wire_case: WireCase, timeout: float
) -> None:
    """Validate the caller deadline before any native command or artifact mutation."""
    case, root, tools, _ = wire_case
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.iterdir() if p.is_file()}
    with pytest.raises(ValueError, match="finite positive native deadline"):
        check_mapped_equivalence(case, root, tools, timeout=timeout)
    assert before == {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.iterdir() if p.is_file()
    }


def test_retained_reader_refuses_an_actual_linked_native_artifact(
    wire_case: WireCase, tmp_path: Path
) -> None:
    """Refuse a linked physical netlist while preserving the original native acquisition."""
    case, original, tools, _ = wire_case
    root = tmp_path / "linked-native-acquisition"
    shutil.copytree(original, root)
    artifact = root / "mapped.json"
    artifact.unlink()
    artifact.symlink_to(original / "mapped.json")
    before = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in original.iterdir()
        if path.is_file()
    }
    with pytest.raises(ValueError, match="must not be symbolic links"):
        read_function_synthesis_measurement(case, root, tools)
    assert artifact.is_symlink()
    assert before == {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in original.iterdir()
        if path.is_file()
    }


def test_reported_cell_count_must_match_the_actual_native_module(
    wire_case: WireCase, tmp_path: Path
) -> None:
    """Reject internally consistent native statistics that misdescribe the actual mapped graph."""
    case, original, tools, _ = wire_case
    root = tmp_path / "changed-resource-count"
    shutil.copytree(original, root)
    path = root / "stat.json"
    value = json.loads(path.read_text())
    module = value["modules"]["\\function_unit"]
    assert module["num_cells"] == 0 and module["num_cells_by_type"] == {}
    module["num_cells"] = 1
    module["num_cells_by_type"] = {"INV": 1}
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="physical cells differ"):
        read_function_synthesis_measurement(case, root, tools)


def test_earlier_complete_native_flow_remains_readable_without_reacquisition(
    inspected_tools: SynthesisToolchain, tmp_path: Path
) -> None:
    """Preserve genuine complete-AIG acquisitions made by the earlier combined native flow."""
    case = SynthesisCase("floor", 8, 0)
    root = tmp_path / "combined-native-acquisition"
    root.mkdir()
    (root / "unit.v").write_text(emit_function_unit_verilog("floor", data_width=8, fraction=0))
    (root / "flow.ys").write_text(SYNTHESIS_FLOW)
    inspected_tools.run(
        [str(inspected_tools.yosys), "-Q", "-T", "-s", "flow.ys"],
        root,
        root / "yosys.log",
        timeout=90,
    )
    inspected_tools.check_library_log(root / "yosys.log")
    check_mapped_equivalence(case, root, inspected_tools)
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.iterdir() if p.is_file()}
    result = read_function_synthesis_measurement(case, root, inspected_tools)
    assert result.resources.total_cells == 0 and result.proof_method == "complete-aiger"
    assert len(result.artifact_sha256) == 15
    assert "export.ys" not in dict(result.artifact_sha256)
    assert before == {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.iterdir() if p.is_file()
    }


def test_native_mapping_normalization_preserves_physical_measurement(
    inspected_tools: SynthesisToolchain, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Normalize a real binary mapping eagerly while preserving its physical counts and full proof."""
    case = SynthesisCase("heaviside", 16, 8)
    baseline_root = tmp_path / "baseline"
    baseline = run_function_synthesis(case, baseline_root, inspected_tools)
    original_bytes = (baseline_root / "mapped.json").stat().st_size
    monkeypatch.setattr("tools.function_synthesis_yosys.MAX_NETLIST_BYTES", original_bytes - 1)
    normalized_root = tmp_path / "normalized"
    normalized = run_function_synthesis(case, normalized_root, inspected_tools)
    raw = (normalized_root / "mapped-native.json").read_bytes()
    compact = (normalized_root / "mapped.json").read_bytes()
    record = json.loads((normalized_root / "mapped-json-normalization.json").read_text())
    assert (
        json.loads(raw)
        == json.loads(compact)
        == json.loads((baseline_root / "mapped.json").read_bytes())
    )
    assert len(compact) < len(raw) == original_bytes
    assert record["parser_budget_bytes"] == 64 * 1024 * 1024
    assert normalized.resources == baseline.resources
    assert (
        normalized.to_dict()["proof"]
        == baseline.to_dict()["proof"]
        == {
            "status": "equivalent",
            "method": "complete-aiger",
            "input_bits": 32,
            "output_bits": 16,
            "latches": 0,
        }
    )
    assert read_function_synthesis_measurement(case, normalized_root, inspected_tools) == normalized
    assert {"mapped-native.json", "mapped-json-normalization.json"} <= dict(
        normalized.artifact_sha256
    ).keys()


@pytest.fixture(scope="module")
def reordered_case(
    inspected_tools: SynthesisToolchain, tmp_path_factory: pytest.TempPathFactory
) -> WireCase:
    """Permute actual binary export inputs natively while preserving all names and output order."""
    case = SynthesisCase("heaviside", 16, 8)
    parent = tmp_path_factory.mktemp("reordered-binary-mapping")
    original, root = parent / "original", parent / "measurement"
    run_function_synthesis(case, original, inspected_tools)
    root.mkdir()
    excluded = {
        "mapped.aig",
        "reference-io.log",
        "reference-io.command.json",
        "mapped-io.log",
        "mapped-io.command.json",
        "equivalence.log",
        "equivalence.command.json",
    }
    for path in original.iterdir():
        if path.is_file() and path.name not in excluded:
            shutil.copyfile(path, root / path.name)
    shutil.copyfile(original / "mapped.aig", root / "mapped-before-permutation.aig")
    inspected_tools.run(
        [
            str(inspected_tools.abc),
            "-c",
            "read_aiger mapped-before-permutation.aig; permute -S 0 -o -f; "
            "write_aiger -s mapped.aig",
        ],
        root,
        root / "input-permutation.log",
        timeout=30,
    )
    check_mapped_equivalence(case, root, inspected_tools)
    measurement = read_function_synthesis_measurement(case, root, inspected_tools)
    return case, root, inspected_tools, measurement


def test_reordered_binary_inputs_are_proved_by_native_names(reordered_case: WireCase) -> None:
    """Prove all original independent named bits despite a different physical input order."""
    case, root, tools, measurement = reordered_case
    names = tuple(f"argument{arg}[{bit}]" for arg in range(2) for bit in range(16))
    outputs = tuple(f"result[{bit}]" for bit in range(16))
    interface = AigerInterface(frozenset(names), frozenset(outputs))
    reference, reference_outputs = read_aiger_port_order(root / "reference-io.log", interface)
    mapped, mapped_outputs = read_aiger_port_order(root / "mapped-io.log", interface)
    assert reference == names and mapped != reference and set(mapped) == set(names)
    assert reference_outputs == mapped_outputs == outputs
    assert "[default = by name]" in (root / "mapped-io.log").read_text()
    assert measurement.to_dict()["proof"] == {
        "status": "equivalent",
        "method": "complete-aiger",
        "input_bits": 32,
        "output_bits": 16,
        "latches": 0,
    }
    assert read_function_synthesis_measurement(case, root, tools) == measurement


@pytest.mark.parametrize(
    "damage", ["matching-mode", "renamed-input", "duplicate-input", "output-order"]
)
def test_reordered_binary_interface_damage_is_refused(
    reordered_case: WireCase, tmp_path: Path, damage: str
) -> None:
    """Refuse missing native name matching, changed independent bits and reordered original outputs."""
    case, original, tools, _ = reordered_case
    root = tmp_path / "damaged-reordered-interface"
    shutil.copytree(original, root)
    path = root / "mapped-io.log"
    original_text = path.read_text()
    text = original_text
    interface = AigerInterface(
        frozenset(f"argument{arg}[{bit}]" for arg in range(2) for bit in range(16)),
        frozenset(f"result[{bit}]" for bit in range(16)),
    )
    mapped, _ = read_aiger_port_order(path, interface)
    first = f"0={mapped[0]}"
    assert first in text and "[default = by name]" in text
    if damage == "matching-mode":
        text = text.replace("[default = by name]", "[default = by order]")
    elif damage == "renamed-input":
        text = text.replace(first, "0=unqualified_argument[0]")
    elif damage == "duplicate-input":
        text = text.replace(first, f"0={mapped[1]}")
    else:
        text = text.replace("0=result[0]", "0=result[1]").replace("1=result[1]", "1=result[0]")
    assert text != original_text
    path.write_text(text)
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.iterdir() if p.is_file()}
    with pytest.raises(ValueError, match="complete named function interface"):
        read_function_synthesis_measurement(case, root, tools)
    assert before == {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.iterdir() if p.is_file()
    }


@pytest.mark.parametrize("budget", [True, 0.0, -1.0, 3301.0, float("nan"), float("inf")])
def test_invalid_shared_mapping_budget_preserves_actual_completed_acquisition(
    wire_case: WireCase, budget: float
) -> None:
    """Reject an invalid total proof budget before touching genuine successful artifacts."""
    case, root, tools, _ = wire_case
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.iterdir() if p.is_file()}
    with pytest.raises(ValueError, match="bounded cohort"):
        check_mapped_equivalence(case, root, tools, cohort_seconds=budget)
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}


def test_exhausted_shared_mapping_budget_cannot_start_a_fresh_native_proof(
    wire_case: WireCase, tmp_path: Path
) -> None:
    """Refuse a new native proof when the actual outer caller already exhausted its budget."""
    from dataclasses import replace

    case, _, original, _ = wire_case
    root = _export_copy(wire_case, tmp_path)
    tools = replace(original, cohort_deadline=0.0)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.iterdir()}
    with pytest.raises(ValueError, match="shared proof cohort budget"):
        check_mapped_equivalence(case, root, tools, timeout=1.0)
    assert set(root.iterdir()) == set(before)
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}


@pytest.mark.parametrize("deadline", [float("nan"), float("inf")])
def test_nonfinite_inherited_mapping_deadline_preserves_real_exports(
    wire_case: WireCase, tmp_path: Path, deadline: float
) -> None:
    """Refuse invalid inherited time bounds before touching the original native exports."""
    from dataclasses import replace

    case, _, original, _ = wire_case
    root = _export_copy(wire_case, tmp_path)
    tools = replace(original, cohort_deadline=deadline)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.iterdir()}
    with pytest.raises(ValueError, match="finite inherited cohort deadline"):
        check_mapped_equivalence(case, root, tools)
    assert set(root.iterdir()) == set(before)
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}


def test_final_source_read_is_charged_after_real_complete_aiger_equivalence(
    wire_case: WireCase, tmp_path: Path
) -> None:
    """Refuse overall success when final source validation consumes the real time allowance."""
    from dataclasses import replace

    case, _, original, _ = wire_case
    positive = tmp_path / "positive"
    positive.mkdir()
    control = _export_copy(wire_case, positive)
    started = time.monotonic()
    check_mapped_equivalence(case, control, original)
    elapsed = time.monotonic() - started
    timeout = min(90.0, max(5.0, elapsed * 2))
    allowance = min(3300.0, max(timeout + 5.0, elapsed * 4 + timeout))
    negative = tmp_path / "negative"
    negative.mkdir()
    root = _export_copy(wire_case, negative)
    deadline = time.monotonic() + allowance
    tools = replace(original, cohort_deadline=deadline)
    delayed = False

    def delay_final_source_read(event: str, args: tuple[object, ...]) -> None:
        """Spend actual remaining wall time after the real equivalence result is retained."""
        nonlocal delayed
        if (
            event == "open"
            and args[0] in (root / "unit.v", str(root / "unit.v"))
            and (root / "equivalence.log").is_file()
            and not delayed
        ):
            delayed = True
            time.sleep(max(0.0, deadline - time.monotonic()) + 0.01)

    delay_final_source_read.__dict__["__cantrace__"] = True
    sys.addaudithook(delay_final_source_read)
    with pytest.raises(
        ValueError, match="Mapped proof validation exhausted the shared cohort budget"
    ):
        check_mapped_equivalence(case, root, tools, timeout=timeout, cohort_seconds=allowance)
    assert delayed
    assert "Networks are equivalent" in (root / "equivalence.log").read_text()
    assert (root / "unit.v").read_bytes() == (control / "unit.v").read_bytes()
