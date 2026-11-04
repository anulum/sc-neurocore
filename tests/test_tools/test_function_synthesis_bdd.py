# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Complete original-function BDD proof contracts

"""Exercise complete native equality, actual graph faults and retained proof refusal."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

from tools.function_synthesis_bdd import prove_complete_bdd_mapping, read_complete_bdd_mapping
from tools.function_synthesis_records import SynthesisCase
from tools.function_synthesis_toolchain import SynthesisToolchain
from tools.function_synthesis_yosys import (
    read_function_synthesis_measurement,
    run_function_synthesis,
)

NativeProof = tuple[SynthesisCase, Path, SynthesisToolchain]


@pytest.fixture(scope="module")
def native_proof(tmp_path_factory: pytest.TempPathFactory) -> NativeProof:
    """Prove an actual mapped function through the public complete native BDD API."""
    base = tmp_path_factory.mktemp("complete-original-bdd")
    tools = SynthesisToolchain.inspect(base / "inspection")
    case = SynthesisCase("floor", 8, 0)
    root = base / "measurement"
    run_function_synthesis(case, root, tools)
    prove_complete_bdd_mapping(case, root, tools)
    return case, root, tools


def test_complete_native_bdd_readback_preserves_original_acquisition(
    native_proof: NativeProof,
) -> None:
    """Keep all original bytes and timestamps while reading a decisive full native proof."""
    case, root, tools = native_proof
    files = [p for p in root.rglob("*") if p.is_file()]
    before = {p: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns) for p in files}
    proof = read_complete_bdd_mapping(case, root, tools)
    assert before == {
        p: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns) for p in files
    }
    assert "complete-bdd-proof/proof-record.json" in dict(proof.artifact_sha256)
    record = json.loads((root / "complete-bdd-proof/proof-record.json").read_text())
    assert record["method"] == "complete-bdd"
    assert record["input_bits"] == record["output_bits"] == 8
    assert record["auxiliary_cut_inputs"] == 0
    assert record["input_order"] == [f"argument0[{i}]" for i in range(8)]
    assert record["output_order"] == [f"result[{i}]" for i in range(8)]


@pytest.mark.parametrize("role", ["reference", "candidate"])
def test_normalization_cannot_drop_its_actual_positional_matching_flag(
    native_proof: NativeProof, tmp_path: Path, role: str
) -> None:
    """Refuse a changed real roundtrip command even when its original positive verdict remains."""
    case, original, tools = native_proof
    root = tmp_path / "changed"
    shutil.copytree(original, root)
    command = root / "complete-bdd-proof" / (role + "-roundtrip.command.json")
    argv = json.loads(command.read_text())
    assert " -n " in argv[-1]
    argv[-1] = argv[-1].replace(" -n ", " ")
    command.write_text(json.dumps(argv))
    before = command.read_bytes(), command.stat().st_mtime_ns
    with pytest.raises(ValueError, match="command changed"):
        read_complete_bdd_mapping(case, root, tools)
    assert (command.read_bytes(), command.stat().st_mtime_ns) == before


def test_full_native_bdd_refuses_a_real_original_result_fault(
    native_proof: NativeProof,
    tmp_path: Path,
) -> None:
    """Reject a changed final result in the actual physical graph through the whole native path."""
    case, root, tools = native_proof
    changed = tmp_path / "changed"
    shutil.copytree(root, changed, ignore=shutil.ignore_patterns("complete-bdd-proof"))
    path = changed / "mapped.json"
    payload = json.loads(path.read_text())
    module = payload["modules"]["function_unit"]
    module["ports"]["result"]["bits"][-1] = "1"
    module["netnames"]["result"]["bits"][-1] = "1"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    with pytest.raises(ValueError, match="all-output difference zero"):
        prove_complete_bdd_mapping(case, changed, tools)
    assert not (changed / "complete-bdd-proof/proof-record.json").exists()
    assert (changed / "complete-bdd-proof/complete-bdd.log").is_file()


def test_complete_native_binary_proof_keeps_both_original_arguments(
    native_proof: NativeProof,
    tmp_path: Path,
) -> None:
    """Compare every bit of two independent operands in a real supported binary function."""
    _, _, tools = native_proof
    case = SynthesisCase("heaviside", 8, 4)
    root = tmp_path / "binary"
    run_function_synthesis(case, root, tools)
    prove_complete_bdd_mapping(case, root, tools)
    read_complete_bdd_mapping(case, root, tools)
    record = json.loads((root / "complete-bdd-proof/proof-record.json").read_text())
    assert record["input_bits"] == 16
    assert set(record["input_order"]) == {f"argument{a}[{b}]" for a in range(2) for b in range(8)}
    assert record["output_bits"] == 8


@pytest.mark.parametrize("budget", [False, 0.0, -1.0, 91.0, float("nan"), float("inf")])
def test_complete_bdd_native_deadline_stays_bounded(
    native_proof: NativeProof,
    budget: float,
) -> None:
    """Refuse invalid caller budgets before altering the existing real acquisition."""
    case, root, tools = native_proof
    record = root / "complete-bdd-proof/proof-record.json"
    before = record.read_bytes()
    with pytest.raises(ValueError):
        prove_complete_bdd_mapping(case, root, tools, timeout=budget)
    assert record.read_bytes() == before


@pytest.mark.parametrize(
    "entry",
    [
        "proof-record.json",
        "checkpoint-exports.json",
        "unit.v",
        "physical-module.json",
        "reference.aig",
        "mapped.aig",
        "reference-plain.aig",
        "candidate-plain.aig",
        "all-differences.aig",
        "singleton.aig",
        "collapsed.aig",
        "reference-roundtrip.log",
        "candidate-io.log",
        "all-output-miter.command.json",
        "complete-bdd.command.json",
    ],
)
def test_actual_native_bdd_artifact_changes_are_refused(
    native_proof: NativeProof,
    tmp_path: Path,
    entry: str,
) -> None:
    """Bind readback to every real source/export/normalization/miter and native proof artifact."""
    case, root, tools = native_proof
    changed = tmp_path / "changed"
    shutil.copytree(root, changed)
    path = changed / "complete-bdd-proof" / entry
    path.write_bytes(path.read_bytes() + b"\nchanged")
    with pytest.raises((OSError, ValueError)):
        read_complete_bdd_mapping(case, changed, tools)


@pytest.mark.parametrize("mode", ["missing", "duplicate", "oversized", "zero", "error"])
def test_a_constant_graph_without_a_completed_native_bdd_protocol_is_refused(
    native_proof: NativeProof,
    tmp_path: Path,
    mode: str,
) -> None:
    """Require one bounded native completion marker in addition to the actual constant graph."""
    case, root, tools = native_proof
    changed = tmp_path / "changed"
    shutil.copytree(root, changed)
    path = changed / "complete-bdd-proof/complete-bdd.log"
    text = path.read_text()
    marker = next(line for line in text.splitlines() if line.startswith("Shared BDD size ="))
    if mode == "missing":
        text = text.replace(marker, "Global BDD computation did not complete")
    elif mode == "duplicate":
        text += "\n" + marker + "\n"
    elif mode == "error":
        text += "\nERROR: native construction failed\n"
    else:
        value = "1000001" if mode == "oversized" else "0"
        replacement = "Shared BDD size = " + value + " nodes.  BDD construction time = 0.01 sec"
        text = text.replace(marker, replacement)
    path.write_text(text)
    with pytest.raises(ValueError, match="did not complete"):
        read_complete_bdd_mapping(case, changed, tools)


@pytest.mark.parametrize("field", ["recorded_at", "normalization_seconds", "case", "input_bits"])
def test_native_bdd_record_metadata_cannot_be_reinterpreted(
    native_proof: NativeProof,
    tmp_path: Path,
    field: str,
) -> None:
    """Refuse changed original time, budget, case identity and complete-domain metadata."""
    case, root, tools = native_proof
    changed = tmp_path / "changed"
    shutil.copytree(root, changed)
    path = changed / "complete-bdd-proof/proof-record.json"
    record = json.loads(path.read_text())
    record[field] = None
    path.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    with pytest.raises(ValueError):
        read_complete_bdd_mapping(case, changed, tools)


def test_native_bdd_readback_rejects_a_nonzero_constant_graph(
    native_proof: NativeProof,
    tmp_path: Path,
) -> None:
    """Refuse an actual native collapsed graph whose all-output difference was changed to one."""
    case, root, tools = native_proof
    changed = tmp_path / "changed"
    shutil.copytree(root, changed)
    path = changed / "complete-bdd-proof/collapsed.aig"
    data = path.read_bytes().split(b"\n", 2)
    assert data[1] == b"0"
    data[1] = b"1"
    path.write_bytes(b"\n".join(data))
    with pytest.raises(ValueError, match="all-output difference zero"):
        read_complete_bdd_mapping(case, changed, tools)


def test_complete_native_bdd_proof_is_not_overwritten(native_proof: NativeProof) -> None:
    """Preserve the original successful proof when a caller requests acquisition twice."""
    case, root, tools = native_proof
    path = root / "complete-bdd-proof/proof-record.json"
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        prove_complete_bdd_mapping(case, root, tools)
    assert path.read_bytes() == before


def test_public_measurement_reports_the_actual_complete_bdd_method(
    native_proof: NativeProof,
) -> None:
    """Expose the successful whole-function proof without changing original physical area."""
    case, root, tools = native_proof
    measurement = read_function_synthesis_measurement(case, root, tools)
    assert measurement.proof_method == "complete-bdd"
    assert measurement.resources.dsp_blocks == 0
    assert measurement.to_dict()["proof"] == {
        "status": "equivalent",
        "method": "complete-bdd",
        "input_bits": 8,
        "output_bits": 8,
        "latches": 0,
    }


@pytest.mark.parametrize(
    "mode", ["object", "nonstring", "length", "executable", "deadline", "nonfinite"]
)
def test_native_bdd_command_receipts_cannot_change_execution_or_budget(
    native_proof: NativeProof, tmp_path: Path, mode: str
) -> None:
    """Refuse altered real native argument vectors and expanded execution deadlines."""
    case, root, tools = native_proof
    changed = tmp_path / "changed"
    shutil.copytree(root, changed)
    path = changed / "complete-bdd-proof/complete-bdd.command.json"
    argv = json.loads(path.read_text())
    if mode == "object":
        argv = {"command": argv}
    elif mode == "nonstring":
        argv[0] = None
    elif mode == "length":
        argv.pop()
    elif mode == "executable":
        argv[7] = "/uninspected/abc"
    else:
        argv[6] = "91" if mode == "deadline" else "nan"
    path.write_text(json.dumps(argv) + "\n")
    with pytest.raises(ValueError, match="command"):
        read_complete_bdd_mapping(case, changed, tools)


@pytest.mark.parametrize(
    "entry",
    [
        "root",
        "complete-bdd-proof",
        "complete-bdd-proof/proof-record.json",
        "mapped.v",
        "complete-bdd-proof/complete-bdd.log",
    ],
)
def test_complete_bdd_reader_refuses_linked_originals_and_native_evidence(
    native_proof: NativeProof, tmp_path: Path, entry: str
) -> None:
    """Keep original graph custody when a retained file or directory becomes a symbolic link."""
    case, root, tools = native_proof
    changed = tmp_path / "changed"
    shutil.copytree(root, changed)
    path = changed if entry == "root" else changed / entry
    retained = tmp_path / "retained"
    path.rename(retained)
    path.symlink_to(retained, target_is_directory=retained.is_dir())
    with pytest.raises(ValueError):
        read_complete_bdd_mapping(case, changed, tools)


@pytest.mark.parametrize(
    "entry", ["complete-bdd.log", "complete-bdd.command.json", "proof-record.json"]
)
def test_complete_bdd_reader_requires_each_actual_native_protocol_artifact(
    native_proof: NativeProof, tmp_path: Path, entry: str
) -> None:
    """Reject incomplete retained native computations without reconstructing their evidence."""
    case, root, tools = native_proof
    changed = tmp_path / "changed"
    shutil.copytree(root, changed)
    (changed / "complete-bdd-proof" / entry).unlink()
    with pytest.raises(ValueError):
        read_complete_bdd_mapping(case, changed, tools)


@pytest.mark.parametrize(
    "stamp,budget",
    [
        ("2026-01-01T00:00:00+01:00", 60),
        ("2026-01-01T00:00:00Z", 60),
        ("2026-01-01T00:00:00+00:00", 0),
        ("2026-01-01T00:00:00+00:00", 61),
    ],
)
def test_complete_bdd_reader_requires_canonical_acquisition_time_and_native_budget(
    native_proof: NativeProof, tmp_path: Path, stamp: str, budget: int
) -> None:
    """Refuse reinterpretation of the original time zone, encoding or normalization limit."""
    case, root, tools = native_proof
    changed = tmp_path / "changed"
    shutil.copytree(root, changed)
    path = changed / "complete-bdd-proof/proof-record.json"
    record = json.loads(path.read_text())
    record.update(recorded_at=stamp, normalization_seconds=budget)
    path.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    with pytest.raises(ValueError, match="acquisition time or native budget"):
        read_complete_bdd_mapping(case, changed, tools)


@pytest.mark.parametrize("kind", ["policy", "order"])
def test_complete_bdd_readback_requires_the_actual_native_named_bit_order(
    native_proof: NativeProof, tmp_path: Path, kind: str
) -> None:
    """Refuse unknown matching policies and permutations before positional native comparison."""
    case, root, tools = native_proof
    changed = tmp_path / "changed"
    shutil.copytree(root, changed)
    path = (
        changed
        / "complete-bdd-proof"
        / ("cec-help.log" if kind == "policy" else "candidate-io.log")
    )
    text = path.read_text()
    if kind == "policy":
        text = text.replace("[default = by name]", "[default = by order]")
    else:
        text = (
            text.replace("argument0[0]", "temporary")
            .replace("argument0[1]", "argument0[0]")
            .replace("temporary", "argument0[1]")
        )
    path.write_text(text)
    with pytest.raises(ValueError, match="policy|named bit order"):
        read_complete_bdd_mapping(case, changed, tools)


@pytest.mark.parametrize("kind", ["policy", "order", "source"])
def test_native_bdd_acquisition_refuses_actual_protocol_or_original_changes(
    native_proof: NativeProof, tmp_path: Path, kind: str
) -> None:
    """Alter actual bytes at native I/O boundaries and require the public acquisition to refuse them."""
    case, root, tools = native_proof
    changed = tmp_path / "changed"
    shutil.copytree(root, changed, ignore=shutil.ignore_patterns("complete-bdd-proof"))
    directory = changed / "complete-bdd-proof"
    entry = {
        "policy": "cec-help.log",
        "order": "candidate-io.log",
        "source": "complete-bdd.command.json",
    }[kind]
    target = directory / entry
    altered = False

    def change_native_artifact(event: str, args: tuple[object, ...]) -> None:
        """Modify retained native data immediately before its actual production read."""
        nonlocal altered
        if (
            event == "open"
            and args[0] == str(target)
            and args[1] == ("x" if kind == "source" else "r")
            and not altered
        ):
            altered = True
            if kind == "source":
                path = changed / "mapped.v"
                path.write_bytes(path.read_bytes() + b"\n// original changed during native BDD\n")
            else:
                text = target.read_text()
                if kind == "policy":
                    text = text.replace("[default = by name]", "[default = by order]")
                else:
                    text = (
                        text.replace("argument0[0]", "temporary")
                        .replace("argument0[1]", "argument0[0]")
                        .replace("temporary", "argument0[1]")
                    )
                target.write_text(text)

    change_native_artifact.__dict__["__cantrace__"] = True
    sys.addaudithook(change_native_artifact)
    with pytest.raises(
        ValueError, match="policy|named input/output order differs|changed|export record differs"
    ):
        prove_complete_bdd_mapping(case, changed, tools)
    assert altered
    assert not (directory / "proof-record.json").exists()
