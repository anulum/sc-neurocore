# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Complete native GIA protocol and scope contracts

"""Exercise actual complete GIA proofs, last-output defects and retained-proof custody."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

from tools.function_synthesis_aiger import AigerInterface, prove_aiger_equivalence
from tools.function_synthesis_gia import (
    check_gia_equivalent_verdict,
    prove_gia_equivalence,
    read_gia_equivalence,
)
from tools.function_synthesis_records import SynthesisCase
from tools.function_synthesis_toolchain import SynthesisToolchain
from tools.function_synthesis_yosys import run_function_synthesis

GiaCase = tuple[Path, SynthesisToolchain, AigerInterface]


@pytest.fixture(scope="module")
def gia_case(tmp_path_factory: pytest.TempPathFactory) -> GiaCase:
    """Acquire an actual complete named model and prove all eight outputs with the native protocol."""
    root = tmp_path_factory.mktemp("real-complete-gia")
    tools = SynthesisToolchain.inspect(root / "inspection")
    run_function_synthesis(SynthesisCase("floor", 8, 0), root / "source", tools)
    proof = root / "proof"
    proof.mkdir()
    for name in ("reference.aig", "candidate.aig"):
        shutil.copyfile(root / "source/reference.aig", proof / name)
    interface = AigerInterface(
        frozenset(f"argument0[{index}]" for index in range(8)),
        frozenset(f"result[{index}]" for index in range(8)),
    )
    prove_gia_equivalence(
        proof / "reference.aig", proof / "candidate.aig", interface, tools, proof, timeout=90
    )
    return proof, tools, interface


def _snapshot(root: Path) -> dict[str, tuple[str, int]]:
    """Read all actual native artifact hashes and original modification timestamps."""
    return {
        p.name: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
        for p in root.iterdir()
    }


def test_every_output_has_a_difference_before_scalar_aggregation(gia_case: GiaCase) -> None:
    """Keep every complete original comparison before combining it into one native miter output."""
    root, tools, interface = gia_case
    before = _snapshot(root)
    for name, expected in (("all-differences.aig", (8, 0, 8)), ("singleton.aig", (8, 0, 1))):
        with (root / name).open("rb") as stream:
            fields = stream.readline().split()
        assert tuple(int(fields[index]) for index in (2, 3, 4)) == expected
    read_gia_equivalence(root / "reference.aig", root / "candidate.aig", interface, tools, root)
    assert _snapshot(root) == before
    record = json.loads((root / "gia-proof.json").read_text())
    assert record["all_compared_output_bits"] == 8
    assert record["singleton_output_bits"] == 1
    assert record["input_bits"] == 8 and record["latches"] == 0


@pytest.mark.parametrize(
    "artifact",
    [
        "reference.aig",
        "candidate.aig",
        "reference-plain.aig",
        "candidate-plain.aig",
        "cec-help.log",
        "reference-io.log",
        "candidate-io.log",
        "reference-roundtrip.log",
        "candidate-roundtrip.log",
        "gia-construction.command.json",
        "all-differences.aig",
        "singleton.aig",
        "equivalence.log",
        "gia-proof.json",
    ],
)
def test_changed_actual_complete_proof_is_preserved_and_refused(
    gia_case: GiaCase, tmp_path: Path, artifact: str
) -> None:
    """Reject changed actual models, transformation receipts and final complete proof custody."""
    source, tools, interface = gia_case
    root = tmp_path / "changed"
    shutil.copytree(source, root)
    path = root / artifact
    path.write_bytes(path.read_bytes() + b"\n")
    before = _snapshot(root)
    with pytest.raises(ValueError):
        read_gia_equivalence(root / "reference.aig", root / "candidate.aig", interface, tools, root)
    assert _snapshot(root) == before


@pytest.mark.parametrize(
    "artifact", ["directory", "gia-proof.json", "candidate.aig", "singleton.aig"]
)
def test_linked_complete_gia_custody_is_refused(
    gia_case: GiaCase, tmp_path: Path, artifact: str
) -> None:
    """Require direct regular proof custody even when linked targets contain genuine evidence."""
    source, tools, interface = gia_case
    root = tmp_path / "linked"
    shutil.copytree(source, root)
    path = root if artifact == "directory" else root / artifact
    target = tmp_path / "retained-target"
    path.rename(target)
    path.symlink_to(target, target_is_directory=target.is_dir())
    with pytest.raises(ValueError):
        read_gia_equivalence(root / "reference.aig", root / "candidate.aig", interface, tools, root)
    assert path.is_symlink() and target.exists()


@pytest.mark.parametrize("seconds", [True, 0.0, -1.0, float("nan"), float("inf"), 91.0])
def test_invalid_gia_budget_preserves_completed_actual_evidence(
    gia_case: GiaCase, seconds: float
) -> None:
    """Keep the actual ninety-second proof ceiling and reject invalid caller budgets."""
    root, tools, interface = gia_case
    before = _snapshot(root)
    with pytest.raises(ValueError, match="finite native deadline"):
        prove_gia_equivalence(
            root / "reference.aig", root / "candidate.aig", interface, tools, root, timeout=seconds
        )
    assert _snapshot(root) == before


@pytest.mark.parametrize("seconds", [0.0, -1.0, float("nan"), float("inf"), 91.0])
def test_retained_native_gia_deadline_is_validated_without_restamping(
    gia_case: GiaCase, tmp_path: Path, seconds: float
) -> None:
    """Reject an invalid retained native deadline while preserving its actual proof bytes."""
    source, tools, interface = gia_case
    root = tmp_path / "invalid-retained-deadline"
    shutil.copytree(source, root)
    record = root / "gia-proof.json"
    payload = json.loads(record.read_text())
    payload["native_seconds"] = seconds
    record.write_text(json.dumps(payload))
    before = _snapshot(root)
    with pytest.raises(ValueError, match="original finite native deadline"):
        read_gia_equivalence(root / "reference.aig", root / "candidate.aig", interface, tools, root)
    assert _snapshot(root) == before


@pytest.mark.parametrize(
    "field",
    [
        "recorded_at",
        "construction",
        "toolchain_sha256",
        "producer_sha256",
        "input_bits",
        "output_order",
        "latches",
    ],
)
def test_changed_original_binding_cannot_restamp_a_native_proof(
    gia_case: GiaCase, tmp_path: Path, field: str
) -> None:
    """Refuse stale identities, omitted comparison bits and rewritten qualification timestamps."""
    source, tools, interface = gia_case
    root = tmp_path / "binding"
    shutil.copytree(source, root)
    path = root / "gia-proof.json"
    record = json.loads(path.read_text())
    record[field] = None
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        read_gia_equivalence(root / "reference.aig", root / "candidate.aig", interface, tools, root)


def test_existing_complete_native_acquisition_is_never_overwritten(gia_case: GiaCase) -> None:
    """Preserve all original native commands instead of rebuilding over a completed proof."""
    root, tools, interface = gia_case
    before = _snapshot(root)
    with pytest.raises(ValueError, match="must not overwrite"):
        prove_gia_equivalence(
            root / "reference.aig", root / "candidate.aig", interface, tools, root, timeout=90
        )
    assert _snapshot(root) == before


@pytest.mark.parametrize(
    "timestamp", ["2026-10-30T00:00:00Z", "2026-10-30T00:00:00+01:00", "2026-10-30T00:00:00"]
)
def test_original_gia_time_must_be_canonical_utc(
    gia_case: GiaCase, tmp_path: Path, timestamp: str
) -> None:
    """Refuse alternate timezone or naive representations without rewriting an actual proof."""
    source, tools, interface = gia_case
    root = tmp_path / "timestamp"
    shutil.copytree(source, root)
    path = root / "gia-proof.json"
    record = json.loads(path.read_text())
    record["recorded_at"] = timestamp
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="canonical UTC"):
        read_gia_equivalence(root / "reference.aig", root / "candidate.aig", interface, tools, root)


def test_actual_changed_matching_policy_is_not_a_retained_proof(
    gia_case: GiaCase, tmp_path: Path
) -> None:
    """Reject an actual help receipt changed to advertise another default matching policy."""
    source, tools, interface = gia_case
    root = tmp_path / "policy"
    shutil.copytree(source, root)
    path = root / "cec-help.log"
    path.write_text(path.read_text().replace("[default = by name]", "[default = by order]"))
    with pytest.raises(ValueError, match="matching policy"):
        read_gia_equivalence(root / "reference.aig", root / "candidate.aig", interface, tools, root)


def test_actual_named_order_difference_does_not_become_by_order_equality(
    gia_case: GiaCase, tmp_path: Path
) -> None:
    """Keep the named comparison valid while refusing its differently ordered GIA interpretation."""
    _, tools, _ = gia_case
    root = tmp_path / "permuted"
    root.mkdir()
    for label, ports in (
        ("reference", "input [7:0] argument0, input [7:0] argument1"),
        ("candidate", "input [7:0] argument1, input [7:0] argument0"),
    ):
        (root / (label + ".v")).write_text(
            "module function_unit(" + ports + ", output [7:0] result);\n"
            "assign result = argument0 ^ argument1;\nendmodule\n"
        )
        (root / (label + ".ys")).write_text(
            "read_verilog " + label + ".v\nhierarchy -check -top function_unit\n"
            "proc\nopt_clean\ntechmap\nopt_clean\naigmap\nopt_clean\n"
            "check -assert\nwrite_aiger -symbols " + label + ".aig\n"
        )
        tools.run(
            [str(tools.yosys), "-Q", "-T", label + ".ys"],
            root,
            root / (label + "-synthesis.log"),
            timeout=90,
        )
    interface = AigerInterface(
        frozenset(f"argument{port}[{index}]" for port in range(2) for index in range(8)),
        frozenset(f"result[{index}]" for index in range(8)),
    )
    good = tmp_path / "original-complete-order"
    good.mkdir()
    for name in ("reference.aig", "candidate.aig"):
        shutil.copyfile(root / "reference.aig", good / name)
    prove_gia_equivalence(
        good / "reference.aig", good / "candidate.aig", interface, tools, good, timeout=90
    )
    with pytest.raises(ValueError, match="identical complete named"):
        prove_gia_equivalence(
            root / "reference.aig", root / "candidate.aig", interface, tools, root, timeout=90
        )
    damaged = tmp_path / "changed-retained-order"
    shutil.copytree(good, damaged)
    for name in ("candidate.aig", "candidate-io.log"):
        shutil.copyfile(root / name, damaged / name)
    with pytest.raises(ValueError, match="identical complete named"):
        read_gia_equivalence(
            damaged / "reference.aig", damaged / "candidate.aig", interface, tools, damaged
        )
    named = tmp_path / "actual-named-proof"
    named.mkdir()
    for name in ("reference.aig", "candidate.aig"):
        shutil.copyfile(root / name, named / name)
    prove_aiger_equivalence(
        named / "reference.aig", named / "candidate.aig", interface, tools, named, timeout=90
    )


@pytest.mark.parametrize("target", ["policy", "model"])
def test_actual_native_gia_acquisition_changes_are_preserved_and_refused(
    gia_case: GiaCase, tmp_path: Path, target: str
) -> None:
    """Reject actual receipt or model changes at their native acquisition boundaries."""
    source, tools, interface = gia_case
    for name in ("reference.aig", "candidate.aig"):
        shutil.copyfile(source / name, tmp_path / name)
    changed = False
    policy = tmp_path / "cec-help.log"

    def change_acquisition(event: str, args: tuple[object, ...]) -> None:
        """Change the real native receipt before reading, or the model as construction starts."""
        nonlocal changed
        if (
            event == "open"
            and args[0] in (policy, str(policy), str(policy).encode())
            and args[1] == "r"
            and target == "policy"
            and not changed
        ):
            changed = True
            text = policy.read_text()
            assert "[default = by name]" in text
            policy.write_text(text.replace("[default = by name]", "[default = by order]"))
        elif (
            event == "subprocess.Popen"
            and args[2] in (tmp_path, str(tmp_path), str(tmp_path).encode())
            and target == "model"
            and (tmp_path / "gia-construction.command.json").is_file()
            and not changed
        ):
            changed = True
            with (tmp_path / "candidate.aig").open("ab") as stream:
                stream.write(b"\nactual model changed at its native subprocess boundary\n")

    change_acquisition.__dict__["__cantrace__"] = True
    sys.addaudithook(change_acquisition)
    expected = (
        "matching policy"
        if target == "policy"
        else "models or the GIA proof implementation changed"
    )
    with pytest.raises(ValueError, match=expected):
        prove_gia_equivalence(
            tmp_path / "reference.aig",
            tmp_path / "candidate.aig",
            interface,
            tools,
            tmp_path,
            timeout=90,
        )
    assert changed
    assert not (tmp_path / "gia-proof.json").exists()
    if target == "model":
        check_gia_equivalent_verdict(tmp_path / "equivalence.log")


def test_genuine_fault_in_last_output_cannot_hide_behind_a_correct_first_output(
    gia_case: GiaCase, tmp_path: Path
) -> None:
    """Refuse a real last-output inversion while the first seven complete outputs still match."""
    source, tools, interface = gia_case
    for name in ("reference.aig", "candidate.aig"):
        shutil.copyfile(source / name, tmp_path / name)
    path = tmp_path / "candidate.aig"
    data = path.read_bytes()
    cursor = data.index(b"\n") + 1
    for _ in range(7):
        cursor = data.index(b"\n", cursor) + 1
    end = data.index(b"\n", cursor)
    literal = int(data[cursor:end])
    path.write_bytes(data[:cursor] + str(literal ^ 1).encode() + data[end:])
    with pytest.raises(ValueError):
        prove_gia_equivalence(
            tmp_path / "reference.aig", path, interface, tools, tmp_path, timeout=90
        )
    assert "NOT EQUIVALENT" in (tmp_path / "equivalence.log").read_text().upper()
    assert not (tmp_path / "gia-proof.json").exists()


@pytest.mark.parametrize("name", ["reference-plain.aig", "singleton.aig", "gia-proof.json"])
@pytest.mark.parametrize("kind", ["file", "directory", "dangling"])
def test_new_native_proof_preserves_preexisting_protocol_artifacts(
    gia_case: GiaCase, tmp_path: Path, name: str, kind: str
) -> None:
    """Refuse overwriting retained owner bytes before any native protocol invocation."""
    source, tools, interface = gia_case
    root = tmp_path / "refused"
    root.mkdir()
    for original in ("reference.aig", "candidate.aig"):
        shutil.copyfile(source / original, root / original)
    path = root / name
    if kind == "file":
        path.write_bytes(b"owner-retained-data")
    elif kind == "directory":
        path.mkdir()
    else:
        path.symlink_to(root / "absent-target")
    before = sorted(p.name for p in root.iterdir())
    with pytest.raises(ValueError, match="overwrite"):
        prove_gia_equivalence(
            root / "reference.aig", root / "candidate.aig", interface, tools, root, timeout=90
        )
    assert sorted(p.name for p in root.iterdir()) == before
    if kind == "file":
        assert path.read_bytes() == b"owner-retained-data"
    elif kind == "directory":
        assert path.is_dir()
    else:
        assert path.is_symlink()


@pytest.mark.parametrize(
    "artifact",
    ["reference-normalize", "candidate-normalize", "gia-construction", "equivalence"],
)
@pytest.mark.parametrize("damage", ["command", "error"])
def test_changed_native_protocol_is_refused_even_with_updated_inventory(
    gia_case: GiaCase, tmp_path: Path, artifact: str, damage: str
) -> None:
    """Recompute native protocol admission independently of a rewritten digest inventory."""
    source, tools, interface = gia_case
    root = tmp_path / "changed-protocol"
    shutil.copytree(source, root)
    path = root / (artifact + (".command.json" if damage == "command" else ".log"))
    if damage == "command":
        argv = json.loads(path.read_bytes())
        argv[-1] += "; version"
        path.write_text(json.dumps(argv))
    else:
        with path.open("a") as output:
            output.write("  ERROR: actual retained transformation was not completed\n")
    record_path = root / "gia-proof.json"
    record = json.loads(record_path.read_bytes())
    record["artifact_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    record_path.write_text(json.dumps(record))
    before = _snapshot(root)
    with pytest.raises(ValueError, match="protocol|transformation"):
        read_gia_equivalence(root / "reference.aig", root / "candidate.aig", interface, tools, root)
    assert _snapshot(root) == before
