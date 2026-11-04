# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Complete source-normalization proof contracts

"""Prove real normalization graphs and reject damaged or genuinely different native evidence."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from tools.function_synthesis_aiger import AigerInterface, prove_aiger_equivalence
from tools.function_synthesis_gia import prove_gia_equivalence
from tools.function_synthesis_records import SynthesisCase
from tools.function_synthesis_source_proof import (
    SOURCE_AIG_FLOWS,
    prove_source_normalization,
    read_native_equivalence,
    read_source_normalization,
)
from tools.function_synthesis_toolchain import SynthesisToolchain

SourceCase = tuple[str, SynthesisCase, Path, SynthesisToolchain]


def _snapshot(root: Path) -> dict[str, tuple[str, int]]:
    """Retain actual bytes and original timestamps of every regular source artifact."""
    return {
        path.relative_to(root).as_posix(): (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_mtime_ns,
        )
        for path in root.rglob("*")
        if path.is_file()
    }


def _prepare(root: Path, tools: SynthesisToolchain, left: str, right: str) -> None:
    """Acquire independent original and checkpoint graphs with the real native compiler."""
    for name, expression in [("original", left), ("reference", right)]:
        phase = root / name
        phase.mkdir()
        (phase / "unit.v").write_text(
            "module function_unit(input signed [7:0] argument0, argument1,"
            " output signed [7:0] result);\n"
            f"assign result = {expression};\nendmodule\n"
        )
        (phase / "flow.ys").write_text(
            "read_verilog -sv unit.v\nhierarchy -check -top function_unit\n"
            f"proc\nopt_clean\nwrite_json ../{name}-word.json\n"
        )
        tools.run(
            [str(tools.yosys), "-Q", "-T", "flow.ys"], phase, phase / "native.log", timeout=90
        )


def _interface() -> AigerInterface:
    """Name the complete sixteen-bit binary input and eight-bit result domain."""
    return AigerInterface(
        frozenset(f"argument{argument}[{bit}]" for argument in range(2) for bit in range(8)),
        frozenset(f"result[{bit}]" for bit in range(8)),
    )


@pytest.fixture(scope="module", params=["identical", "reassociated", "division", "unsupported"])
def source_case(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> SourceCase:
    """Prove real component graphs; the case supplies geometry, not an atan2 accuracy claim."""
    mode = str(request.param)
    root = tmp_path_factory.mktemp("actual-source-" + mode)
    tools = SynthesisToolchain.inspect(root / "inspection")
    choices = {
        "identical": ("argument0 + argument1", "argument0 + argument1"),
        "reassociated": ("(argument0 + argument1) + 8'sd3", "(argument0 + 8'sd3) + argument1"),
        "division": (
            "argument1 == 0 ? 8'sd0 : ($signed(argument0) / $signed(argument1))",
            "argument1 == 0 ? 8'sd0 : ($signed(argument0) / $signed(argument1))",
        ),
        "unsupported": ("argument0 ^ argument1", "argument0 ^ argument1"),
    }
    left, right = choices[mode]
    _prepare(root, tools, left, right)
    case = SynthesisCase("atan2", 8, 4)
    prove_source_normalization(case, root, tools, timeout=90)
    return mode, case, root, tools


def test_complete_original_domain_and_retained_readonly_result(source_case: SourceCase) -> None:
    """Require actual structural identity or decisive complete native equivalence, without restamping."""
    mode, case, root, tools = source_case
    before = _snapshot(root)
    method = read_source_normalization(case, root, tools)
    assert method == ("structural" if mode in {"identical", "division"} else "complete-aiger")
    assert _snapshot(root) == before
    if mode == "unsupported":
        record = json.loads((root / "source-structure/structural-comparison.json").read_text())
        assert record == {
            "all_output_function_keys_identical": None,
            "structural_interpretation": "unsupported",
            "unsupported_operation": "$xor",
        }


@pytest.mark.parametrize("source_case", ["division"], indirect=True)
def test_division_structural_identity_has_independent_full_native_calibration(
    source_case: SourceCase, tmp_path: Path
) -> None:
    """Independently prove the signed division graph, including the guarded zero divisor."""
    _, _, original, tools = source_case
    for name in ["original-word.json", "reference-word.json"]:
        shutil.copyfile(original / name, tmp_path / name)
    for stem, flow in SOURCE_AIG_FLOWS.items():
        (tmp_path / f"{stem}.ys").write_text(flow)
        tools.run(
            [str(tools.yosys), "-Q", "-T", f"{stem}.ys"],
            tmp_path,
            tmp_path / f"{stem}-native.log",
            timeout=90,
        )
    prove_aiger_equivalence(
        tmp_path / "reference.aig",
        tmp_path / "candidate.aig",
        _interface(),
        tools,
        tmp_path,
        timeout=90,
    )
    read_native_equivalence(tmp_path, "reference.aig", "candidate.aig", _interface(), tools)


@pytest.mark.parametrize("source_case", ["identical"], indirect=True)
@pytest.mark.parametrize("timeout", [True, 0.0, -1.0, float("inf"), float("nan"), 91.0])
def test_invalid_native_deadline_preserves_actual_inputs(
    source_case: SourceCase, tmp_path: Path, timeout: float
) -> None:
    """Refuse invalid source-proof budgets before creating any proof output."""
    _, case, original, tools = source_case
    for name in ["original-word.json", "reference-word.json"]:
        shutil.copyfile(original / name, tmp_path / name)
    before = _snapshot(tmp_path)
    with pytest.raises(ValueError, match="finite native deadline"):
        prove_source_normalization(case, tmp_path, tools, timeout=timeout)
    assert _snapshot(tmp_path) == before


def test_genuinely_different_native_checkpoint_is_not_equivalent(tmp_path: Path) -> None:
    """Refuse a real separately compiled wrong result, rather than a fabricated verdict."""
    tools = SynthesisToolchain.inspect(tmp_path / "inspection")
    _prepare(tmp_path, tools, "argument0 + argument1", "(argument0 + argument1) ^ 8'd1")
    with pytest.raises(ValueError):
        prove_source_normalization(SynthesisCase("atan2", 8, 4), tmp_path, tools, timeout=90)
    assert "NOT EQUIVALENT" in (tmp_path / "source-native/equivalence.log").read_text()


@pytest.mark.parametrize("source_case", ["identical"], indirect=True)
def test_malformed_word_geometry_is_not_sent_to_native_fallback(
    source_case: SourceCase, tmp_path: Path
) -> None:
    """Keep structural validation failures distinct from a supported native fallback request."""
    _, case, original, tools = source_case
    for name in ["original-word.json", "reference-word.json"]:
        shutil.copyfile(original / name, tmp_path / name)
    path = tmp_path / "original-word.json"
    value = json.loads(path.read_text())
    cell = next(iter(value["modules"]["function_unit"]["cells"].values()))
    cell["parameters"]["A_WIDTH"] = "00000000000000000000000000000001"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="geometry"):
        prove_source_normalization(case, tmp_path, tools, timeout=90)
    assert not (tmp_path / "source-native").exists()


@pytest.mark.parametrize("source_case", ["identical"], indirect=True)
@pytest.mark.parametrize("damage", ["roots", "type", "link"])
def test_damaged_structural_identity_is_refused(
    source_case: SourceCase, tmp_path: Path, damage: str
) -> None:
    """Refuse incomplete output identities, malformed records and linked source evidence."""
    _, case, original, tools = source_case
    root = tmp_path / "proof"
    shutil.copytree(original, root)
    path = root / "source-structure/structural-comparison.json"
    if damage == "link":
        path.unlink()
        path.symlink_to(original / "source-structure/structural-comparison.json")
    else:
        value = json.loads(path.read_text())
        if damage == "roots":
            value["original_output_roots"].pop()
        else:
            value = []
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        read_source_normalization(case, root, tools)


@pytest.mark.parametrize("source_case", ["reassociated"], indirect=True)
def test_native_equivalence_cannot_be_relabelled_as_structural_identity(
    source_case: SourceCase, tmp_path: Path
) -> None:
    """Recompute actual source graphs before admitting a rewritten positive structural flag."""
    _, case, original, tools = source_case
    root = tmp_path / "proof"
    shutil.copytree(original, root)
    path = root / "source-structure/structural-comparison.json"
    value = json.loads(path.read_text())
    assert value["all_output_function_keys_identical"] is False
    value["all_output_function_keys_identical"] = True
    value["mapped_output_roots"] = value["original_output_roots"]
    path.write_text(json.dumps(value))
    before = _snapshot(root)
    with pytest.raises(ValueError, match="differs from the actual source graphs"):
        read_source_normalization(case, root, tools)
    assert _snapshot(root) == before


@pytest.mark.parametrize("source_case", ["identical"], indirect=True)
@pytest.mark.parametrize("damage", ["equal-fabricated-roots", "node-count"])
def test_structural_roots_and_size_are_reconstructed_from_native_checkpoints(
    source_case: SourceCase, tmp_path: Path, damage: str
) -> None:
    """Reject coherent root-list or expression-count edits without changing retained artifacts."""
    _, case, original, tools = source_case
    root = tmp_path / "proof"
    shutil.copytree(original, root)
    path = root / "source-structure/structural-comparison.json"
    value = json.loads(path.read_text())
    assert value["all_output_function_keys_identical"] is True
    if damage == "equal-fabricated-roots":
        value["original_output_roots"] = [0] * case.data_width
        value["mapped_output_roots"] = [0] * case.data_width
    else:
        value["canonical_nodes"] += 1
    path.write_text(json.dumps(value))
    before = _snapshot(root)
    with pytest.raises(ValueError, match="differs from the actual source graphs"):
        read_source_normalization(case, root, tools)
    assert _snapshot(root) == before


@pytest.mark.parametrize("source_case", ["unsupported"], indirect=True)
@pytest.mark.parametrize(
    "damage", ["flag", "missing", "status", "operation-type", "operation-value"]
)
def test_unqualified_operation_trace_cannot_substitute_for_proof(
    source_case: SourceCase, tmp_path: Path, damage: str
) -> None:
    """Require the exact actual unsupported-operation protocol before considering native evidence."""
    _, case, original, tools = source_case
    root = tmp_path / "proof"
    shutil.copytree(original, root)
    path = root / "source-structure/structural-comparison.json"
    value = json.loads(path.read_text())
    if damage == "flag":
        value["all_output_function_keys_identical"] = 0
    elif damage == "missing":
        del value["all_output_function_keys_identical"]
    elif damage == "status":
        value["structural_interpretation"] = "not-a-completed-interpretation"
    elif damage == "operation-type":
        value["unsupported_operation"] = True
    else:
        value["unsupported_operation"] = "$not_the_actual_operation"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="no completed result"):
        read_source_normalization(case, root, tools)


@pytest.mark.parametrize("source_case", ["reassociated"], indirect=True)
@pytest.mark.parametrize(
    "name", ["reference.ys", "candidate.ys", "reference-word.json", "original-word.json"]
)
def test_actual_native_source_inputs_must_match_the_original_checkpoints(
    source_case: SourceCase, tmp_path: Path, name: str
) -> None:
    """Refuse changed native commands or either independently bound source export."""
    _, case, original, tools = source_case
    root = tmp_path / "proof"
    shutil.copytree(original, root)
    with (root / "source-native" / name).open("ab") as stream:
        stream.write(b"\n")
    before = _snapshot(root)
    with pytest.raises(ValueError, match="different original or checkpoint exports"):
        read_source_normalization(case, root, tools)
    assert _snapshot(root) == before


@pytest.mark.parametrize("source_case", ["reassociated"], indirect=True)
def test_complete_gia_reader_and_dangling_record_refusal(
    source_case: SourceCase, tmp_path: Path
) -> None:
    """Read a genuine GIA proof and reject altered fallback evidence or a linked record."""
    _, _, original, tools = source_case
    for name in ["reference.aig", "candidate.aig"]:
        shutil.copyfile(original / "source-native" / name, tmp_path / name)
    prove_gia_equivalence(
        tmp_path / "reference.aig",
        tmp_path / "candidate.aig",
        _interface(),
        tools,
        tmp_path,
        timeout=90,
    )
    read_native_equivalence(tmp_path, "reference.aig", "candidate.aig", _interface(), tools)
    verdict = tmp_path / "equivalence.log"
    original_verdict = verdict.read_bytes()
    assert b"Calling the old CEC engine." in original_verdict
    verdict.write_bytes(original_verdict.replace(b"Calling the old CEC engine.", b"", 1))
    with pytest.raises(ValueError, match="GIA fallback did not prove"):
        read_native_equivalence(tmp_path, "reference.aig", "candidate.aig", _interface(), tools)
    verdict.write_bytes(original_verdict)
    read_native_equivalence(tmp_path, "reference.aig", "candidate.aig", _interface(), tools)
    record = tmp_path / "gia-proof.json"
    record.unlink()
    record.symlink_to(tmp_path / "missing-gia-record")
    with pytest.raises(ValueError):
        read_native_equivalence(tmp_path, "reference.aig", "candidate.aig", _interface(), tools)
