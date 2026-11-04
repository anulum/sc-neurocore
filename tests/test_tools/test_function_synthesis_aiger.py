# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Complete native AIG proof admission

"""Exercise native complete-domain checks and deliberate corruption of actual receipts."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

from tools.function_synthesis_aiger import (
    AigerInterface,
    check_aiger_header,
    check_aiger_names,
    check_equivalent_verdict,
    prove_aiger_equivalence,
)
from tools.function_synthesis_records import SynthesisCase
from tools.function_synthesis_toolchain import SynthesisToolchain
from tools.function_synthesis_yosys import run_function_synthesis

AigCase = tuple[Path, SynthesisToolchain, AigerInterface]


@pytest.fixture(scope="module")
def aig_case(tmp_path_factory: pytest.TempPathFactory) -> AigCase:
    """Acquire a real complete zero-cell function and explicit native comparison receipts."""
    root = tmp_path_factory.mktemp("actual-aiger")
    tools = SynthesisToolchain.inspect(root / "inspection")
    run_function_synthesis(SynthesisCase("floor", 8, 0), root / "original", tools)
    proof = root / "native"
    proof.mkdir()
    for name in ["reference.aig", "candidate.aig"]:
        shutil.copyfile(root / "original/reference.aig", proof / name)
    interface = AigerInterface(
        frozenset(f"argument0[{index}]" for index in range(8)),
        frozenset(f"result[{index}]" for index in range(8)),
    )
    prove_aiger_equivalence(
        proof / "reference.aig", proof / "candidate.aig", interface, tools, proof, timeout=90
    )
    return proof, tools, interface


def test_real_complete_native_interface_and_positive_verdict(aig_case: AigCase) -> None:
    """Admit actual ABC results with all original input and output names present exactly once."""
    root, _, interface = aig_case
    for name, label in [("reference.aig", "reference"), ("candidate.aig", "candidate")]:
        check_aiger_header(root / name, interface)
        check_aiger_names(root / (label + "-io.log"), interface)
    check_equivalent_verdict(root / "equivalence.log")
    assert "[default = by name]" in (root / "cec-help.log").read_text()


@pytest.mark.parametrize(
    "header",
    [
        b"aag 8 8 0 8 0\n",
        b"aig 8 8 0 8 0",
        b"aig 8 8 0 8\n",
        b"aig x 8 0 8 0\n",
        b"aig 8 7 0 8 1\n",
        b"aig 8 8 1 8 0\n",
        b"aig 8 8 0 7 0\n",
        b"aig 9 8 0 8 0\n",
    ],
)
def test_damaged_actual_export_header_is_refused(
    aig_case: AigCase, tmp_path: Path, header: bytes
) -> None:
    """Refuse malformed, sequential, narrowed and inconsistent real-export geometry."""
    root, _, interface = aig_case
    path = tmp_path / "damaged.aig"
    path.write_bytes(header + (root / "reference.aig").read_bytes().split(b"\n", 1)[1])
    with pytest.raises(ValueError):
        check_aiger_header(path, interface)


def test_linked_actual_export_cannot_enter_native_proof(aig_case: AigCase, tmp_path: Path) -> None:
    """Require direct regular export custody even when linked target bytes are valid."""
    root, _, interface = aig_case
    path = tmp_path / "linked.aig"
    path.symlink_to(root / "reference.aig")
    with pytest.raises(ValueError, match="regular"):
        check_aiger_header(path, interface)


@pytest.mark.parametrize("mutation", ["missing", "duplicate-line", "name", "index", "token"])
def test_actual_native_interface_receipt_damage_is_refused(
    aig_case: AigCase, tmp_path: Path, mutation: str
) -> None:
    """Detect omitted, repeated, substituted and malformed original primary input identities."""
    root, _, interface = aig_case
    text = (root / "reference-io.log").read_text()
    line = next(line for line in text.splitlines() if line.startswith("Primary inputs"))
    if mutation == "missing":
        text = text.replace(line, "")
    elif mutation == "duplicate-line":
        text += line + "\n"
    elif mutation == "name":
        text = text.replace("argument0[0]", "another_input[0]")
    elif mutation == "index":
        text = text.replace("1=argument0[1]", "0=argument0[1]")
    else:
        text = text.replace("0=argument0[0]", "argument0[0]")
    path = tmp_path / "damaged-io.log"
    path.write_text(text)
    with pytest.raises(ValueError):
        check_aiger_names(path, interface)


@pytest.mark.parametrize("mutation", ["absent", "negative", "conflict", "error"])
def test_damaged_actual_verdict_never_becomes_positive(
    aig_case: AigCase, tmp_path: Path, mutation: str
) -> None:
    """Keep actual native success distinct from missing, conflicting or errored receipts."""
    root, _, _ = aig_case
    text = (root / "equivalence.log").read_text()
    if mutation == "absent":
        text = "\n".join(line for line in text.splitlines() if not line.startswith("Networks are "))
    elif mutation == "negative":
        text = text.replace("Networks are equivalent", "Networks are NOT EQUIVALENT")
    elif mutation == "conflict":
        text += "Networks are NOT EQUIVALENT.\n"
    else:
        text += "ERROR: deliberately damaged retained proof\n"
    path = tmp_path / "damaged-verdict.log"
    path.write_text(text)
    with pytest.raises(ValueError, match="not proved"):
        check_equivalent_verdict(path)


@pytest.mark.parametrize("name", ["", "bad name", "a=b"])
def test_incomplete_named_domain_is_refused(name: str) -> None:
    """Refuse names that cannot identify complete individual native port bits."""
    with pytest.raises(ValueError, match="named bit domains"):
        AigerInterface(frozenset({name}), frozenset({"result"}))


def test_empty_primary_domain_is_refused() -> None:
    """Prevent vacuous native proofs that discard the original independent inputs."""
    with pytest.raises(ValueError, match="nonempty"):
        AigerInterface(frozenset(), frozenset({"result"}))


@pytest.mark.parametrize("seconds", [True, 0.0, -1.0, float("nan"), float("inf"), 91.0])
def test_native_deadline_admission_never_rewrites_existing_receipts(
    aig_case: AigCase, seconds: float
) -> None:
    """Keep the native proof ceiling at ninety seconds and reject invalid caller budgets."""
    root, tools, interface = aig_case
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    with pytest.raises(ValueError, match="admitted deadline"):
        prove_aiger_equivalence(
            root / "reference.aig", root / "candidate.aig", interface, tools, root, timeout=seconds
        )
    assert {p.name: p.read_bytes() for p in root.iterdir()} == before


def test_native_export_paths_are_distinct_local_names(aig_case: AigCase, tmp_path: Path) -> None:
    """Reject one-export, outside-directory and unsafe command-token proof requests."""
    root, tools, interface = aig_case
    reference = root / "reference.aig"
    for candidate in [reference, tmp_path / "outside.aig", root / "unsafe;candidate.aig"]:
        with pytest.raises(ValueError, match="distinct local"):
            prove_aiger_equivalence(reference, candidate, interface, tools, root, timeout=90)


def test_real_connected_output_fault_has_a_native_negative_verdict(
    aig_case: AigCase, tmp_path: Path
) -> None:
    """Compare all eight outputs of an actual inverted function and retain native NOT EQUIVALENT."""
    original, tools, interface = aig_case
    shutil.copyfile(original / "reference.aig", tmp_path / "reference.aig")
    (tmp_path / "fault.v").write_text(
        "module function_unit(input [7:0] argument0, output [7:0] result);\n"
        "assign result = ~argument0;\nendmodule\n"
    )
    (tmp_path / "flow.ys").write_text(
        "read_verilog fault.v\nhierarchy -check -top function_unit\n"
        "proc\nopt\ntechmap\nopt_clean\naigmap\nopt_clean\n"
        "check -assert\nwrite_aiger -symbols candidate.aig\n"
    )
    tools.run(
        [str(tools.yosys), "-Q", "-T", "flow.ys"], tmp_path, tmp_path / "synthesis.log", timeout=90
    )
    with pytest.raises(ValueError, match="not proved"):
        prove_aiger_equivalence(
            tmp_path / "reference.aig",
            tmp_path / "candidate.aig",
            interface,
            tools,
            tmp_path,
            timeout=90,
        )
    assert "Networks are NOT EQUIVALENT" in (tmp_path / "equivalence.log").read_text()


@pytest.mark.parametrize("target", ["native-policy-receipt", "candidate-export"])
def test_actual_native_acquisition_tampering_is_refused(
    aig_case: AigCase, tmp_path: Path, target: str
) -> None:
    """Refuse genuinely changed exports or a changed native matching-policy receipt."""
    original, tools, interface = aig_case
    for name in ["reference.aig", "candidate.aig"]:
        shutil.copyfile(original / name, tmp_path / name)
    changed = False
    policy = tmp_path / "cec-help.log"

    def change_acquisition(event: str, args: tuple[object, ...]) -> None:
        """Alter the real completed native receipt or the export at its actual command boundary."""
        nonlocal changed
        if (
            event == "open"
            and args[0] in (policy, str(policy), str(policy).encode())
            and args[1] == "r"
            and target == "native-policy-receipt"
            and not changed
        ):
            changed = True
            text = policy.read_text()
            assert "[default = by name]" in text
            policy.write_text(text.replace("[default = by name]", "[default = by order]"))
        elif (
            event == "subprocess.Popen"
            and args[2] in (tmp_path, str(tmp_path), str(tmp_path).encode())
            and target == "candidate-export"
            and not changed
        ):
            assert (tmp_path / "cec-help.command.json").is_file()
            changed = True
            with (tmp_path / "candidate.aig").open("ab") as stream:
                stream.write(b"\nactual acquisition changed at its native subprocess boundary\n")

    change_acquisition.__dict__["__cantrace__"] = True
    sys.addaudithook(change_acquisition)
    expected = (
        "matching complete inputs by name"
        if target == "native-policy-receipt"
        else "exports changed during their proof"
    )
    with pytest.raises(ValueError, match=expected):
        prove_aiger_equivalence(
            tmp_path / "reference.aig",
            tmp_path / "candidate.aig",
            interface,
            tools,
            tmp_path,
            timeout=90,
        )
    assert changed
    if target == "candidate-export":
        check_equivalent_verdict(tmp_path / "equivalence.log")
