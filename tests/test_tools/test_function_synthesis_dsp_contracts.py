# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native configured DSP proof contracts

"""Exercise all required real macro configurations and complete retained-proof custody."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import threading
from pathlib import Path
from typing import cast

import pytest

from tools.function_synthesis_dsp_contracts import (
    NativeDspContract,
    prove_dsp_contract,
    read_dsp_contract,
)
from tools.function_synthesis_toolchain import SynthesisToolchain
from tools.function_synthesis_word_graph import MacroConfiguration

ContractCase = tuple[MacroConfiguration, Path, SynthesisToolchain, NativeDspContract]


@pytest.fixture(
    scope="module",
    params=[
        ("0000101", "DIRECT", "DIRECT"),
        ("0010101", "DIRECT", "DIRECT"),
        ("0110101", "DIRECT", "DIRECT"),
        ("1010101", "CASCADE", "DIRECT"),
        ("1010101", "DIRECT", "CASCADE"),
    ],
)
def contract_case(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> ContractCase:
    """Prove every actual admitted arithmetic mode using the installed native DSP library."""
    configuration = cast(MacroConfiguration, request.param)
    root = tmp_path_factory.mktemp("real-native-dsp-contract")
    tools = SynthesisToolchain.inspect(root / "inspection")
    directory = root / "contract"
    contract = prove_dsp_contract(configuration, tools, directory, timeout=90)
    return configuration, directory, tools, contract


def _snapshot(directory: Path) -> dict[str, tuple[str, int]]:
    """Retain every nested artifact hash and modification timestamp."""
    snapshot: dict[str, tuple[str, int]] = {}
    for parent, _, names in os.walk(directory):
        for name in names:
            path = Path(parent) / name
            snapshot[path.relative_to(directory).as_posix()] = (
                hashlib.sha256(path.read_bytes()).hexdigest(),
                path.stat().st_mtime_ns,
            )
    return snapshot


def test_full_actual_domain_native_unsat_and_immutable_readback(
    contract_case: ContractCase,
) -> None:
    """Keep all192 operand inputs,26 independent controls and all144 outputs and the actual native whole-miter UNSAT."""
    configuration, directory, tools, contract = contract_case
    for name in ["reference.aig", "candidate.aig"]:
        with (directory / name).open("rb") as stream:
            fields = stream.readline().split()
        assert tuple(int(fields[index]) for index in [2, 3, 4]) == (218, 0, 144)
    assert "UNSATISFIABLE" in (directory / "equivalence.log").read_text()
    before = _snapshot(directory)
    assert read_dsp_contract(configuration, tools, directory) == contract
    assert _snapshot(directory) == before
    assert contract.to_dict()["configuration"] == list(configuration)
    assert contract.to_dict()["independent_input_bits"] == 218
    assert contract.to_dict()["independent_operand_bits"] == 192
    assert contract.to_dict()["independent_auxiliary_control_bits"] == 26


@pytest.mark.parametrize(
    "artifact",
    [
        "contract.v",
        "miter.v",
        "flow.ys",
        "synthesis.log",
        "reference.aig",
        "candidate.aig",
        "miter.aig",
        "reference-io.log",
        "candidate-io.log",
        "miter-io.log",
        "paired-word.json",
        "equivalence.command.json",
        "contract-record.json",
    ],
)
def test_changed_actual_contract_evidence_is_preserved_and_refused(
    contract_case: ContractCase, tmp_path: Path, artifact: str
) -> None:
    """Refuse damaged successful evidence while preserving the owner-controlled acquisition."""
    configuration, source, tools, _ = contract_case
    root = tmp_path / "changed"
    shutil.copytree(source, root)
    changed = root / artifact
    changed.write_bytes(changed.read_bytes() + b"\n")
    if artifact == "contract-record.json":
        record = json.loads(changed.read_text())
        record["producer_sha256"] = "0" * 64
        changed.write_text(json.dumps(record))
    before = _snapshot(root)
    with pytest.raises(ValueError):
        read_dsp_contract(configuration, tools, root)
    assert _snapshot(root) == before


@pytest.mark.parametrize("name", ["contract", "contract-record.json", "reference.aig"])
def test_linked_native_contract_cannot_claim_original_custody(
    contract_case: ContractCase, tmp_path: Path, name: str
) -> None:
    """Reject linked proof closures and artifacts even when actual target bytes are genuine."""
    configuration, source, tools, _ = contract_case
    root = tmp_path / "contract"
    shutil.copytree(source, root)
    path = root if name == "contract" else root / name
    target = tmp_path / "retained-target"
    path.rename(target)
    path.symlink_to(target, target_is_directory=target.is_dir())
    with pytest.raises(ValueError, match="linked"):
        read_dsp_contract(configuration, tools, root)
    assert path.is_symlink() and target.exists()


@pytest.mark.parametrize(
    "damage", ["absent", "nonobject", "nontext", "noncanonical", "offset", "malformed"]
)
def test_original_native_qualification_time_is_required(
    contract_case: ContractCase, tmp_path: Path, damage: str
) -> None:
    """Refuse missing or rewritten timestamp representations instead of restamping old results."""
    configuration, source, tools, _ = contract_case
    root = tmp_path / "timestamp"
    shutil.copytree(source, root)
    path = root / "contract-record.json"
    record = json.loads(path.read_text())
    if damage == "absent":
        del record["recorded_at"]
    elif damage == "nonobject":
        record = []
    elif damage == "nontext":
        record["recorded_at"] = 123
    elif damage == "noncanonical":
        record["recorded_at"] = record["recorded_at"].replace("+00:00", "Z")
    elif damage == "offset":
        record["recorded_at"] = record["recorded_at"].replace("+00:00", "+01:00")
    else:
        record["recorded_at"] = "not-a-timestamp"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        read_dsp_contract(configuration, tools, root)


@pytest.mark.parametrize("damage", ["missing", "sat", "conflict", "error"])
def test_inconclusive_or_conflicting_actual_native_verdict_is_refused(
    contract_case: ContractCase, tmp_path: Path, damage: str
) -> None:
    """Require the original decisive whole-miter UNSAT rather than a command exit code."""
    configuration, source, tools, _ = contract_case
    root = tmp_path / "verdict"
    shutil.copytree(source, root)
    path = root / "equivalence.log"
    text = path.read_text()
    if damage == "missing":
        text = "\n".join(line for line in text.splitlines() if "UNSATISFIABLE" not in line)
    elif damage == "sat":
        text = text.replace("UNSATISFIABLE", "SATISFIABLE")
    elif damage == "conflict":
        text += "SATISFIABLE\n"
    else:
        text += "ERROR: deliberately damaged actual native receipt\n"
    path.write_text(text)
    with pytest.raises(ValueError):
        read_dsp_contract(configuration, tools, root)


@pytest.mark.parametrize("seconds", [True, 0.0, -1.0, float("nan"), float("inf"), 91.0])
def test_invalid_native_deadline_does_not_open_an_acquisition(
    contract_case: ContractCase, tmp_path: Path, seconds: float
) -> None:
    """Preserve the actual ninety-second ceiling and require finite positive proof budgets."""
    configuration, _, tools, _ = contract_case
    root = tmp_path / "invalid"
    with pytest.raises(ValueError, match="finite deadline"):
        prove_dsp_contract(configuration, tools, root, timeout=seconds)
    assert not root.exists()


@pytest.mark.parametrize(
    "configuration",
    [
        ("0000000", "DIRECT", "DIRECT"),
        ("0000101", "UNKNOWN", "DIRECT"),
        ("0000101", "DIRECT", "UNKNOWN"),
    ],
)
def test_unknown_actual_arithmetic_configuration_is_not_qualified(
    contract_case: ContractCase, tmp_path: Path, configuration: MacroConfiguration
) -> None:
    """Refuse modes or selected operand sources with no complete arithmetic specification."""
    root = tmp_path / "unsupported"
    with pytest.raises(ValueError):
        prove_dsp_contract(configuration, contract_case[2], root, timeout=90)
    assert not root.exists()


def test_source_change_during_actual_native_proof_is_preserved_and_refused(
    contract_case: ContractCase, tmp_path: Path
) -> None:
    """Reject changed bound inputs after a real subprocess and successful native UNSAT run."""
    configuration, _, tools, _ = contract_case
    root = tmp_path / "changed-during-proof"
    changed = threading.Event()

    def change_acquisition(event: str, args: tuple[object, ...]) -> None:
        """Change this acquisition at its real native subprocess boundary after source binding."""
        if (
            event == "subprocess.Popen"
            and args[2] in (root, str(root), str(root).encode())
            and not changed.is_set()
        ):
            assert (root / "synthesis.command.json").is_file()
            with (root / "contract.v").open("a") as stream:
                stream.write("\n// actual acquisition changed at its native subprocess boundary\n")
            changed.set()

    change_acquisition.__dict__["__cantrace__"] = True
    sys.addaudithook(change_acquisition)
    with pytest.raises(ValueError, match="specification changed during proof"):
        prove_dsp_contract(configuration, tools, root, timeout=90)
    assert changed.is_set()
    assert "UNSATISFIABLE" in (root / "equivalence.log").read_text()
    assert not (root / "contract-record.json").exists()
