# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bounded complete native GIA protocol qualification

"""Exercise real bounded fallback and complete defective-model refusal through public proof APIs."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from tests.test_tools.test_function_synthesis_gia_fallback import GiaCase, _snapshot
from tests.test_tools.test_function_synthesis_gia_fallback import arctan_case as arctan_case
from tests.test_tools.test_function_synthesis_gia_fallback import check_arctan_fallback
from tools.function_synthesis_gia import (
    check_gia_equivalent_verdict,
    prove_gia_equivalence,
    read_gia_equivalence,
)
from tools.function_synthesis_toolchain import SynthesisToolError


def test_actual_undecided_empty_reduction_is_not_an_equivalence_verdict() -> None:
    """Require a decisive proof after the real zero-reduction race at the native deadline."""
    root = Path(__file__).resolve().parents[1] / "fixtures/function_synthesis"
    log = root / "undecided_empty_reduction.rpt"
    provenance = json.loads((root / "undecided_empty_reduction.json").read_text())
    original = log.read_bytes()
    assert hashlib.sha256(original).hexdigest() == provenance["log_sha256"]
    assert provenance["command"] == "&read singleton.aig; &cec -m -T 1 -v"
    assert b"Network after reduction is empty.\n" in original
    assert b"Networks are UNDECIDED after the new CEC engine." in original
    assert b"Timed out after " not in original
    with pytest.raises(ValueError, match="Complete native GIA equivalence was not proved"):
        check_gia_equivalent_verdict(log)
    assert log.read_bytes() == original


@pytest.fixture(scope="module")
def paired_case(arctan_case: GiaCase, tmp_path_factory: pytest.TempPathFactory) -> GiaCase:
    """Require the real complete arctangent singleton to exhaust a smaller internal allowance."""
    source, tools, interface = arctan_case
    root = tmp_path_factory.mktemp("actual-bounded-paired-arctan32")
    for name in ("word.aig", "physical.aig"):
        shutil.copyfile(source / name, root / name)
    engine = prove_gia_equivalence(
        root / "word.aig",
        root / "physical.aig",
        interface,
        tools,
        root,
        timeout=90,
        singleton_seconds=1,
    )
    check_arctan_fallback(root, engine)
    return root, tools, interface


def test_smaller_singleton_allowance_always_exercises_the_complete_paired_proof(
    paired_case: GiaCase,
) -> None:
    """Fail rather than silently bypass the fallback when a host decides the bounded singleton."""
    root, tools, interface = paired_case
    before = _snapshot(root)
    engine = read_gia_equivalence(root / "word.aig", root / "physical.aig", interface, tools, root)
    record = json.loads((root / "gia-proof.json").read_text())
    check_arctan_fallback(root, engine)
    assert record["native_seconds"] == 90.0
    assert record["input_bits"] == len(interface.inputs)
    assert record["all_compared_output_bits"] == len(interface.outputs)
    assert "Timed out after " in (root / "equivalence.log").read_text()
    assert "Networks are UNDECIDED" in (root / "equivalence.log").read_text()
    singleton = json.loads((root / "equivalence.command.json").read_text())
    paired = json.loads((root / "paired-equivalence.command.json").read_text())
    assert singleton[-1] == "&read singleton.aig; &cec -m -T 1 -v"
    expected_pair = (
        "&read reference-plain.aig; &cec -T 60 -v candidate-plain.aig"
        if dict(tools.versions)["yosys"].startswith("Yosys 0.33 ")
        else "cec -T 60 -n -v reference-plain.aig candidate-plain.aig"
    )
    assert paired[-1] == expected_pair
    assert _snapshot(root) == before


@pytest.mark.parametrize("seconds", [True, 0, -1, 61])
def test_invalid_singleton_allowance_does_not_start_or_modify_a_proof(
    arctan_case: GiaCase, tmp_path: Path, seconds: int
) -> None:
    """Reject Boolean, nonpositive and larger-than-native allowances before creating artifacts."""
    root, tools, interface = arctan_case
    before = _snapshot(root)
    with pytest.raises(ValueError, match="singleton allowance"):
        prove_gia_equivalence(
            root / "word.aig",
            root / "physical.aig",
            interface,
            tools,
            tmp_path,
            timeout=90,
            singleton_seconds=seconds,
        )
    assert _snapshot(root) == before
    assert list(tmp_path.iterdir()) == []


def _faulty_candidate(source: Path, root: Path, rare: bool) -> None:
    """Change one complete real AIG output while retaining every original port and its order."""
    shutil.copyfile(source / "word.aig", root / "word.aig")
    header, body = (source / "physical.aig").read_bytes().split(b"\n", 1)
    fields = header.split()
    maximum, inputs, latches, count, gates = map(int, fields[1:])
    assert fields[0] == b"aig" and latches == 0 and maximum == inputs + gates
    fields = body.split(b"\n", count)
    outputs = [int(line) for line in fields[:-1]]
    encoded = fields[-1]
    cursor = 0
    for _ in range(2 * gates):
        while encoded[cursor] & 128:
            cursor += 1
        cursor += 1
    original_gates, symbols = encoded[:cursor], encoded[cursor:]
    assert f"o{count - 1} result[31]".encode() in symbols.splitlines()
    added = bytearray()

    def variable(value: int) -> None:
        """Append one unsigned AIGER delta without changing existing native gate bytes."""
        while value >= 128:
            added.append((value & 127) | 128)
            value >>= 7
        added.append(value)

    def conjunction(a: int, b: int) -> int:
        """Append one actual conjunction whose fanins precede its new AIG variable."""
        nonlocal maximum, gates
        maximum += 1
        gates += 1
        literal = 2 * maximum
        a, b = max(a, b), min(a, b)
        variable(literal - a)
        variable(a - b)
        return literal

    if rare:
        assert inputs >= 64
        pattern = [2 * (index + 1) ^ int(index % 3 != 0) for index in range(64)]
        trigger = pattern[0]
        for value in pattern[1:]:
            trigger = conjunction(trigger, value)
        both = conjunction(outputs[-1], trigger)
        neither = conjunction(outputs[-1] ^ 1, trigger ^ 1)
        outputs[-1] = conjunction(both ^ 1, neither ^ 1)
    else:
        outputs[-1] ^= 1
    header = f"aig {maximum} {inputs} 0 {count} {gates}\n".encode()
    (root / "physical.aig").write_bytes(
        header
        + b"".join(f"{literal}\n".encode() for literal in outputs)
        + original_gates
        + added
        + symbols
    )


@pytest.mark.parametrize("rare", [False, True])
def test_actual_non_equivalent_complete_models_never_receive_a_positive_record(
    paired_case: GiaCase, tmp_path: Path, rare: bool
) -> None:
    """Refuse both an immediately visible real fault and a rare fault after bounded GIA admission."""
    source, tools, interface = paired_case
    _faulty_candidate(source, tmp_path, rare)
    before = {name: (tmp_path / name).read_bytes() for name in ("word.aig", "physical.aig")}
    with pytest.raises((ValueError, SynthesisToolError)):
        prove_gia_equivalence(
            tmp_path / "word.aig",
            tmp_path / "physical.aig",
            interface,
            tools,
            tmp_path,
            timeout=90,
            singleton_seconds=1,
        )
    assert not (tmp_path / "gia-proof.json").exists()
    singleton = (tmp_path / "equivalence.log").read_text()
    if rare:
        assert "Networks are UNDECIDED" in singleton and "Timed out after " in singleton
        paired = (tmp_path / "paired-equivalence.log").read_text()
        assert "Networks are NOT EQUIVALENT" in paired or "Networks are UNDECIDED" in paired
    else:
        assert "Networks are NOT EQUIVALENT" in singleton
        assert not (tmp_path / "paired-equivalence.log").exists()
    assert before == {name: (tmp_path / name).read_bytes() for name in before}


@pytest.mark.parametrize(
    "damage",
    [
        "negative",
        "duplicate",
        "error",
        "timeout-marker",
        "command",
        "geometry",
        "engine",
        "engine-type",
        "singleton-seconds-type",
        "singleton-seconds-range",
        "singleton-seconds-command",
        "link",
    ],
)
def test_changed_complete_arctan_strategy_is_preserved_and_refused(
    paired_case: GiaCase, tmp_path: Path, damage: str
) -> None:
    """Reject damaged actual solver receipts, command scope and selected-strategy provenance."""
    source, tools, interface = paired_case
    root = tmp_path / "changed-arctan"
    shutil.copytree(source, root)
    record_path = root / "gia-proof.json"
    record = json.loads(record_path.read_text())
    ivy = record["engine"] == "ivy-complete-all-output-or-after-undecided-gia"
    if record["engine"] == "complete-native-output-comparisons":
        partition = root / "native-output-comparisons"
        logs = record["output_partition"]["artifact_sha256"]
        path = next(
            partition / name
            for name in logs
            if name.endswith("/equivalence.log")
            and any(
                line.startswith("Networks are equivalent")
                for line in (partition / name).read_text().splitlines()
            )
        )
        assert path.with_suffix(".command.json").relative_to(partition).as_posix() in logs
    else:
        selected = "ivy-equivalence" if ivy else "paired-equivalence"
        path = root / (selected + ".log")
    if damage == "negative":
        positive, negative = (
            ("UNSATISFIABLE", "SATISFIABLE (output = 0)")
            if ivy
            else ("Networks are equivalent", "Networks are NOT EQUIVALENT")
        )
        transcript = path.read_text()
        assert positive in transcript
        path.write_text(transcript.replace(positive, negative))
    elif damage == "duplicate":
        contradiction = (
            "UNDECIDED      Time =     0.00 sec\n" if ivy else "Networks are UNDECIDED.\n"
        )
        path.write_text(path.read_text() + contradiction)
    elif damage == "error":
        path.write_text(path.read_text() + "ERROR: actual retained proof damaged\n")
    elif damage == "timeout-marker":
        path = root / "equivalence.log"
        path.write_text(
            "\n".join(
                line
                for line in path.read_text().splitlines()
                if not line.startswith(("Timed out after ", "Networks are equivalent"))
            )
        )
    elif damage == "command":
        path = path.with_suffix(".command.json")
        argv = json.loads(path.read_text())
        argv[-1] = "cec -T 60 -n -v reference-plain.aig reference-plain.aig"
        path.write_text(json.dumps(argv))
    elif damage == "geometry":
        record["all_compared_output_bits"] -= 1
    elif damage == "engine":
        record["engine"] = "unqualified-equivalence"
    elif damage == "engine-type":
        record["engine"] = False
    elif damage == "singleton-seconds-type":
        record["singleton_seconds"] = True
    elif damage == "singleton-seconds-range":
        record["singleton_seconds"] = 61
    elif damage == "singleton-seconds-command":
        record["singleton_seconds"] = 2
    else:
        original = tmp_path / "original-native.log"
        path.rename(original)
        path.symlink_to(original)
    record_path.write_text(json.dumps(record))
    before = _snapshot(tmp_path)
    with pytest.raises(ValueError):
        read_gia_equivalence(root / "word.aig", root / "physical.aig", interface, tools, root)
    assert _snapshot(tmp_path) == before
