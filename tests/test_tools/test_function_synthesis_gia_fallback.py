# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Isolated complete native GIA fallback qualification

"""Keep the large real exp qualification independent of fast native proof-contract tests."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from tools.function_synthesis_aiger import AigerInterface, check_equivalent_verdict
from tools.function_synthesis_gia import (
    check_gia_equivalent_verdict,
    prove_gia_equivalence,
    read_gia_equivalence,
)
from tools.function_synthesis_netlist import Netlist
from tools.function_synthesis_records import SynthesisCase
from tools.function_synthesis_toolchain import SynthesisToolchain
from tools.function_synthesis_yosys import run_function_synthesis

GiaCase = tuple[Path, SynthesisToolchain, AigerInterface]


@pytest.fixture(scope="module")
def fallback_case(tmp_path_factory: pytest.TempPathFactory) -> GiaCase:
    """Acquire the real exp residual whose complete native GIA proof uses both checker stages."""
    root = tmp_path_factory.mktemp("actual-gia-exp-fallback")
    tools = SynthesisToolchain.inspect(root / "inspection")
    case = SynthesisCase("exp", 8, 4)
    result = run_function_synthesis(case, root / "measurement", tools)
    assert result.resources.dsp_blocks == 17
    residual = root / "measurement/composition-proof/primitive-proof"
    record = json.loads((residual / "gia-proof.json").read_text())
    interface = AigerInterface(frozenset(record["input_order"]), frozenset(record["output_order"]))
    physical = Netlist.read(root / "measurement/mapped.json")
    primitives = [cell for cell in physical.cells.values() if cell.kind == "DSP48E1"]
    primary = set(physical.primary_inputs({"argument0": case.data_width}, case.data_width))
    primitive_outputs = (
        {
            bit
            for cell in primitives
            for port, direction in cell.directions.items()
            if direction == "output"
            for bit in cell.connections[port]
        }
        - primary
        - {"0", "1", "x", "z"}
    )
    primitive_input_bits = sum(
        len(cell.connections[port])
        for cell in primitives
        for port, direction in cell.directions.items()
        if direction == "input"
    )
    assert len(interface.inputs) == case.input_bits + len(primitive_outputs)
    assert len(interface.outputs) == case.data_width + primitive_input_bits
    proof = root / "default-native-proof"
    proof.mkdir()
    for name in ("word.aig", "physical.aig"):
        os.link(residual / name, proof / name)
    engine = prove_gia_equivalence(
        proof / "word.aig",
        proof / "physical.aig",
        interface,
        tools,
        proof,
        timeout=90,
    )
    assert engine == "gia-complete-all-output-or"
    assert "Calling the old CEC engine." in (proof / "equivalence.log").read_text()
    check_gia_equivalent_verdict(proof / "equivalence.log")
    return proof, tools, interface


@pytest.mark.parametrize(
    "damage", ["final", "fallback", "zero-evidence", "duplicate", "negative", "error"]
)
def test_incomplete_or_contradictory_actual_fallback_transcript_is_refused(
    fallback_case: GiaCase, tmp_path: Path, damage: str
) -> None:
    """Reject damaged actual two-engine transcripts without inventing another solver result."""
    root, _, _ = fallback_case
    text = (root / "equivalence.log").read_text()
    if damage == "final":
        text = "\n".join(
            line for line in text.splitlines() if not line.startswith("Networks are equivalent")
        )
    elif damage == "fallback":
        text = text.replace("Calling the old CEC engine.", "")
    elif damage == "zero-evidence":
        text = "\n".join(
            line
            for line in text.splitlines()
            if not line.startswith(("Network after reduction is empty.", "NBeg ="))
        )
    elif damage == "duplicate":
        text += "Networks are equivalent.  Time = 0.00 sec\n"
    elif damage == "negative":
        text = text.replace("Networks are UNDECIDED", "Networks are NOT EQUIVALENT")
    else:
        text += "ERROR: actual retained transcript deliberately damaged\n"
    path = tmp_path / "changed-verdict.log"
    path.write_text(text)
    with pytest.raises(ValueError):
        check_gia_equivalent_verdict(path)


def _snapshot(root: Path) -> dict[str, tuple[int, int, bytes]]:
    """Retain modes, timestamps, file bytes and link targets without traversing native directories as files."""
    metadata = root.lstat()
    entries = {".": (metadata.st_mode, metadata.st_mtime_ns, b"")}
    for directory, subdirectories, files in os.walk(root, followlinks=False):
        for name in (*subdirectories, *files):
            path = Path(directory) / name
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                content = os.fsencode(os.readlink(path))
            elif stat.S_ISREG(metadata.st_mode):
                content = path.read_bytes()
            else:
                content = b""
            entries[path.relative_to(root).as_posix()] = (
                metadata.st_mode,
                metadata.st_mtime_ns,
                content,
            )
    return entries


@pytest.fixture(scope="module")
def arctan_case(tmp_path_factory: pytest.TempPathFactory) -> GiaCase:
    """Acquire complete 32-bit arctangent with the original 900/90 native correctness profile."""
    root = tmp_path_factory.mktemp("actual-gia-arctan32")
    tools = SynthesisToolchain.inspect(root / "inspection")
    result = run_function_synthesis(
        SynthesisCase("arctan", 32, 16),
        root / "measurement",
        tools,
        synthesis_timeout=900,
        proof_timeout=90,
    )
    assert result.resources.dsp_blocks > 0
    proof = root / "measurement/composition-proof/primitive-proof"
    record = json.loads((proof / "gia-proof.json").read_text())
    interface = AigerInterface(frozenset(record["input_order"]), frozenset(record["output_order"]))
    assert len(interface.inputs) >= 32 and len(interface.outputs) >= 32
    return proof, tools, interface


def check_arctan_fallback(root: Path, engine: str) -> None:
    """Check the complete successful fallback transcript and corresponding proof record."""
    record = json.loads((root / "gia-proof.json").read_text())
    assert engine == record["engine"]
    assert record["singleton_seconds"] == 1
    assert "Networks are UNDECIDED" in (root / "equivalence.log").read_text()
    paired = (root / "paired-equivalence.log").read_text()
    files = {"paired-equivalence.log", "paired-equivalence.command.json"}
    if engine in {
        "ivy-complete-all-output-or-after-undecided-gia",
        "complete-native-output-comparisons",
    }:
        command = json.loads((root / "paired-equivalence.command.json").read_text())
        assert command[4:7] == ["--signal=KILL", "--verbose", "90.0"]
        deadline = Path(command[3])
        assert deadline.name == "timeout"
        watchdog = f"sending signal KILL to command '{command[7]}'"
        assert "Networks are UNDECIDED" in paired or paired.rstrip().splitlines()[-1:] in (
            [f"{deadline}: {watchdog}"],
            [f"{deadline.name}: {watchdog}"],
        )
    if engine == "ivy-complete-all-output-or-after-undecided-gia":
        assert record["schema_version"] == "sc-neurocore.complete-gia-equivalence.v4"
        assert "UNSATISFIABLE" in (root / "ivy-equivalence.log").read_text()
        ivy = json.loads((root / "ivy-equivalence.command.json").read_text())
        assert ivy[4:7] == ["--signal=KILL", "--verbose", "60.0"]
        assert ivy[-1] == "read_aiger singleton.aig; iprove -N 5 -v"
        files.update(("ivy-equivalence.log", "ivy-equivalence.command.json"))
    elif engine == "complete-native-output-comparisons":
        assert record["schema_version"] == "sc-neurocore.complete-gia-equivalence.v5"
        partition = record["output_partition"]
        assert partition["input_bits"] == record["input_bits"]
        assert partition["output_bits"] == record["all_compared_output_bits"]
        assert partition["native_comparisons"] > 0
        assert partition["representative_original_output_indices"]
        files.update(("ivy-equivalence.log", "ivy-equivalence.command.json"))
    else:
        assert engine == "complete-paired-output-cec-after-undecided-gia"
        assert record["schema_version"] == "sc-neurocore.complete-gia-equivalence.v3"
        check_equivalent_verdict(root / "paired-equivalence.log")
        assert not (root / "ivy-equivalence.log").exists()
    assert files <= set(record["artifact_sha256"])


def test_complete_arctan_native_strategy_is_retained_without_restamping(
    arctan_case: GiaCase,
) -> None:
    """Read the actual successful full-domain strategy with every original artifact preserved."""
    root, tools, interface = arctan_case
    before = _snapshot(root)
    engine = read_gia_equivalence(root / "word.aig", root / "physical.aig", interface, tools, root)
    check_arctan_fallback(root, engine)
    assert _snapshot(root) == before
