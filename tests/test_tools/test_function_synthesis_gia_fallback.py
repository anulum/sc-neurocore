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
from pathlib import Path

import pytest

from tools.function_synthesis_aiger import AigerInterface
from tools.function_synthesis_gia import check_gia_equivalent_verdict
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
    proof = root / "measurement/composition-proof/primitive-proof"
    record = json.loads((proof / "gia-proof.json").read_text())
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
    assert "Calling the old CEC engine." in (proof / "equivalence.log").read_text()
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
