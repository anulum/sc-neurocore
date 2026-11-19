# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Retained compositional proof reader deadline

"""Charge final retained-artifact validation to the real inherited reader deadline."""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from tests.test_tools.test_function_synthesis_grouped_mapping import _snapshot
from tests.test_tools.test_function_synthesis_grouped_mapping import (
    grouped_measurement as grouped_measurement,
)
from tools.function_synthesis_proof import read_compositional_mapping_proof
from tools.function_synthesis_records import SynthesisCase
from tools.function_synthesis_toolchain import SynthesisToolchain
from tools.function_synthesis_yosys import FunctionSynthesisMeasurement


def test_final_readonly_artifact_validation_cannot_outlive_its_inherited_deadline(
    grouped_measurement: tuple[
        SynthesisCase, Path, SynthesisToolchain, FunctionSynthesisMeasurement
    ],
) -> None:
    """Refuse a real final read delay after complete reconstruction without changing the proof."""
    case, root, tools, measurement = grouped_measurement
    assert measurement.proof_method == "compositional"
    record = root / "composition-proof/proof-record.json"
    original = _snapshot(root)
    phase = "calibrate"
    reads = 0
    final_read = 0
    deadline = 0.0
    delayed = False

    def delay_final_record_read(event: str, args: tuple[object, ...]) -> None:
        """Delay only the observed final record read of this actual retained acquisition."""
        nonlocal reads, delayed
        if phase != "idle" and event == "open" and args[0] in (record, str(record)):
            reads += 1
            if phase == "expire" and reads == final_read:
                delayed = True
                time.sleep(max(0.0, deadline - time.monotonic()) + 0.01)

    delay_final_record_read.__dict__["__cantrace__"] = True
    sys.addaudithook(delay_final_record_read)
    started = time.monotonic()
    positive = read_compositional_mapping_proof(case, root, tools)
    elapsed = time.monotonic() - started
    final_read = reads
    assert final_read >= 2
    deadline = time.monotonic() + min(3300.0, max(5.0, elapsed * 4 + 5.0))
    bounded = replace(tools, cohort_deadline=deadline)
    reads = 0
    phase = "expire"
    try:
        with pytest.raises(
            ValueError, match="Compositional proof reader exhausted the shared case budget"
        ):
            read_compositional_mapping_proof(case, root, bounded)
    finally:
        phase = "idle"
    assert delayed and reads == final_read
    assert time.monotonic() > deadline
    assert _snapshot(root) == original
    assert read_compositional_mapping_proof(case, root, tools) == positive
    assert _snapshot(root) == original
