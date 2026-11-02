# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Executed formal workflow evidence-table contracts

"""Execute actual workflow gate and summary scripts on explicit table controls."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import textwrap

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/yosys-synth.yml"
HEADER = "job\tstatus\twall_seconds\tmax_rss_kib\tlocation\tnative_status\tcgroup_peak_bytes\n"


def _step(name: str) -> str:
    """Read an actual literal Bash step from the maintained workflow."""
    text = WORKFLOW.read_text().split("      - name: " + name + "\n", 1)[1]
    body = text.split("        run: |\n", 1)[1].split("\n      - ", 1)[0]
    return textwrap.dedent(body).replace("${{ matrix.shard }}", "0")


def _table(root: Path, rows: list[tuple[str, str, str]]) -> Path:
    """Create parser inputs only; no row is represented as collected proof evidence."""
    directory = root / "formal-timing"
    directory.mkdir()
    table = directory / "formal_timing_shard_0.tsv"
    table.write_text(
        HEADER
        + "".join(
            f"hdl/formal/control_{index}.sby\t{status}\t1\tNA\t{location}\t{native}\t1024\n"
            for index, (status, location, native) in enumerate(rows)
        )
    )
    return table


@pytest.mark.parametrize(
    "status,location,native,accepted",
    [
        ("PASS", "hosted", "PASS", True),
        ("ATTESTED", "ML350, attested", "PASS", True),
        ("UNPROVEN-RESOURCE", "ML350, attested", "UNPROVEN-RESOURCE", True),
        ("UNPROVEN-RESOURCE", "hosted", "UNPROVEN-RESOURCE", False),
        ("UNPROVEN-RESOURCE", "ML350, attested", "PASS", False),
        ("ATTESTED", "ML350, attested", "UNPROVEN-RESOURCE", False),
        ("PASS", "ML350, attested", "PASS", False),
        ("MEMORY", "hosted", "MEMORY", False),
        ("TIMEOUT", "ML350, attested", "TIMEOUT", False),
        ("ERROR", "external, refused", "NA", False),
        ("UNKNOWN", "hosted", "UNKNOWN", False),
    ],
)
def test_actual_workflow_gate_distinguishes_qualified_outcomes(
    tmp_path: Path, status: str, location: str, native: str, accepted: bool
) -> None:
    """Run the real workflow gate on supported and contradictory table tuples."""
    _table(tmp_path, [(status, location, native)])
    script = _step("Fail when any SymbiYosys row lacks qualifying evidence")
    result = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "RUNNER_TEMP": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert (result.returncode == 0) is accepted, result.stdout + result.stderr


def test_actual_summary_never_counts_unproven_as_a_native_proof(tmp_path: Path) -> None:
    """Render the maintained summary with separate real schema outcome categories."""
    table = _table(
        tmp_path,
        [
            ("PASS", "hosted", "PASS"),
            ("ATTESTED", "ML350, attested", "PASS"),
            ("UNPROVEN-RESOURCE", "ML350, attested", "UNPROVEN-RESOURCE"),
        ],
    )
    script = _step("Run and time this shard's SymbiYosys jobs")
    start = script.index('{\n  echo "### SymbiYosys timing')
    stop = script.index('} >> "$GITHUB_STEP_SUMMARY"', start) + len('} >> "$GITHUB_STEP_SUMMARY"')
    summary = tmp_path / "summary.md"
    env = {**os.environ, "table": str(table), "SHARD": "0", "GITHUB_STEP_SUMMARY": str(summary)}
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", script[start:stop]],
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    rendered = summary.read_text()
    assert "Native proofs: 2. **UNPROVEN-RESOURCE: 1.**" in rendered
    assert "are not proofs" in rendered
    assert "UNPROVEN-RESOURCE |" in rendered
