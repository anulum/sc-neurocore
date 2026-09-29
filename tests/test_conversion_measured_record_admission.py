# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual comparison record input admission

"""Refuse corrupt external comparison metadata and raw timing evidence through public replay."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_conversion_measured_order import (
    comparison,
    environment,
    go_library,
    mojo_library,
    rust_library,
)

__all__ = ["comparison", "environment", "go_library", "mojo_library", "rust_library"]

_PROBE = r"""
import sys
measurement = None
if sys.argv[1]:
    import coverage
    measurement = coverage.Coverage(
        data_file=sys.argv[1], branch=True, source=[sys.argv[2]], config_file=False
    )
    measurement.start()
from sc_neurocore.conversion import ConvertedSNN
model = ConvertedSNN([[[1.0]]], [None], [1.0], T=1)
try:
    model.run([1.0])
except RuntimeError as error:
    assert "dense IF comparison cannot order this runtime" in str(error), str(error)
else:
    raise AssertionError("corrupt comparison admitted")
assert model.run([1.0], backend="numpy").tolist() == [1.0]
assert "torch" not in sys.modules
if measurement is not None:
    measurement.stop()
    measurement.save()
"""


@pytest.mark.parametrize(
    "field,value",
    [
        ("document", []),
        ("schema_version", "other"),
        ("kernel", "other"),
        ("meta.python", "other"),
        ("meta.cpu", None),
        ("meta.cpu", ""),
        ("meta.numpy", "other"),
        ("meta.juliacall", "other"),
        ("meta.samples", True),
        ("meta.samples", 2),
        ("meta.warmup", 0),
        ("backends.numpy.cases", []),
        ("backends.numpy.cases.0.name", ""),
        ("backends.numpy.cases.0.input_sha256", "short"),
        ("backends.numpy.cases.0.response_sha256", "z" * 64),
        ("backends.go.available", False),
        ("backends.go.used", False),
        ("backends.go.full_bit_parity", False),
        ("backends.go.cases.0.samples_ns", [0, 0, 0]),
        ("backends.go.cases.0.first_call_ns", True),
    ],
)
def test_corrupt_measurement_evidence_cannot_select_a_runtime(
    comparison: Path,
    environment: dict[str, str],
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    """Edit a real comparison input and exercise its public refusal in an owned child."""
    record = json.loads(comparison.read_text())
    if field == "document":
        record = value
    else:
        target = record
        parts = field.split(".")
        for part in parts[:-1]:
            target = target[int(part)] if isinstance(target, list) else target[part]
        target[parts[-1]] = value
    path = tmp_path / "corrupt.json"
    path.write_text(json.dumps(record))
    settings = dict(environment, SC_NEUROCORE_IF_BENCHMARK=str(path))
    parent_data = os.environ.get("COVERAGE_FILE", "")
    child_data = f"{parent_data}-{field}" if parent_data else ""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _PROBE,
            child_data,
            str(Path(__file__).resolve().parents[1] / "src/sc_neurocore/conversion"),
        ],
        env=settings,
        capture_output=True,
        text=True,
        timeout=120,
    )
    (tmp_path / "stderr.txt").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
