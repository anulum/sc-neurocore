# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Photonic crosstalk benchmark evidence contract

"""Validate the tracked local benchmark schema, source hashes, and claims."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_REPOSITORY = Path(__file__).resolve().parents[1]
_RESULT = _REPOSITORY / "benchmarks/results/local_python_2026-07-14_photonic_crosstalk.json"


def test_photonic_benchmark_is_current_honest_and_source_hashed() -> None:
    """Reject stale source evidence or promotion claims from a non-isolated host."""
    payload: dict[str, Any] = json.loads(_RESULT.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["benchmark"] == "photonic_crosstalk_pairs"
    assert payload["evidence_class"] == "local_regression_non_isolated"
    assert payload["promotion_eligible"] is False
    assert payload["workload"]["pair_count"] == 4096
    assert payload["host"]["isolated_cpus"] == ""
    assert "not a universal speed claim" in payload["interpretation"]

    for relative_path, expected_hash in payload["source_hashes"].items():
        source = _REPOSITORY / relative_path
        assert source.is_file(), relative_path
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected_hash

    for runtime in ("python", "rust", "go", "julia", "mojo"):
        assert payload["runtimes"][runtime]["median_ns_per_batch"] > 0.0
    for runtime, error in payload["max_absolute_first_pair_error"].items():
        assert error <= payload["parity_envelope"][runtime]
