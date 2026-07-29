# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Validate the committed Compte benchmark's source and parity bindings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmarks.bench_compte_wm import BACKENDS, KERNEL, SOURCE_PATHS

_ROOT = Path(__file__).resolve().parents[1]
_RESULT = _ROOT / "benchmarks/results/bench_compte_wm.json"


def test_committed_benchmark_binds_current_sources_and_all_lanes() -> None:
    payload = json.loads(_RESULT.read_text(encoding="utf-8"))
    assert payload["kernel"] == KERNEL
    assert payload["model"] == "CompteWMNeuron"
    assert payload["production_speed_claim"] is False
    assert payload["network_behavior_claimed"] is False
    assert payload["passed"] is True
    assert set(payload["results"]) == set(BACKENDS)
    for backend in BACKENDS:
        row = payload["results"][backend]
        assert row["available"] is True and row["used"] is True
        assert row["trace_matches_python"] is True and row["events_exact"] is True
        assert row["event_count"] > 0
    assert set(payload["source_hashes"]) == set(SOURCE_PATHS)
    for relative, expected in payload["source_hashes"].items():
        actual = hashlib.sha256((_ROOT / relative).read_bytes()).hexdigest()
        assert actual == expected
    assert payload["rust_safety"]["passed"] is True


def test_benchmark_parity_gaps_respect_declared_tolerances() -> None:
    payload = json.loads(_RESULT.read_text(encoding="utf-8"))
    for row in payload["results"].values():
        assert row["parity_max_abs_diff"] <= row["parity_tolerance"]
