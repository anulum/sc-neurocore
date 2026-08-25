# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bertram benchmark evidence gate

from __future__ import annotations

import hashlib
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_RESULT = _ROOT / "benchmarks/results/bench_bertram_phantom.json"
_SOURCES = {
    "python": _ROOT / "src/sc_neurocore/neurons/models/bertram_phantom.py",
    "rust": _ROOT / "src/sc_neurocore/accel/rust/safety/bertram_phantom.rs",
    "julia": _ROOT / "src/sc_neurocore/accel/julia/neurons/bertram_phantom.jl",
    "go": _ROOT / "src/sc_neurocore/accel/go/services/bertram_phantom.go",
    "mojo": _ROOT / "src/sc_neurocore/accel/mojo/kernels/bertram_phantom.mojo",
}


def test_result_is_bound_to_all_five_runtime_sources() -> None:
    payload = json.loads(_RESULT.read_text(encoding="utf-8"))
    expected = {
        name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in _SOURCES.items()
    }
    assert payload["source_hashes"] == expected


def test_result_records_real_consistent_events_without_speed_claim() -> None:
    payload = json.loads(_RESULT.read_text(encoding="utf-8"))
    assert payload["steps"] == 10_000
    assert {entry["events"] for entry in payload["backend_summary"].values()} == {18}
    assert payload["max_native_state_error"] < 5e-9
    assert payload["hardware_measurement_claimed"] is False
    assert payload["production_speed_claim"] is False
