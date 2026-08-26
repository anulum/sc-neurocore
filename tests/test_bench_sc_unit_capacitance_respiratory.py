# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — retained unit-capacitance respiratory benchmark evidence gate

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmarks.bench_model_sc_unit_capacitance_respiratory import SOURCE_HASH_PATHS

_ROOT = Path(__file__).resolve().parents[1]
_RESULT = _ROOT / "benchmarks/results/bench_sc_unit_capacitance_respiratory.json"


def test_sc_benchmark_is_source_bound_and_complete() -> None:
    payload = json.loads(_RESULT.read_text(encoding="utf-8"))
    expected = {
        source: hashlib.sha256(path.read_bytes()).hexdigest()
        for source, path in SOURCE_HASH_PATHS.items()
    }
    assert payload["source_hashes"] == expected
    assert payload["steps"] == 20_000
    assert payload["repeats"] == 5
    assert payload["production_speed_claim"] is False
    assert payload["hardware_measurement_claimed"] is False
    results = payload["results"]
    assert {result["backend"] for result in results} == {"python", "rust", "go", "julia", "mojo"}
    counts = {result["backend"]: result["spikes"] for result in results}
    assert counts["python"] == counts["rust"] == counts["go"] == counts["julia"] == 5
    assert counts["mojo"] in (4, 5)
    assert max(counts.values()) - min(counts.values()) <= 1
