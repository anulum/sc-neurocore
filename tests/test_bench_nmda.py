# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NMDA benchmark evidence gate

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "benchmarks/results/bench_nmda.json"


def test_benchmark_is_source_bound_and_five_runtime() -> None:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    assert set(payload["backends"]) == {"python", "rust", "go", "julia", "mojo"}
    assert (payload["steps"], payload["repeats"]) == (20_000, 3)
    assert (payload["source_current"], payload["sc_current"]) == (0.6, 5.0)
    for name, digest in payload["source_hashes"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    for result in payload["backends"].values():
        assert result["source_median_ns_per_step"] > 0
        assert result["sc_median_ns_per_step"] > 0
    assert payload["production_speed_claim"] is False
    assert payload["hardware_measurement_claimed"] is False
