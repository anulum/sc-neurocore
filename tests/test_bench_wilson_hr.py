# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Wilson-HR benchmark evidence gate

"""Committed source-bound five-runtime Wilson-HR benchmark contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "benchmarks/results/bench_wilson_hr_simulate.json"
SC_RESULT = ROOT / "benchmarks/results/bench_sc_resetting_wilson_hr_simulate.json"


def test_benchmark_is_source_bound_and_five_runtime() -> None:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "sc-neurocore.polyglot-benchmark.v1"
    assert payload["benchmark"] == "wilson_hr_simulate_rk4"
    assert payload["model"] == "WilsonHRNeuron"
    assert payload["evidence_class"] == "local_regression_non_isolated"
    assert payload["workload"] == {"n_steps": 2_000_000, "current": 0.1, "repeats": 5}
    assert set(payload["backends"]) == {"python", "rust", "julia", "go", "mojo"}
    for name, digest in payload["source_hashes"].items():
        if isinstance(digest, str):
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    for name, row in payload["backends"].items():
        assert row["median_ms"] > 0.0, name
        assert row["min_ms"] > 0.0, name
        assert row["speedup_vs_python"] > 0.0, name
    assert payload["backends"]["python"]["parity_max_abs_diff"] == 0.0
    assert payload["backends"]["rust"]["parity_max_abs_diff"] == 0.0
    assert payload["backends"]["julia"]["parity_max_abs_diff"] == 0.0
    assert payload["backends"]["go"]["parity_max_abs_diff"] == 0.0
    assert payload["backends"]["mojo"]["parity_max_abs_diff"] <= 1.0e-9
    assert len({row["event_count"] for row in payload["backends"].values()}) == 1
    assert payload["production_speed_claim"] is False
    assert payload["hardware_measurement_claimed"] is False


def test_retained_sc_benchmark_is_source_bound_and_five_runtime() -> None:
    payload = json.loads(SC_RESULT.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "sc-neurocore.polyglot-benchmark.v1"
    assert payload["benchmark"] == "sc_resetting_wilson_hr_simulate_rk4"
    assert payload["model"] == "SCResettingWilsonHRNeuron"
    assert payload["workload"] == {"n_steps": 2_000_000, "current": 2.0, "repeats": 5}
    assert set(payload["backends"]) == {"python", "rust", "julia", "go", "mojo"}
    for name, digest in payload["source_hashes"].items():
        if isinstance(digest, str):
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    for name, row in payload["backends"].items():
        assert row["median_ms"] > 0.0, name
        assert row["min_ms"] > 0.0, name
        assert row["speedup_vs_python"] > 0.0, name
    assert payload["backends"]["python"]["parity_max_abs_diff"] == 0.0
    assert payload["backends"]["rust"]["parity_max_abs_diff"] == 0.0
    assert payload["backends"]["julia"]["parity_max_abs_diff"] == 0.0
    assert payload["backends"]["go"]["parity_max_abs_diff"] == 0.0
    assert payload["backends"]["mojo"]["parity_max_abs_diff"] <= 2.5e-12
    assert len({row["event_count"] for row in payload["backends"].values()}) == 1
    assert payload["production_speed_claim"] is False
    assert payload["hardware_measurement_claimed"] is False
