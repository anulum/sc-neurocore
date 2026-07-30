# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Model 50 dual-identity benchmark evidence contracts

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import ModuleType

import pytest

from benchmarks import (
    bench_model_non_resetting_lif,
    bench_model_sc_non_resetting_adaptive_lif,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("module", "artifact", "model", "events"),
    [
        (
            bench_model_non_resetting_lif,
            "bench_non_resetting_lif.json",
            "NonResettingLIFNeuron",
            4,
        ),
        (
            bench_model_sc_non_resetting_adaptive_lif,
            "bench_sc_non_resetting_adaptive_lif.json",
            "SCNonResettingAdaptiveLIFNeuron",
            577,
        ),
    ],
)
def test_committed_benchmark_is_source_bound_and_parity_clean(
    module: ModuleType, artifact: str, model: str, events: int
) -> None:
    payload = json.loads((ROOT / "benchmarks/results" / artifact).read_text(encoding="utf-8"))
    assert payload["schema_version"] == "sc-neurocore.polyglot-benchmark.v1"
    assert payload["model"] == model
    assert payload["production_speed_claim"] is False
    assert payload["hardware_measurement_claimed"] is False
    assert set(payload["backend_summary"]) == {"python", "rust", "julia", "go", "mojo"}
    for backend, row in payload["backend_summary"].items():
        assert row["available"] is True
        assert row["trace_matches_python"] is True
        assert row["event_vector_matches_python"] is True
        assert row["events"] == row["spikes"] == events
        assert row["parity_max_abs_diff"] <= module.backends.PARITY_ATOL[backend]
        assert row["median_ns_per_step"] > 0.0
    for relative, digest in payload["source_hashes"].items():
        if isinstance(digest, str):
            assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest
    for record in payload["binary_hashes"].values():
        assert len(record["sha256"]) == 64
        assert record["size_bytes"] > 0
        assert record["path"]
