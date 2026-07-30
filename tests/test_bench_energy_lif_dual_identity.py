# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from types import ModuleType
import pytest
from benchmarks import bench_model_energy_lif, bench_model_sc_normalized_energy_lif

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    ("module", "artifact", "model", "events"),
    [
        (bench_model_energy_lif, "bench_energy_lif.json", "EnergyLIFNeuron", 17),
        (
            bench_model_sc_normalized_energy_lif,
            "bench_sc_normalized_energy_lif.json",
            "SCNormalizedEnergyLIFNeuron",
            153,
        ),
    ],
)
def test_committed_energy_lif_benchmark_is_bound_and_parity_clean(
    module: ModuleType, artifact: str, model: str, events: int
) -> None:
    payload = json.loads((ROOT / "benchmarks/results" / artifact).read_text())
    assert payload["model"] == model
    assert payload["production_speed_claim"] is False
    assert set(payload["backend_summary"]) == {"python", "rust", "julia", "go", "mojo"}
    for backend, row in payload["backend_summary"].items():
        assert row["trace_matches_python"] is True
        assert row["event_vector_matches_python"] is True
        assert row["events"] == events
        assert row["parity_max_abs_diff"] <= module.backends.PARITY_ATOL[backend]
    for relative, digest in payload["source_hashes"].items():
        if isinstance(digest, str):
            assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest
