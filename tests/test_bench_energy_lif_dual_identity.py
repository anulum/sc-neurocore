# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Committed energy-LIF benchmark identity contracts
"""Bind the committed energy-LIF benchmark records to their sources and parity."""

from __future__ import annotations
import json
from pathlib import Path
from types import ModuleType
import pytest
from benchmarks import bench_model_energy_lif, bench_model_sc_normalized_energy_lif

from tools.benchmark_evidence_gate import committed_source_hash_failures

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    ("module", "artifact", "model", "events"),
    [
        (bench_model_energy_lif, "bench_energy_lif.json", "EnergyLIFNeuron", 17),
        (
            bench_model_sc_normalized_energy_lif,
            "bench_sc_normalized_energy_lif.json",
            "SCNormalizedEnergyLIFNeuron",
            1500,
        ),
    ],
)
def test_committed_energy_lif_benchmark_is_bound_and_parity_clean(
    module: ModuleType, artifact: str, model: str, events: int
) -> None:
    """Every backend of a committed record matches Python and its measured sources.

    Parameters
    ----------
    module : ModuleType
        Benchmark module that owns the parity tolerances.
    artifact : str
        Committed result file under ``benchmarks/results``.
    model : str
        Model identity the record must name.
    events : int
        Event count every backend must report at the committed step count.
    """
    payload = json.loads((ROOT / "benchmarks/results" / artifact).read_text())
    assert payload["model"] == model
    assert payload["production_speed_claim"] is False
    assert set(payload["backend_summary"]) == {"python", "rust", "julia", "go", "mojo"}
    for backend, row in payload["backend_summary"].items():
        assert row["trace_matches_python"] is True
        assert row["event_vector_matches_python"] is True
        assert row["events"] == events
        assert row["parity_max_abs_diff"] <= module.backends.PARITY_ATOL[backend]
    assert not committed_source_hash_failures(
        ROOT / "benchmarks/results" / artifact, repo_root=ROOT
    )
