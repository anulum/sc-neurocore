# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Larter-Breakspear benchmark evidence gate

import json
from pathlib import Path

from tools.benchmark_evidence_gate import committed_source_hash_failures

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "benchmarks/results/bench_larter_breakspear.json"


def test_benchmark_is_source_bound_and_five_runtime() -> None:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "sc-neurocore.polyglot-benchmark.v1"
    assert payload["benchmark"] == "Larter-Breakspear source and retained SC dual-identity RK4"
    assert payload["models"] == [
        "LarterBreakspearNeuron",
        "SCDecoupledAdaptationIonMassNeuron",
    ]
    assert payload["evidence_class"] == "local_regression_non_isolated"
    assert set(payload["backends"]) == {"python", "rust", "go", "julia", "mojo"}
    assert (payload["steps"], payload["repeats"], payload["coupling"]) == (20_000, 3, 0.0)
    assert not committed_source_hash_failures(RESULT, repo_root=ROOT)
    for result in payload["backends"].values():
        assert result["source_median_ns_per_step"] > 0
        assert result["sc_median_ns_per_step"] > 0
    assert payload["production_speed_claim"] is False
    assert payload["hardware_measurement_claimed"] is False
