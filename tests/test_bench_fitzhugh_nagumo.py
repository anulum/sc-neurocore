# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — FitzHugh-Nagumo benchmark and receipt evidence gates

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

from benchmarks import bench_fitzhugh_nagumo_simulate as benchmark
from sc_neurocore.neurons.models import FitzHughNagumoNeuron

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "benchmarks/results/bench_fitzhugh_nagumo_simulate.json"
RECEIPT = ROOT / "src/sc_neurocore/neurons/reference_receipts/fitzhugh_nagumo_1961.json"


def test_committed_benchmark_is_source_bound_and_five_runtime() -> None:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "sc-neurocore.polyglot-benchmark.v1"
    assert payload["model"] == "FitzHughNagumoNeuron"
    assert payload["evidence_class"] == "local_regression_non_isolated"
    assert payload["production_speed_claim"] is False
    assert payload["hardware_measurement_claimed"] is False
    assert set(payload["measured_order"]) == set(benchmark.BACKENDS)
    for backend in benchmark.BACKENDS:
        row = payload["backends"][backend]
        assert row["available"] is True and row["used"] is True
        assert row["event_count_matches_python"] is True
        assert row["final_state_matches_python"] is True
        assert row["parity_max_abs_diff"] <= benchmark.PARITY_ATOL[backend]
    expected = {
        source: hashlib.sha256((ROOT / source).read_bytes()).hexdigest()
        for source in benchmark.SOURCES
    }
    assert payload["source_hashes"] == expected


def test_reference_receipt_replays_bitwise() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    neuron = FitzHughNagumoNeuron()
    digest = hashlib.sha256()
    event_indices: list[int] = []
    for index in range(receipt["oracle"]["steps"]):
        event = neuron.step(receipt["drive"]["current"])
        if event:
            event_indices.append(index)
        digest.update(struct.pack("<ddq", neuron.v, neuron.w, event))
    assert event_indices == receipt["oracle"]["event_indices"]
    assert len(event_indices) == receipt["oracle"]["events"]
    assert [neuron.v, neuron.w] == receipt["oracle"]["final_state"]
    assert digest.hexdigest() == receipt["oracle"]["trace_sha256"]
