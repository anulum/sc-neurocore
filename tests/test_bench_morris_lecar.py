# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Morris-Lecar benchmark and receipt evidence gates

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

from benchmarks import bench_model_morris_lecar as benchmark
from sc_neurocore.neurons.models import MorrisLecarNeuron

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "benchmarks/results/local_python_2026-06-17_morris_lecar_rk4.json"
RECEIPT = ROOT / "src/sc_neurocore/neurons/reference_receipts/morris_lecar_1981.json"
BACKENDS = {"python", "rust", "go", "julia", "mojo"}


def test_committed_benchmark_is_source_bound_and_five_runtime() -> None:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    assert payload["evidence_class"] == "local_regression_non_isolated"
    assert payload["production_speed_claim"] is False
    assert payload["hardware_measurement_claimed"] is False
    assert set(payload["backend_summary"]) == BACKENDS
    assert {row["backend"] for row in payload["results"]} == BACKENDS
    assert all(not row.get("skipped", False) for row in payload["results"])
    for row in payload["results"]:
        counts = row.get("spike_counts", [row.get("spikes")])
        assert set(counts) == {476}
    for source, path in benchmark.SOURCE_HASH_PATHS.items():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert payload["source_hashes"][source] == digest


def test_reference_receipt_replays_bitwise() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    neuron = MorrisLecarNeuron()
    digest = hashlib.sha256()
    event_indices: list[int] = []
    current = receipt["drive"]["current"]
    for index in range(receipt["oracle"]["steps"]):
        event = neuron.step(current)
        if event:
            event_indices.append(index)
        digest.update(struct.pack("<ddq", neuron.v, neuron.w, event))
    assert event_indices == receipt["oracle"]["event_indices"]
    assert len(event_indices) == receipt["oracle"]["events"]
    assert [neuron.v, neuron.w] == receipt["oracle"]["final_state"]
    assert digest.hexdigest() == receipt["oracle"]["trace_sha256"]
