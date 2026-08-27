# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Wang-Buzsaki benchmark and receipt evidence gates

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

from benchmarks import bench_model_wang_buzsaki as benchmark
from sc_neurocore.neurons.models import WangBuzsakiNeuron

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "benchmarks/results/bench_wang_buzsaki.json"
RECEIPT = ROOT / "src/sc_neurocore/neurons/reference_receipts/wang_buzsaki_1996.json"


def test_committed_benchmark_is_source_bound_and_five_runtime() -> None:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "sc-neurocore.polyglot-benchmark.v1"
    assert payload["model"] == "WangBuzsakiNeuron"
    assert payload["production_speed_claim"] is False
    assert payload["hardware_measurement_claimed"] is False
    assert payload["environment"]["single_cpu_pinned"] is True
    assert payload["environment"]["exclusive_cpu_isolation_claimed"] is False
    assert set(payload["measured_order"]) == set(benchmark.BACKENDS)
    for backend in benchmark.BACKENDS:
        row = payload["backends"][backend]
        assert row["available"] is True and row["used"] is True
        assert row["spike_count_matches_python"] is True
        assert row["final_state_matches_python"] is True
        assert row["parity_max_abs_diff"] <= benchmark.PARITY_ATOL
    expected = {
        source: hashlib.sha256((ROOT / source).read_bytes()).hexdigest()
        for source in benchmark.SOURCES
    }
    assert payload["source_hashes"] == expected


def test_reference_receipt_replays_bitwise() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    neuron = WangBuzsakiNeuron()
    digest = hashlib.sha256()
    event_indices: list[int] = []
    pattern = receipt["drive"]["pattern"]
    for index in range(receipt["oracle"]["steps"]):
        event = neuron.step(pattern[index % len(pattern)])
        if event:
            event_indices.append(index)
        digest.update(struct.pack("<dddq", neuron.v, neuron.h, neuron.n, event))
    assert event_indices == receipt["oracle"]["event_indices"]
    assert len(event_indices) == receipt["oracle"]["events"]
    assert [neuron.v, neuron.h, neuron.n] == receipt["oracle"]["final_state"]
    assert digest.hexdigest() == receipt["oracle"]["trace_sha256"]
