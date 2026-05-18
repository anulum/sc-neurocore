# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Online O(1) adaptation benchmark

from __future__ import annotations

import json

from sc_neurocore.benchmarks.online_o1_adaptation import (
    ONLINE_O1_ADAPTATION_BENCHMARK_SCHEMA_VERSION,
    build_online_o1_adaptation_benchmark,
    write_online_o1_adaptation_benchmark,
)
from sc_neurocore.learning.online_o1 import OnlineO1Config


def test_online_o1_adaptation_benchmark_reports_speed_and_resources() -> None:
    config = OnlineO1Config(
        weight_bits=8,
        trace_bits=6,
        reward_bits=4,
        learning_shift=3,
        trace_decay_shift=2,
    )

    report = build_online_o1_adaptation_benchmark(
        config=config,
        n_synapses=1024,
        target_weight=192,
        max_pairings=16,
    )

    assert report["schema_version"] == ONLINE_O1_ADAPTATION_BENCHMARK_SCHEMA_VERSION
    assert report["evidence_class"] == "deterministic_simulation"
    assert report["hardware_measurement_claimed"] is False
    assert report["protocol"] == "pre_post_reward_pairing"
    assert report["target_weight"] == 192
    assert report["python"]["steps_to_target"] == 16
    assert report["python"]["final_weight"] == 216
    assert report["python"]["weight_trace"] == [27, 54, 81, 108, 135, 162, 189, 216]
    assert report["resource_estimate"]["total_state_bits"] == 26624
    assert report["resource_estimate"]["bram36_tiles"] == 1
    assert report["resource_estimate"]["estimated_luts"] == 48
    if report["rust"]["available"]:
        assert report["rust"]["steps_to_target"] == report["python"]["steps_to_target"]
        assert report["rust"]["weight_trace"] == report["python"]["weight_trace"]
        assert report["parity"]["rust_matches_python"] is True


def test_write_online_o1_adaptation_benchmark_writes_canonical_json(tmp_path) -> None:
    output = tmp_path / "online_o1_adaptation.json"

    path = write_online_o1_adaptation_benchmark(output)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert path == output
    assert payload == build_online_o1_adaptation_benchmark()
    assert path.read_text(encoding="utf-8").endswith("\n")
