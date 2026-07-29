# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC Compte network benchmark evidence tests

"""Validate benchmark scope, source custody, and deterministic receipts."""

from __future__ import annotations

import hashlib
import json

from benchmarks import bench_sc_compte_wm_network as benchmark


def test_network_benchmark_is_source_bound_and_claim_bounded() -> None:
    payload = benchmark.build_payload(steps=8, repeats=2)
    assert payload["passed"] is True
    assert payload["repeat_receipts_exact"] is True
    assert payload["production_speed_claimed"] is False
    assert payload["hardware_measurement_claimed"] is False
    assert payload["persistent_bump_claimed"] is False
    assert payload["distractor_resistance_claimed"] is False
    hashes = payload["source_sha256"]
    assert isinstance(hashes, dict)
    assert set(hashes) == set(benchmark.SOURCE_PATHS)
    for relative, digest in hashes.items():
        assert digest == hashlib.sha256((benchmark.REPOSITORY / relative).read_bytes()).hexdigest()


def test_network_benchmark_rejects_empty_measurements() -> None:
    for steps, repeats in ((0, 1), (1, 0), (-1, 1)):
        try:
            benchmark.build_payload(steps, repeats)
        except ValueError as error:
            assert "positive" in str(error)
        else:
            raise AssertionError("invalid benchmark configuration was accepted")


def test_committed_network_benchmark_receipt_has_current_source_custody() -> None:
    payload = json.loads(benchmark.DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["configuration"]["cells"] == 2560
    assert payload["configuration"]["steps"] == 1000
    assert payload["repeat_receipts_exact"] is True
    for relative in benchmark.SOURCE_PATHS:
        assert (
            payload["source_sha256"][relative]
            == hashlib.sha256((benchmark.REPOSITORY / relative).read_bytes()).hexdigest()
        )
