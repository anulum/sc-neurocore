# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Community benchmark contribution tests

"""Tests for the opt-in, privacy-controlled benchmark contribution path."""

from __future__ import annotations

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from sc_neurocore.studio import benchmark_contribution as bc
from sc_neurocore.studio.app import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(), base_url="http://127.0.0.1")


@pytest.fixture
def databank(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "contributions.jsonl"
    monkeypatch.setattr(bc, "_DATABANK_FILE", path)
    return path


def _small_run() -> dict[str, object]:
    return bc.run_local_benchmark(n_channels=64, n_taps=16, repeats=3)


def test_safe_environment_omits_machine_identifiers() -> None:
    env = bc.safe_environment()
    assert set(env) <= bc.ALLOWED_ENVIRONMENT_KEYS
    assert "cpu" in env and "os" in env
    assert bc._find_forbidden(env) is None


def test_local_run_produces_a_valid_bit_exact_submission() -> None:
    submission = _small_run()
    assert submission["schema_version"] == bc.SUBMISSION_SCHEMA_VERSION
    assert submission["hardware_measurement_claimed"] is False
    backends = submission["backends"]
    assert isinstance(backends, list) and backends
    assert all(b["bit_exact"] for b in backends)
    assert submission["parity"]["bit_exact_all"] is True
    assert bc.validate_submission(submission) == []


def test_validation_rejects_bad_schema_and_identifying_keys() -> None:
    good = _small_run()
    assert bc.validate_submission({"schema_version": "wrong"})  # non-empty errors
    leaky = dict(good)
    leaky["environment"] = {**good["environment"], "hostname": "my-box"}
    errors = bc.validate_submission(leaky)
    assert any("machine-identifying" in e or "disallowed" in e for e in errors)
    bad_handle = dict(good)
    bad_handle["contributor"] = {"handle": "x" * 99}
    assert any("handle" in e for e in bc.validate_submission(bad_handle))


def test_store_and_leaderboard_roundtrip(databank: Path) -> None:
    bc.store_contribution(_small_run(), handle="ada")
    bc.store_contribution(_small_run(), handle="grace")
    assert databank.is_file()
    rows = bc.load_databank()
    assert len(rows) == 2
    board = bc.databank_leaderboard()
    assert board["count"] == 2
    assert board["entries"][0]["speedup"] >= board["entries"][1]["speedup"]
    assert {e["handle"] for e in board["entries"]} == {"ada", "grace"}


def test_store_refuses_an_identifying_submission(databank: Path) -> None:
    leaky = _small_run()
    leaky["environment"]["machine_id"] = "abc123"
    with pytest.raises(ValueError, match="machine-identifying"):
        bc.store_contribution(leaky, handle="x")
    assert not databank.is_file()  # nothing was written


def test_api_schema_endpoint_advertises_consent_and_privacy(client: TestClient) -> None:
    body = client.get("/api/benchmarks/schema").json()
    assert body["schema_version"] == bc.SUBMISSION_SCHEMA_VERSION
    assert "hostname" in body["forbidden_keys"]
    assert "opt-in" in body["consent"]


def test_api_contribute_and_databank(client: TestClient, databank: Path) -> None:
    run = client.post(
        "/api/benchmarks/run", json={"n_channels": 64, "n_taps": 16, "repeats": 3}
    ).json()
    ok = client.post("/api/benchmarks/contribute", json={"submission": run, "handle": "an"})
    assert ok.status_code == 200, ok.text
    assert client.get("/api/benchmarks/databank").json()["count"] == 1

    leaky = dict(run)
    leaky["environment"] = {**run["environment"], "ip": "10.0.0.1"}
    rejected = client.post(
        "/api/benchmarks/contribute", json={"submission": leaky, "handle": ""}
    )
    assert rejected.status_code == 400
