# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Community benchmark contribution tests

"""Tests for the opt-in, privacy-controlled benchmark contribution path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from sc_neurocore.studio import benchmark_contribution as bc
from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform import StudioRuntimeSettings
from sc_neurocore.studio.platform.bootstrap import bootstrap_studio_admin_identity


@pytest.fixture
def client() -> TestClient:
    """Provide the actual Studio API for opt-in benchmark contributions."""
    return TestClient(create_app(), base_url="http://127.0.0.1", headers={"sec-fetch-site": "none"})


@pytest.fixture
def databank(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Place contribution persistence in a test-owned directory."""
    path = tmp_path / "contributions.jsonl"
    monkeypatch.setattr(bc, "_DATABANK_FILE", path)
    return path


def _small_run() -> dict[str, object]:
    return bc.run_local_benchmark(n_channels=64, n_taps=16, repeats=3)


def test_safe_environment_omits_machine_identifiers() -> None:
    """Collect the allowed hardware facts without machine-identifying fields."""
    env = bc.safe_environment()
    assert set(env) <= bc.ALLOWED_ENVIRONMENT_KEYS
    assert "cpu" in env and "os" in env
    assert bc._find_forbidden(env) is None


def test_local_run_produces_a_valid_bit_exact_submission() -> None:
    """Retain backend parity and declared schema in the local benchmark submission."""
    submission = _small_run()
    assert submission["schema_version"] == bc.SUBMISSION_SCHEMA_VERSION
    assert submission["hardware_measurement_claimed"] is False
    backends = submission["backends"]
    assert isinstance(backends, list) and backends
    assert all(b["bit_exact"] for b in backends)
    parity = submission["parity"]
    assert isinstance(parity, dict)
    assert parity["bit_exact_all"] is True
    assert bc.validate_submission(submission) == []


def test_validation_rejects_bad_schema_and_identifying_keys() -> None:
    """Reject unsupported submission schemas, identifying fields and invalid handles."""
    good = _small_run()
    assert bc.validate_submission({"schema_version": "wrong"})  # non-empty errors
    leaky = dict(good)
    environment = good["environment"]
    assert isinstance(environment, dict)
    leaky["environment"] = {**environment, "hostname": "my-box"}
    errors = bc.validate_submission(leaky)
    assert any("machine-identifying" in e or "disallowed" in e for e in errors)
    bad_handle = dict(good)
    bad_handle["contributor"] = {"handle": "x" * 99}
    assert any("handle" in e for e in bc.validate_submission(bad_handle))


def test_store_and_leaderboard_roundtrip(databank: Path) -> None:
    """Retain contributed handles and order the stored benchmark leaderboard."""
    assert bc.load_databank() == []
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
    """Refuse an identifying payload before creating any databank file."""
    leaky = _small_run()
    environment = leaky["environment"]
    assert isinstance(environment, dict)
    environment["machine_id"] = "abc123"
    with pytest.raises(ValueError, match="machine-identifying"):
        bc.store_contribution(leaky, handle="x")
    assert not databank.is_file()  # nothing was written


def test_api_schema_endpoint_advertises_consent_and_privacy(client: TestClient) -> None:
    """Publish the opt-in consent statement and forbidden contribution fields."""
    body = client.get("/api/benchmarks/schema").json()
    assert body["schema_version"] == bc.SUBMISSION_SCHEMA_VERSION
    assert "hostname" in body["forbidden_keys"]
    assert "opt-in" in body["consent"]


def test_api_contribute_and_databank(client: TestClient, databank: Path) -> None:
    """Persist an accepted HTTP contribution and refuse an identifying follow-up."""
    run = client.post(
        "/api/benchmarks/run", json={"n_channels": 64, "n_taps": 16, "repeats": 3}
    ).json()
    ok = client.post("/api/benchmarks/contribute", json={"submission": run, "handle": "an"})
    assert ok.status_code == 200, ok.text
    assert client.get("/api/benchmarks/databank").json()["count"] == 1

    leaky = dict(run)
    leaky["environment"] = {**run["environment"], "ip": "10.0.0.1"}
    rejected = client.post("/api/benchmarks/contribute", json={"submission": leaky, "handle": ""})
    assert rejected.status_code == 400


@pytest.mark.parametrize("budget", ["records", "bytes", "record-size"])
def test_lab_contribution_requires_identity_and_preserves_capped_store(
    tmp_path: Path, budget: str
) -> None:
    """Real lab HTTP contributions enforce authentication, both caps and page bounds."""
    bootstrap = bootstrap_studio_admin_identity(tmp_path / "identities.json")
    submission = _small_run()
    record_bytes = len((json.dumps(submission, separators=(",", ":")) + "\n").encode())
    max_bytes = record_bytes * 2 - 1 if budget == "bytes" else 16 * 1024 * 1024
    if budget == "record-size":
        max_bytes = record_bytes - 1
    path = tmp_path / "contributions.jsonl"
    app = create_app(
        StudioRuntimeSettings(
            deployment_profile="production",
            enforce_route_policies=True,
            allow_header_principal=False,
            identity_file_path=str(tmp_path / "identities.json"),
            audit_log_path=str(tmp_path / "audit.jsonl"),
            job_root_path=str(tmp_path / "jobs"),
            benchmark_databank_path=str(path),
            benchmark_max_bytes=max_bytes,
            benchmark_max_records=1 if budget == "records" else 10,
        )
    )
    with TestClient(app, base_url="http://127.0.0.1", headers={"sec-fetch-site": "none"}) as client:
        assert (
            client.post("/api/benchmarks/contribute", json={"submission": submission}).status_code
            == 401
        )
        assert not path.exists()
        client.headers["authorization"] = f"Bearer {bootstrap.bearer_token}"
        first = client.post("/api/benchmarks/contribute", json={"submission": submission})
        if budget == "record-size":
            assert first.status_code == 413
            assert first.json() == {"detail": "benchmark_record_too_large"}
            assert not path.exists()
            return
        assert first.status_code == 200
        before = path.read_bytes()
        second = client.post("/api/benchmarks/contribute", json={"submission": submission})
        assert second.status_code == 507
        assert second.json() == {"detail": "benchmark_databank_full"}
        assert path.read_bytes() == before
        assert client.get("/api/benchmarks/databank?limit=1").json()["count"] == 1
        assert client.get("/api/benchmarks/databank?limit=1&offset=1").json()["count"] == 0
        assert client.get("/api/benchmarks/databank?limit=1001").status_code == 422


@pytest.mark.parametrize("limit,offset", [(0, 0), (1001, 0), (1, -1), (1, 10001)])
def test_public_databank_refuses_invalid_page_bounds(
    tmp_path: Path, limit: int, offset: int
) -> None:
    """Reject unsupported pages without changing the persisted contributions."""
    path = tmp_path / "databank.jsonl"
    path.write_text("{}\n")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="page is outside the supported bounds"):
        bc.load_databank(path=path, limit=limit, offset=offset)
    assert path.read_bytes() == before


@pytest.mark.parametrize("max_bytes,max_records", [(0, 1), (1024, 0)])
def test_public_databank_requires_positive_store_budgets(
    tmp_path: Path, max_bytes: int, max_records: int
) -> None:
    """Refuse an invalid read budget before parsing existing contributions."""
    path = tmp_path / "databank.jsonl"
    path.write_text("{}\n")
    with pytest.raises(ValueError, match="budgets must be positive"):
        bc.load_databank(path=path, max_bytes=max_bytes, max_records=max_records)
    assert path.read_text() == "{}\n"


@pytest.mark.parametrize("shape", ["bytes", "record", "records", "non-object"])
def test_existing_databank_is_refused_when_it_exceeds_its_contract(
    tmp_path: Path, shape: str
) -> None:
    """Keep oversized or malformed durable input unchanged when a read is refused."""
    path = tmp_path / "databank.jsonl"
    content = {
        "bytes": b"{}\n{}\n",
        "record": b"x" * (1024 * 1024 + 1),
        "records": b"\n{}\n{}\n",
        "non-object": b"[]\n",
    }[shape]
    path.write_bytes(content)
    with pytest.raises(ValueError):
        bc.load_databank(
            path=path,
            max_bytes=3 if shape == "bytes" else 2 * 1024 * 1024,
            max_records=1 if shape == "records" else 10,
        )
    assert path.read_bytes() == content
