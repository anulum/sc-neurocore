# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public benchmark contribution refusal contracts

"""Exercise contribution refusals against actual benchmark results and storage."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from sc_neurocore.refusals import AuthoredRefusal
from sc_neurocore.studio import benchmark_contribution as bc
from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform import StudioRuntimeSettings

IDENTIFYING_MARKER = "operator-private-key-19"
ENVIRONMENT_REFUSAL = "environment may only carry cpu, os, python, numpy and toolchains"
IDENTIFYING_REFUSAL = "submission must not carry machine-identifying keys"


@pytest.fixture(scope="module")
def benchmark_surface(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[TestClient, Path, dict[str, object]]]:
    """Run the real API and keep its databank in a real test-owned directory."""
    root = tmp_path_factory.mktemp("benchmark-refusals")
    path = root / "databank" / "contributions.jsonl"
    with pytest.MonkeyPatch.context() as paths:
        paths.setattr(bc, "_DATABANK_FILE", path)
        app = create_app(StudioRuntimeSettings(job_root_path=str(root / "jobs")))
        with TestClient(app, base_url="http://127.0.0.1") as client:
            response = client.post(
                "/api/benchmarks/run", json={"n_channels": 64, "n_taps": 16, "repeats": 3}
            )
            assert response.status_code == 200, response.text
            submission: dict[str, object] = response.json()
            assert bc.validate_submission(submission) == []
            yield client, path, submission


REFUSED_FIELDS = [
    ("schema_version", "unrecognised", f"schema_version must be {bc.SUBMISSION_SCHEMA_VERSION!r}"),
    ("kernel", "unrecognised", f"kernel must be {bc.KERNEL!r}"),
    ("workload", {}, "workload must declare n_channels and n_taps"),
    ("backends", [], "backends must be a non-empty list"),
    ("backends", [{}], "each backend needs a name and median_call_ms"),
    ("environment", None, "environment is required"),
    ("environment", {IDENTIFYING_MARKER: "private-value-19"}, ENVIRONMENT_REFUSAL),
    ("hardware_measurement_claimed", "yes", "hardware_measurement_claimed must be a boolean"),
    (
        "contributor",
        {"handle": "invalid/handle"},
        "contributor.handle must be <=40 chars of letters/digits/space/.-_",
    ),
    ("metadata", {"HoStNaMe": "private-value-19"}, IDENTIFYING_REFUSAL),
    ("metadata", [{"nested": {"path": "/private/operator-path-19"}}], IDENTIFYING_REFUSAL),
]


@pytest.mark.parametrize(("field", "value", "reason"), REFUSED_FIELDS)
def test_contribution_refuses_invalid_fields_without_echo_or_write(
    benchmark_surface: tuple[TestClient, Path, dict[str, object]],
    field: str,
    value: object,
    reason: str,
) -> None:
    """Each declared violation stays authored through both the public store and HTTP."""
    client, path, original = benchmark_surface
    payload = copy.deepcopy(original)
    payload[field] = value
    before_payload = copy.deepcopy(payload)
    before_file = path.read_bytes() if path.exists() else None
    before_directory = path.parent.exists()
    handle = "invalid/handle" if field == "contributor" else "researcher"

    assert bc.validate_submission(payload) == [reason]
    with pytest.raises(ValueError) as failure:
        bc.store_contribution(payload, handle=handle)
    assert isinstance(failure.value, AuthoredRefusal)
    assert type(failure.value).__name__ == "BenchmarkSubmissionRefused"
    assert str(failure.value) == reason
    response = client.post(
        "/api/benchmarks/contribute", json={"submission": payload, "handle": handle}
    )
    assert response.status_code == 400, response.text
    assert response.json() == {"detail": reason}
    assert IDENTIFYING_MARKER not in response.text
    assert "private-value-19" not in response.text
    assert "/private/operator-path-19" not in response.text
    assert payload == before_payload
    assert (path.read_bytes() if path.exists() else None) == before_file
    assert path.parent.exists() == before_directory


def test_multiple_privacy_violations_preserve_order_without_names(
    benchmark_surface: tuple[TestClient, Path, dict[str, object]],
) -> None:
    """Joined diagnostics retain their order without including submitted key names."""
    client, path, original = benchmark_surface
    payload = copy.deepcopy(original)
    payload["schema_version"] = "unrecognised"
    payload["environment"] = {"hostname": "private-value-19", IDENTIFYING_MARKER: "private"}
    reasons = [
        f"schema_version must be {bc.SUBMISSION_SCHEMA_VERSION!r}",
        ENVIRONMENT_REFUSAL,
        IDENTIFYING_REFUSAL,
    ]
    before = path.read_bytes() if path.exists() else None
    assert bc.validate_submission(payload) == reasons
    response = client.post("/api/benchmarks/contribute", json={"submission": payload})
    assert response.status_code == 400
    assert response.json() == {"detail": "; ".join(reasons)}
    assert IDENTIFYING_MARKER not in response.text
    assert "hostname" not in response.text
    assert "private-value-19" not in response.text
    assert (path.read_bytes() if path.exists() else None) == before


def test_valid_run_contribution_and_leaderboard_preserve_full_submission(
    benchmark_surface: tuple[TestClient, Path, dict[str, object]],
) -> None:
    """Nondefault real timings and parity survive the complete opt-in persistence flow."""
    client, path, submission = benchmark_surface
    assert submission["workload"] == {
        "n_channels": 64,
        "n_taps": 16,
        "elements": 1024,
        "spike_density": 0.5,
    }
    assert submission["parity"] == {"reference": "python", "tolerance": 0, "bit_exact_all": True}
    backends = submission["backends"]
    assert isinstance(backends, list) and backends
    for backend in backends:
        assert backend["repeats"] == 3
        assert backend["bit_exact"] is True
        assert backend["median_call_ms"] >= 0
        assert backend["backend"] != "julia"
    assert not path.exists()
    response = client.post(
        "/api/benchmarks/contribute", json={"submission": submission, "handle": " researcher "}
    )
    assert response.status_code == 200, response.text
    assert response.json() == {"stored": True, "schema_version": bc.SUBMISSION_SCHEMA_VERSION}
    expected = copy.deepcopy(submission)
    expected["contributor"] = {"handle": "researcher"}
    assert json.loads(path.read_text()) == expected
    assert bc.load_databank() == [expected]
    assert submission["contributor"] == {"handle": ""}
    board = client.get("/api/benchmarks/databank")
    assert board.status_code == 200, board.text
    fastest = max(backends, key=lambda backend: backend["speedup_over_python"])
    environment = submission["environment"]
    assert isinstance(environment, dict)
    assert board.json() == {
        "count": 1,
        "entries": [
            {
                "cpu": environment["cpu"],
                "handle": "researcher",
                "fastest_backend": fastest["backend"],
                "speedup": fastest["speedup_over_python"],
                "workload": submission["workload"],
            }
        ],
    }


def test_corrupt_databank_uses_shared_generic_input_refusal(
    benchmark_surface: tuple[TestClient, Path, dict[str, object]],
) -> None:
    """A real JSON decoding error never becomes an authored contribution diagnostic."""
    client, path, _ = benchmark_surface
    previous = path.read_bytes() if path.exists() else None
    path.parent.mkdir(parents=True, exist_ok=True)
    corrupt = b'{"operator-private-key-19": broken/private/path}\n'
    try:
        path.write_bytes(corrupt)
        response = client.get("/api/benchmarks/databank")
        assert response.status_code == 422, response.text
        assert response.json() == {"detail": "Invalid input"}
        assert path.read_bytes() == corrupt
    finally:
        if previous is None:
            path.unlink()
        else:
            path.write_bytes(previous)


def test_unwritable_databank_is_an_internal_error_without_public_path(
    benchmark_surface: tuple[TestClient, Path, dict[str, object]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An actual directory at the file destination fails without exposing storage text."""
    client, original_path, submission = benchmark_surface
    original_bytes = original_path.read_bytes() if original_path.exists() else None
    blocked = tmp_path / "operator-private-directory-19"
    blocked.mkdir()
    monkeypatch.setattr(bc, "_DATABANK_FILE", blocked)
    response = client.post(
        "/api/benchmarks/contribute", json={"submission": submission, "handle": "researcher"}
    )
    assert response.status_code == 500, response.text
    assert response.json() == {"detail": "Internal error"}
    assert blocked.is_dir() and list(blocked.iterdir()) == []
    assert (original_path.read_bytes() if original_path.exists() else None) == original_bytes
