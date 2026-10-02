# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Training refusal admission through actual HTTP

"""Exercise typed training refusals without fabricated parser exceptions."""

import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from sc_neurocore.datasets.encoders import EventBinning
from sc_neurocore.datasets.manifest import build_manifest
from sc_neurocore.datasets.splits import group_split
from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.event_training_contract import EventTrainingContract
from sc_neurocore.studio.event_training_data import DATASET_ROOT_ENV
from sc_neurocore.studio.platform import StudioJobManager, StudioRuntimeSettings
from sc_neurocore.studio.training_contract import TrainingConfigError, resolve_training_config
from sc_neurocore.studio.training_refusals import TrainingRefusal
from tests.event_dataset_support import write_shd

_ROUTES = ("/api/training/start", "/api/studio/training/weight-restore/attach")
_DIAGNOSTICS = (
    "invalid literal",
    "could not convert",
    "object has no attribute",
    "is not subscriptable",
)


def _object(value: object) -> dict[str, object]:
    """Read a JSON object only after checking its actual key shape."""
    assert isinstance(value, dict) and all(isinstance(key, str) for key in value)
    return cast(dict[str, object], value)


@pytest.fixture
def experiment(tmp_path: Path) -> Iterator[tuple[TestClient, dict[str, object]]]:
    """Build a real recording manifest and an isolated application ledger."""
    recordings = tmp_path / "recordings"
    write_shd(recordings, {"train": [0, 0, 1, 1, 2, 2, 3, 3], "test": [4]})
    manifest = build_manifest("shd", recordings, version="generated-format-fixture")
    contract = EventTrainingContract(
        manifest,
        group_split(manifest, fractions={"train": 0.5, "evaluation": 0.5}, seed=7),
        EventBinning(1.0, 4, 700, 1, "merge"),
        "train",
        "evaluation",
    )
    settings = StudioRuntimeSettings(
        job_root_path=str(tmp_path / "jobs"), audit_log_path=str(tmp_path / "audit.jsonl")
    )
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        yield client, _object(contract.to_dict())


def _post_rejected(client: TestClient, route: str, config: dict[str, object]) -> object:
    """Post a real admission request and prove that no worker was allocated."""
    body = config if route == _ROUTES[0] else {"source_job_id": "sj_missing", "config": config}
    response = client.post(
        route, content=json.dumps(body), headers={"content-type": "application/json"}
    )
    assert response.status_code == 422, response.text
    assert all(fragment not in response.text for fragment in _DIAGNOSTICS)
    manager = cast(FastAPI, client.app).state.studio_job_manager
    assert isinstance(manager, StudioJobManager)
    assert manager.list_records() == ()
    return _object(response.json())["detail"]


@pytest.mark.parametrize("route", _ROUTES)
@pytest.mark.parametrize("field", ("bytes", "index", "label"))
def test_manifest_conversion_is_not_echoed(
    experiment: tuple[TestClient, dict[str, object]], route: str, field: str
) -> None:
    """Actual integer parser failures become the fixed event-data refusal."""
    client, contract = experiment
    manifest = _object(contract["manifest"])
    records = manifest["files" if field == "bytes" else "samples"]
    assert isinstance(records, list)
    _object(records[0])[field] = "caller-text-xyz"
    detail = _post_rejected(
        client, route, {"dataset": "shd", "timesteps": 4, "event_data": contract}
    )
    if route == _ROUTES[0]:
        assert _object(detail)["reason"] == "the event data declaration is malformed."
    else:
        assert detail == "event_data: the event data declaration is malformed."
    assert "caller-text-xyz" not in str(detail)


@pytest.mark.parametrize("route", _ROUTES)
def test_authored_split_reason_survives(
    experiment: tuple[TestClient, dict[str, object]], route: str
) -> None:
    """The split producer's deliberately written reason reaches both APIs."""
    client, contract = experiment
    _object(contract["split"])["seed"] = "caller-text-xyz"
    detail = _post_rejected(
        client, route, {"dataset": "shd", "timesteps": 4, "event_data": contract}
    )
    reason = "seed must be an integer in [0, 2**32)"
    assert (
        _object(detail)["reason"] == reason
        if route == _ROUTES[0]
        else detail == f"event_data: {reason}"
    )


@pytest.mark.parametrize("route", _ROUTES)
@pytest.mark.parametrize(
    ("criterion", "reason"),
    [
        ({"metric": "val_loss", "threshold": "caller-text-xyz"}, "threshold must be a number."),
        ({"metric": "val_loss", "threshold": 10**400}, "the criterion declaration is malformed."),
        (
            {"metric": "val_loss", "threshold": 1.0, "\ud800": True},
            "unknown preregistration field(s) \\ud800.",
        ),
    ],
)
def test_preregistration_refusal_is_owned_and_json_safe(
    experiment: tuple[TestClient, dict[str, object]],
    route: str,
    criterion: dict[str, object],
    reason: str,
) -> None:
    """Type errors, actual float overflow and invalid Unicode stay transport-safe."""
    client, _ = experiment
    detail = _post_rejected(client, route, {"preregistration": criterion})
    assert (
        _object(detail)["reason"] == reason
        if route == _ROUTES[0]
        else detail == f"preregistration: {reason}"
    )


def test_public_config_refusal_remains_value_error() -> None:
    """Public library callers keep compatibility and the same useful reason."""
    with pytest.raises(ValueError, match="hidden: layer 0 width") as caught:
        resolve_training_config({"hidden": ["caller-text-xyz"]})
    assert isinstance(caught.value, TrainingConfigError)
    assert isinstance(caught.value, TrainingRefusal)


@pytest.mark.parametrize("route", _ROUTES)
@pytest.mark.parametrize("field", ("lr", "max_grad_norm"))
def test_unrepresentable_scalar_is_an_authored_refusal(
    experiment: tuple[TestClient, dict[str, object]], route: str, field: str
) -> None:
    """Real float overflow is refused as invalid configuration before a job exists."""
    client, _ = experiment
    detail = _post_rejected(client, route, {field: 10**400})
    if route == _ROUTES[0]:
        assert _object(detail)["field"] == field
    else:
        assert isinstance(detail, str) and detail.startswith(f"{field}: must be")


@pytest.mark.parametrize("route", _ROUTES)
def test_http_admission_rechecks_files_redirected_during_digest_reads(
    experiment: tuple[TestClient, dict[str, object]],
    route: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Actual file redirection cannot pass either HTTP admission boundary."""
    client, contract = experiment
    root = tmp_path / "recordings"
    recording = root / "shd_test.h5"
    saved = root / "shd_test.original"
    outside = tmp_path / "outside.h5"
    original = recording.read_bytes()
    outside.write_bytes(original)
    monkeypatch.setenv(DATASET_ROOT_ENV, str(root))
    active = True
    redirected = False

    def redirect(event: str, arguments: tuple[object, ...]) -> None:
        """Schedule a real filesystem mutation at the first digest open."""
        nonlocal redirected
        if active and not redirected and event == "open" and arguments[0] == str(recording):
            redirected = True
            recording.rename(saved)
            recording.symlink_to(outside)

    sys.addaudithook(redirect)
    try:
        detail = _post_rejected(
            client, route, {"dataset": "shd", "timesteps": 4, "event_data": contract}
        )
        assert redirected
        assert detail == (
            "Invalid input"
            if route == _ROUTES[0]
            else "event dataset file lies outside the configured root"
        )
        assert str(root) not in str(detail)
    finally:
        active = False
        if redirected:
            recording.unlink()
            saved.rename(recording)
    assert recording.read_bytes() == outside.read_bytes() == original


@pytest.mark.parametrize(
    ("field", "reason"),
    [
        ("lr", "must be a positive finite number."),
        ("max_grad_norm", "must be a finite number at or above 0."),
    ],
)
def test_large_integer_cannot_break_authored_refusal_formatting(field: str, reason: str) -> None:
    """A real oversized integer is refused without invoking its decimal formatter."""
    with pytest.raises(TrainingConfigError) as caught:
        resolve_training_config({field: 10**5000})
    assert caught.value.field == field
    assert caught.value.reason == reason
    assert isinstance(caught.value, TrainingRefusal)
