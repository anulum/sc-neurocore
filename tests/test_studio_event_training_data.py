# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Temporal training data custody

"""Exercise actual local recordings, split admission and Torch batches."""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from sc_neurocore.datasets.encoders import EventBinning
from sc_neurocore.datasets.manifest import build_manifest
from sc_neurocore.datasets.splits import group_split
from sc_neurocore.studio.event_training_contract import (
    EventTrainingContract,
    resolve_event_training_contract,
)
from sc_neurocore.studio.event_training_data import (
    DATASET_ROOT_ENV,
    event_training_loaders,
    verify_event_training_data,
)
from sc_neurocore.studio.training_contract import resolve_training_config
from tests.event_dataset_support import write_shd


@pytest.fixture
def admitted(tmp_path: Path) -> Iterator[EventTrainingContract]:
    """Write generated SHD-format recordings and bind their speaker split."""
    write_shd(tmp_path, {"train": [0, 0, 1, 1, 2, 2, 3, 3], "test": [4]})
    manifest = build_manifest("shd", tmp_path, version="generated-format-fixture")
    plan = group_split(manifest, fractions={"train": 0.5, "evaluation": 0.5}, seed=7)
    contract = EventTrainingContract(
        manifest, plan, EventBinning(1.0, 4, 700, 1, "merge"), "train", "evaluation"
    )
    previous = os.environ.get(DATASET_ROOT_ENV)
    os.environ[DATASET_ROOT_ENV] = str(tmp_path)
    try:
        yield resolve_event_training_contract(contract.to_dict(), dataset="shd", timesteps=4)
    finally:
        if previous is None:
            os.environ.pop(DATASET_ROOT_ENV, None)
        else:
            os.environ[DATASET_ROOT_ENV] = previous


def test_temporal_batches_preserve_event_times_and_partial_final_batch(
    admitted: EventTrainingContract, tmp_path: Path
) -> None:
    """The real loaders retain time and every held-out sample."""
    train, evaluation = event_training_loaders(admitted, batch_size=3)
    assert len(train.dataset) == 4
    batches = list(evaluation)
    assert [batch[0].shape[0] for batch in batches] == [3, 1]
    spikes = torch.cat([batch[0] for batch in batches])
    labels = torch.cat([batch[1] for batch in batches])
    assert spikes.shape == (4, 4, 700)
    assert spikes[:, 0].sum().item() == 0
    assert spikes[:, 1].sum().item() == 4
    assert spikes[:, 2, 699].sum().item() == 4
    assert spikes[:, 3].sum().item() == 0
    assert labels.tolist() == [
        admitted.manifest.samples[position].label
        for position in admitted.split.assignment["evaluation"]
    ]


def test_training_configuration_retains_exact_portable_data_custody(
    admitted: EventTrainingContract,
) -> None:
    """Resolve and reload the public request without discarding input custody."""
    resolved = resolve_training_config(
        {"dataset": "shd", "timesteps": 4, "event_data": admitted.to_dict()}
    )
    assert resolved.event_data is not None
    assert resolved.event_data.receipt() == admitted.receipt()
    assert resolve_training_config(resolved.to_public_dict()) == resolved


def test_changed_file_is_refused_before_a_loader_is_returned(
    admitted: EventTrainingContract, tmp_path: Path
) -> None:
    """A changed corpus cannot silently use the previously declared digest."""
    write_shd(tmp_path, {"train": [8, 8, 9, 9], "test": [4]})
    with pytest.raises(ValueError, match="differ from the manifest"):
        event_training_loaders(admitted, batch_size=2)


def test_missing_file_error_does_not_expose_operator_root(
    admitted: EventTrainingContract, tmp_path: Path
) -> None:
    """An unavailable recording is reported without a private absolute path."""
    (tmp_path / "shd_train.h5").unlink()
    with pytest.raises(ValueError) as caught:
        verify_event_training_data(admitted)
    assert str(tmp_path) not in str(caught.value)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", "other", "schema"),
        ("digests", {}, "digests differ"),
        ("training_cpu_threads", True, "one CPU thread"),
        ("manifest", [], "must be an object"),
        ("train_split", 1, "must be named"),
        ("train_split", "evaluation", "distinct"),
        ("evaluation_split", "absent", "distinct"),
    ],
)
def test_invalid_input_contract_is_refused(
    admitted: EventTrainingContract, field: str, value: object, message: str
) -> None:
    """Incorrect portable declarations cannot enter the training runner."""
    payload = admitted.to_dict()
    payload[field] = value
    with pytest.raises(ValueError, match=message):
        resolve_event_training_contract(payload, dataset="shd", timesteps=4)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("n_steps", True, "must be an integer"),
        ("width", "700", "must be an integer"),
        ("dt_ms", True, "must be a number"),
        ("encoder", "poisson-rates", "event-binning"),
    ],
)
def test_invalid_encoder_types_are_refused(
    admitted: EventTrainingContract, field: str, value: object, message: str
) -> None:
    """Encoder values must retain their types and temporal semantics."""
    payload = admitted.to_dict()
    payload["encoder"][field] = value
    with pytest.raises(ValueError, match=message):
        resolve_event_training_contract(payload, dataset="shd", timesteps=4)


@pytest.mark.parametrize(
    ("encoder", "message"),
    [
        (EventBinning(1.0, 4, 699, 1, "merge"), "geometry"),
        (EventBinning(1.0, 4, 700, 1, "separate"), "no polarity"),
        (EventBinning(1.0, 3, 700, 1, "merge"), "timesteps"),
    ],
)
def test_semantically_incompatible_encoder_is_refused(
    admitted: EventTrainingContract, encoder: EventBinning, message: str
) -> None:
    """Valid encoders cannot silently change the declared sensor or window."""
    payload = admitted.to_dict()
    payload["encoder"] = encoder.declaration()
    with pytest.raises(ValueError, match=message):
        resolve_event_training_contract(payload, dataset="shd", timesteps=4)


def test_contract_dataset_must_match_training_request(admitted: EventTrainingContract) -> None:
    """A training label cannot describe a different corpus."""
    with pytest.raises(ValueError, match="requested dataset"):
        resolve_event_training_contract(admitted.to_dict(), dataset="nmnist", timesteps=4)


def test_extra_fields_are_refused(admitted: EventTrainingContract) -> None:
    """An unrecognised request field is never silently discarded."""
    payload = admitted.to_dict()
    payload["root"] = "/private/operator/path"
    with pytest.raises(ValueError, match="exactly"):
        resolve_event_training_contract(payload, dataset="shd", timesteps=4)


def test_coerced_manifest_types_are_refused(admitted: EventTrainingContract) -> None:
    """A numeric string cannot acquire authority through parser coercion."""
    payload = admitted.to_dict()
    payload["manifest"]["samples"][0]["index"] = "0"
    with pytest.raises(ValueError, match="types and values"):
        resolve_event_training_contract(payload, dataset="shd", timesteps=4)


def test_unconfigured_operator_root_is_refused(admitted: EventTrainingContract) -> None:
    """No HTTP request supplies a filesystem location on behalf of the operator."""
    del os.environ[DATASET_ROOT_ENV]
    with pytest.raises(ValueError, match="has not configured"):
        verify_event_training_data(admitted)


def test_unavailable_operator_root_is_refused(
    admitted: EventTrainingContract, tmp_path: Path
) -> None:
    """An unavailable root cannot become a silently empty training corpus."""
    os.environ[DATASET_ROOT_ENV] = str(tmp_path / "missing")
    with pytest.raises(ValueError, match="root is unavailable"):
        verify_event_training_data(admitted)


def test_outside_file_is_refused_before_manifest_rebuild(admitted: EventTrainingContract) -> None:
    """A direct library caller cannot bypass the configured root boundary."""
    manifest = replace(
        admitted.manifest,
        files=(replace(admitted.manifest.files[0], path="../outside.h5"),),
    )
    with pytest.raises(ValueError, match="outside the configured root"):
        verify_event_training_data(replace(admitted, manifest=manifest))


def test_corrupt_recording_error_does_not_expose_operator_root(
    admitted: EventTrainingContract, tmp_path: Path
) -> None:
    """Unreadable HDF5 data is reported without the private file location."""
    (tmp_path / "shd_train.h5").write_bytes(b"corrupt recording")
    with pytest.raises(ValueError, match="could not be verified") as caught:
        verify_event_training_data(admitted)
    assert str(tmp_path) not in str(caught.value)
