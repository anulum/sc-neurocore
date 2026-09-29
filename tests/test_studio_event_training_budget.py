# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Event input memory admission

"""Check encoded input accounting against actual Torch batches and HTTP admission."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from sc_neurocore.datasets.encoders import EventBinning
from sc_neurocore.datasets.manifest import build_manifest
from sc_neurocore.datasets.splits import group_split
from sc_neurocore.studio.event_training_budget import (
    DEFAULT_EVENT_INPUT_MAX_BYTES,
    EVENT_INPUT_LIMIT_ENV,
    admit_event_training_input,
)
from sc_neurocore.studio.event_training_contract import EventTrainingContract
from sc_neurocore.studio.event_training_data import DATASET_ROOT_ENV, event_training_loaders
from sc_neurocore.studio.training_contract import TrainingConfigError, resolve_training_config
from tests.event_dataset_support import write_shd


@pytest.fixture
def contract(tmp_path: Path) -> Iterator[EventTrainingContract]:
    """Create generated recordings and restore the actual operator environment."""
    previous = {key: os.environ.get(key) for key in (EVENT_INPUT_LIMIT_ENV, DATASET_ROOT_ENV)}
    os.environ.pop(EVENT_INPUT_LIMIT_ENV, None)
    os.environ[DATASET_ROOT_ENV] = str(tmp_path)
    write_shd(tmp_path, {"train": [0, 0, 1, 1, 2, 2, 3, 3], "test": [4]})
    manifest = build_manifest("shd", tmp_path, version="generated-format-fixture")
    try:
        yield EventTrainingContract(
            manifest,
            group_split(manifest, fractions={"train": 0.5, "evaluation": 0.5}, seed=7),
            EventBinning(1.0, 4, 700, 1, "merge"),
            "train",
            "evaluation",
        )
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_accounting_matches_real_float_input_batch(contract: EventTrainingContract) -> None:
    """The receipt accounts for source sample tensors and their stacked batch."""
    receipt = admit_event_training_input(contract, batch_size=3)
    train, _ = event_training_loaders(contract, batch_size=3)
    spikes, _ = next(iter(train))
    assert receipt["sample_tensors_bytes"] == spikes.numel() * spikes.element_size()
    assert receipt["collation_bytes"] == receipt["sample_tensors_bytes"]
    assert receipt["encoding_buffer_bytes"] == 4 * 700
    assert receipt["accounted_input_bytes"] == 70_000
    assert receipt["operator_limit_bytes"] == DEFAULT_EVENT_INPUT_MAX_BYTES


def test_large_requested_batch_accounts_for_actual_part_size(
    contract: EventTrainingContract,
) -> None:
    """A batch larger than the corpus does not charge nonexistent samples."""
    receipt = admit_event_training_input(contract, batch_size=64)
    assert receipt["requested_batch_size"] == 64
    assert receipt["effective_batch_size"] == 4
    assert receipt["accounted_input_bytes"] == 92_400


def test_exact_budget_boundary_is_admitted_and_one_byte_less_is_refused(
    contract: EventTrainingContract,
) -> None:
    """The operator limit is enforced as bytes rather than rounded megabytes."""
    os.environ[EVENT_INPUT_LIMIT_ENV] = "70000"
    assert admit_event_training_input(contract, 3)["accounted_input_bytes"] == 70_000
    os.environ[EVENT_INPUT_LIMIT_ENV] = "69999"
    with pytest.raises(ValueError, match="reduce batch_size or timesteps"):
        event_training_loaders(contract, 3)


@pytest.mark.parametrize("configured", ["", "abc", "1.5", "0", "-1"])
def test_invalid_operator_limit_is_refused(
    contract: EventTrainingContract, configured: str
) -> None:
    """An invalid configuration cannot disable admission silently."""
    os.environ[EVENT_INPUT_LIMIT_ENV] = configured
    with pytest.raises(ValueError, match="must be a positive integer"):
        admit_event_training_input(contract, 3)


@pytest.mark.parametrize("batch_size", [True, 0, -1, sys.maxsize + 1])
def test_invalid_loader_batch_is_refused(contract: EventTrainingContract, batch_size: int) -> None:
    """An unrepresentable batch is rejected before handing it to Torch."""
    with pytest.raises(ValueError, match="batch size"):
        admit_event_training_input(contract, batch_size)


def test_huge_window_is_refused_by_configuration_before_tensor_allocation(
    contract: EventTrainingContract,
) -> None:
    """Arithmetic admission can reject a huge request without materialising it."""
    oversized = replace(contract, encoder=EventBinning(1.0, 10**18, 700, 1, "merge"))
    with pytest.raises(TrainingConfigError, match="input buffers require") as caught:
        resolve_training_config(
            {"dataset": "shd", "timesteps": 10**18, "event_data": oversized.to_dict()}
        )
    assert caught.value.field == "event_data"


def test_empty_parts_are_refused_for_direct_library_call(contract: EventTrainingContract) -> None:
    """A direct budget call cannot report success for an empty input."""
    split = replace(contract.split, assignment={"train": (), "evaluation": ()})
    with pytest.raises(ValueError, match="must not be empty"):
        admit_event_training_input(replace(contract, split=split), 3)
