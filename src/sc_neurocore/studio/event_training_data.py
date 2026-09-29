# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Verified temporal training data

"""Serve verified event recordings from the operator's configured local root."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from sc_neurocore.datasets.event_samples import read_event_sample
from sc_neurocore.datasets.manifest import build_manifest
from sc_neurocore.studio.event_training_contract import EventTrainingContract
from sc_neurocore.studio.event_training_budget import admit_event_training_input

DATASET_ROOT_ENV = "SC_NEUROCORE_STUDIO_DATASET_ROOT"


def verify_event_training_data(contract: EventTrainingContract) -> Path:
    """Verify dataset files and sample metadata against the declared manifest.

    Parameters
    ----------
    contract:
        Structurally resolved input declaration.

    Returns
    -------
    pathlib.Path
        Absolute operator-configured root. It is never supplied by an HTTP
        request or included in exported checkpoints.

    Raises
    ------
    ValueError
        If the operator has not configured a root, the expected files are
        unavailable, or their digests, labels or groups differ.

    Notes
    -----
    Rebuilding the manifest checks sample metadata as well as file bytes.
    This is a content check, not an immutable filesystem snapshot; the
    operator must keep the dataset unchanged while training runs.
    """
    configured = os.environ.get(DATASET_ROOT_ENV)
    if not configured:
        raise ValueError("the operator has not configured a local event dataset root")
    root = Path(configured).resolve()
    if not root.is_dir():
        raise ValueError("the configured event dataset root is unavailable")
    for record in contract.manifest.files:
        if not (root / record.path).resolve().is_relative_to(root):
            raise ValueError("event dataset file lies outside the configured root")
    try:
        actual = build_manifest(
            contract.manifest.dataset.name, root, version=contract.manifest.version
        )
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError("the configured event dataset could not be verified") from exc
    if actual.digest != contract.manifest.digest:
        raise ValueError("event dataset files or sample metadata differ from the manifest")
    for record in actual.files:
        if not (root / record.path).resolve().is_relative_to(root):
            raise ValueError("event dataset file lies outside the configured root")
    return root


class EventTrainingDataset(Dataset[tuple[Any, int]]):
    """Read and encode an admitted plan's samples lazily for a Torch loader.

    Parameters
    ----------
    root:
        Verified operator root.
    contract:
        Portable file, split and encoder custody.
    part:
        A declared split part used by the training or evaluation loader.
    """

    def __init__(self, root: Path, contract: EventTrainingContract, part: str) -> None:
        self.root = root
        self.contract = contract
        self.positions = contract.split.assignment[part]

    def __len__(self) -> int:
        """Return the number of samples in this plan part."""
        return len(self.positions)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        """Return one float spike tensor ``(timesteps,channels)`` and label.

        Parameters
        ----------
        index:
            Position inside this split part.

        Returns
        -------
        tuple
            Torch tensor and recorded class label, without static repetition.
        """
        import torch

        sample = self.contract.manifest.samples[self.positions[index]]
        events = read_event_sample(self.root, self.contract.manifest.dataset.name, sample)
        spikes = self.contract.encoder.encode(events)
        return torch.from_numpy(spikes).float(), sample.label


def event_training_loaders(contract: EventTrainingContract, batch_size: int) -> tuple[Any, Any]:
    """Build lazy train/evaluation loaders after verifying the full manifest.

    Parameters
    ----------
    contract:
        Structurally resolved manifest, plan and encoder.
    batch_size:
        Requested batch size. Incomplete final batches are retained.

    Returns
    -------
    tuple
        Torch loaders whose batches have shape ``(batch,timesteps,channels)``.
        The runner transposes the first two axes for ``SpikingNet``.
    """
    from torch.utils.data import DataLoader

    admit_event_training_input(contract, batch_size)
    root = verify_event_training_data(contract)
    train = EventTrainingDataset(root, contract, contract.train_split)
    evaluation = EventTrainingDataset(root, contract, contract.evaluation_split)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(evaluation, batch_size=batch_size, shuffle=False),
    )
