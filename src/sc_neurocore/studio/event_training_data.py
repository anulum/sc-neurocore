# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Verified temporal training data

"""Serve verified event recordings from the operator's configured local root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from sc_neurocore.datasets.event_samples import read_event_sample
from sc_neurocore.studio._event_training_runtime import (
    DATASET_ROOT_ENV as DATASET_ROOT_ENV,
    verify_event_training_data as verify_event_training_data,
)
from sc_neurocore.studio.event_training_contract import EventTrainingContract
from sc_neurocore.studio.event_training_budget import admit_event_training_input


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
