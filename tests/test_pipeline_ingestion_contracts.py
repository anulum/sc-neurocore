# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for pipeline ingestion contracts

"""Contracts for multimodal dataset preparation and sample access."""

from __future__ import annotations

import numpy as np

from sc_neurocore.pipeline.ingestion import DataIngestor, MultimodalDataset


def test_data_ingestor_wraps_raw_signals_and_labels() -> None:
    raw = {"signals": np.random.randn(10, 4), "labels": np.array([0, 1] * 5)}

    dataset = DataIngestor().prepare_dataset(raw)

    assert isinstance(dataset, MultimodalDataset)


def test_multimodal_dataset_returns_named_sample_mapping() -> None:
    dataset = MultimodalDataset(
        data={"eeg": np.random.randn(10, 4), "emg": np.random.randn(10, 2)},
        labels=np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
    )

    sample = dataset.get_sample(0)

    assert isinstance(sample, dict)
    assert {"eeg", "emg"} <= set(sample)
