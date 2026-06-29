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


def test_data_ingestor_zeros_a_constant_modality() -> None:
    # A modality with no dynamic range cannot be min-max normalised; it must
    # collapse to zeros rather than divide by a zero range.
    dataset = DataIngestor().prepare_dataset({"flat": [5.0, 5.0, 5.0], "varied": [0.0, 1.0, 2.0]})

    assert np.array_equal(dataset.data["flat"], np.zeros(3))
    assert dataset.data["varied"].max() == 1.0 and dataset.data["varied"].min() == 0.0
