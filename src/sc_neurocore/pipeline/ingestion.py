# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Data ingestion and normalization for multimodal SC

from typing import Any

"""Data ingestion and normalization for multimodal SC training pipelines."""

import numpy as np
from dataclasses import dataclass
from typing import Dict


@dataclass
class MultimodalDataset:
    """
    A container for multimodal training data.
    """

    data: Dict[str, np.ndarray[Any, Any]]  # {'vision': [...], 'audio': [...]}
    labels: np.ndarray[Any, Any]

    def get_sample(self, idx: int) -> Dict[str, np.ndarray[Any, Any]]:
        return {k: v[idx] for k, v in self.data.items()}


class DataIngestor:
    """
    Ingests and normalizes multimodal datasets for SC training.
    """

    def prepare_dataset(self, raw_data: Dict[str, Any]) -> MultimodalDataset:
        """
        Normalizes and packages raw multimodal data.
        """
        processed_data = {}
        for k, v in raw_data.items():
            arr = np.array(v)
            # Normalize to [0, 1]
            arr_min = np.min(arr)
            arr_max = np.max(arr)
            if arr_max > arr_min:
                processed_data[k] = (arr - arr_min) / (arr_max - arr_min)
            else:
                processed_data[k] = np.zeros_like(arr)

        return MultimodalDataset(
            data=processed_data, labels=np.zeros(len(list(processed_data.values())[0]))
        )
