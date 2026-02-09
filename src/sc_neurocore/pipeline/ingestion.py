"""Data ingestion and normalization for multimodal SC training pipelines."""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class MultimodalDataset:
    """
    A container for multimodal training data.
    """
    data: Dict[str, np.ndarray] # {'vision': [...], 'audio': [...]}
    labels: np.ndarray
    
    def get_sample(self, idx: int) -> Dict[str, np.ndarray]:
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
                
        return MultimodalDataset(data=processed_data, labels=np.zeros(len(list(processed_data.values())[0])))
