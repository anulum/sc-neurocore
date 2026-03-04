from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .vectorized_layer import VectorizedSCLayer


@dataclass
class MemristiveDenseLayer(VectorizedSCLayer):
    """
    Simulates a Dense Layer mapped to a Memristor Crossbar.
    Includes hardware non-idealities.
    """

    stuck_rate: float = 0.01  # % of devices stuck at 0 or 1
    variability: float = 0.05  # Sigma of weight noise (conductance drift)

    def __post_init__(self):  # type: ignore
        super().__post_init__()  # type: ignore
        self.apply_hardware_defects()  # type: ignore

    def apply_hardware_defects(self):  # type: ignore
        """
        Corrupt weights based on physical properties.
        """
        # 1. Variability (Write Noise)
        noise = np.random.normal(0, self.variability, self.weights.shape)
        self.weights = np.clip(self.weights + noise, 0, 1)

        # 2. Stuck-At Faults
        mask = np.random.random(self.weights.shape) < self.stuck_rate
        stuck_vals = np.random.randint(0, 2, self.weights.shape)  # 0 or 1
        self.weights[mask] = stuck_vals[mask]

        # Refresh packed representation
        self._refresh_packed_weights()  # type: ignore
