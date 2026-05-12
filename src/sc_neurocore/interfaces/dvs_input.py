# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Interface for Dynamic Vision Sensors (Event Cameras)

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass
class DVSInputLayer:
    """
    Interface for Dynamic Vision Sensors (Event Cameras).
    Converts AER events (x, y, t, p) into SC Bitstreams.
    """

    height: int
    width: int
    decay_tau: float = 100.0  # Time constant to decay old events

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.decay_tau)) or float(self.decay_tau) <= 0.0:
            raise ValueError("decay_tau must be finite and positive")
        # Surface potential representing event density
        self.surface = np.zeros((self.height, self.width), dtype=np.float32)
        self.last_update_time = 0.0

    def process_events(self, events: list[tuple[int, int, float, int]]) -> np.ndarray[Any, Any]:
        """
        Integrate a batch of events.
        Events format: (x, y, timestamp_ms, polarity)
        Returns: Frame of probabilities [0, 1]
        """
        if not events:
            return self.surface
        self._validate_events(events)

        current_time = events[-1][2]
        dt = current_time - self.last_update_time

        # Exponential decay of old activity
        # V_new = V_old * exp(-dt/tau)
        decay_factor = np.exp(-dt / self.decay_tau)
        self.surface *= decay_factor

        # Add new events
        for x, y, t, p in events:
            if 0 <= x < self.width and 0 <= y < self.height:
                # Polarity is usually -1 or 1.
                # We want activity map. Let's just accumulate magnitude or positive density.
                self.surface[y, x] += 1.0

        # Clip/Sigmoid to [0, 1] for SC generation
        # Simple saturation
        output_probs = np.tanh(self.surface)  # Maps 0->0, High->1

        self.last_update_time = current_time
        return output_probs

    @staticmethod
    def _validate_events(events: list[tuple[int, int, float, int]]) -> None:
        previous_t: float | None = None
        for _, _, t, p in events:
            timestamp = float(t)
            if not math.isfinite(timestamp):
                raise ValueError("event timestamp must be finite")
            if previous_t is not None and timestamp < previous_t:
                raise ValueError("event timestamps must be monotonically non-decreasing")
            if p not in {-1, 0, 1}:
                raise ValueError("event polarity must be -1, 0, or 1")
            previous_t = timestamp

    def generate_bitstream_frame(self, length: int = 256) -> np.ndarray[Any, Any]:
        """
        Generate a HxWxLength bitstream cube from current surface state.
        """
        probs = np.tanh(self.surface)
        # Vectorized generation
        # (H, W, Length)
        rands = np.random.random((self.height, self.width, length))
        bits = (rands < probs[:, :, None]).astype(np.uint8)
        return bits
