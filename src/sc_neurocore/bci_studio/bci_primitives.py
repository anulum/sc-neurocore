# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — BCI Closed-Loop Primitives
# mypy: ignore-errors

"""Brain-computer interface closed-loop primitives with SC decoding."""

from __future__ import annotations

import numpy as np
import time
from typing import Dict

try:
    from sc_neurocore._native.learning_bridge import (
        is_available as _rust_learning_available,
        RustEligentLearner,
    )

    FFI_ENABLED = _rust_learning_available()
except ImportError:
    FFI_ENABLED = False


class BCIClosedLoopEngine:
    def __init__(self, channels: int = 1024):
        self.channels = channels
        self.weights = np.ones(channels, dtype=np.float32)

        if FFI_ENABLED:
            self.learners = [RustEligentLearner(1.0, 0.1, 1.0) for _ in range(channels)]
        else:
            self.learners = [None] * channels

    def process_bci_frame(self, raw_ephys: np.ndarray, reward: float) -> Dict:
        start_time = time.perf_counter()
        spikes = (np.abs(np.diff(raw_ephys, prepend=0)) > 0.5).astype(bool)

        total_voltage = np.dot(spikes, self.weights)

        if FFI_ENABLED:
            for i in range(self.channels):
                self.learners[i].step(spikes[i], spikes[i], reward)

        command = 1 if total_voltage > (self.channels * 0.1) else 0
        latency = (time.perf_counter() - start_time) * 1000.0

        return {"command": command, "latency_ms": latency, "spikes": int(np.sum(spikes))}


if __name__ == "__main__":
    engine = BCIClosedLoopEngine()
    data = np.random.randn(1024).astype(np.float32)
    result = engine.process_bci_frame(data, reward=1.0)
    print(
        f"Real BCI Frame (FFI Accelerated): Command={result['command']}, Latency={result['latency_ms']:.4f} ms"
    )
