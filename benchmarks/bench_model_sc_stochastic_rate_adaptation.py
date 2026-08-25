# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

"""Source/binary-bound five-runtime SC stochastic-adaptation benchmark."""

from __future__ import annotations
import hashlib
import json
import time
import numpy as np
from sc_neurocore.accel.sc_stochastic_rate_adaptation import (
    backend_available,
    simulate_sc_stochastic_rate_adaptation,
)

STEPS = 20_000
REPEATS = 5


def main() -> None:
    drive = np.tile(np.array([0.0, 10.0, 25.0, 50.0]), STEPS // 4)
    uniforms = np.random.default_rng(42).random(STEPS)
    result = {}
    for backend in ("python", "rust", "julia", "go", "mojo"):
        if not backend_available(backend):
            result[backend] = {"available": False}
            continue
        timings = []
        last = None
        for _ in range(REPEATS):
            start = time.perf_counter_ns()
            last = simulate_sc_stochastic_rate_adaptation(drive, uniforms, backend=backend)
            timings.append(time.perf_counter_ns() - start)
        events = np.asarray(last["events"], dtype=np.int64)
        result[backend] = {
            "available": True,
            "median_ns": int(np.median(timings)),
            "events": int(events.sum()),
            "trace_sha256": hashlib.sha256(events.tobytes()).hexdigest(),
        }
    print(
        json.dumps(
            {
                "model": "SCStochasticRateAdaptationNeuron",
                "steps": STEPS,
                "repeats": REPEATS,
                "results": result,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
