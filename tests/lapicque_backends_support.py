# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Lapicque backend parity fixtures

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from sc_neurocore.neurons.models.lapicque import LapicqueNeuron

TRACE_ATOL = 2.0e-15
GOLDENS = ((0.0, 0), (0.5, 0), (2.0, 71), (5.0, 200), (20.0, 500))
COMPILED_BACKENDS = ("rust", "julia", "go", "mojo")


def run_backend(
    backend: str,
    *,
    current: float,
    n_steps: int = 1_000,
    factory: Callable[[], LapicqueNeuron] = LapicqueNeuron,
) -> tuple[npt.NDArray[np.float64], int, float]:
    neuron = factory()
    trace, spikes = neuron.simulate(n_steps, current, backend=backend)
    return trace, spikes, neuron.v


def configured() -> LapicqueNeuron:
    return LapicqueNeuron(
        v=0.25, v_rest=-0.1, v_reset=-0.2, v_threshold=1.3, tau=7.5, resistance=1.7, dt=0.37
    )


def c_arguments(neuron: LapicqueNeuron) -> tuple[float, ...]:
    return (
        neuron.v,
        neuron.v_rest,
        neuron.v_reset,
        neuron.v_threshold,
        neuron.tau,
        neuron.resistance,
        neuron.dt,
    )
