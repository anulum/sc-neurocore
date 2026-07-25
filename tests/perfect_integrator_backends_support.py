# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Perfect Integrator backend parity fixtures

"""Shared numeric fixtures for Perfect Integrator backend contract tests."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from sc_neurocore.neurons.models.perfect_integrator import PerfectIntegratorNeuron

GOLDENS = ((0.0, 0), (0.333, 32), (0.7, 66), (2.0, 200), (3.0, 250), (5.0, 500), (20.0, 1000))
COMPILED_BACKENDS = ("rust", "julia", "go", "mojo")


def run_backend(
    backend: str,
    *,
    current: float,
    n_steps: int = 1_000,
    factory: Callable[[], PerfectIntegratorNeuron] = PerfectIntegratorNeuron,
) -> tuple[npt.NDArray[np.float64], int, float]:
    """Run one backend and return its trace, event count, and final state."""
    neuron = factory()
    trace, spikes = neuron.simulate(n_steps, current, backend=backend)
    return trace, spikes, neuron.v


def configured_neuron() -> PerfectIntegratorNeuron:
    """Return a non-default state that exercises the complete native ABI."""
    return PerfectIntegratorNeuron(v=0.25, c_m=1.7, v_threshold=1.3, v_reset=-0.2, dt=0.37)


def c_arguments(neuron: PerfectIntegratorNeuron) -> tuple[float, ...]:
    """Return numeric fields in the C-ABI declaration order."""
    return (neuron.v, neuron.c_m, neuron.v_threshold, neuron.v_reset, neuron.dt)
