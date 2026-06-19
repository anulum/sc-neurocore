# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spike trace analysis: divergence detection, causal chains

"""Analyze execution traces to debug SNN behavior.

- find_divergence: compare two traces, find first timestep where spikes differ
- causal_chain: trace backward from a spike to find which input spikes caused it
- spike_diff: summary of differences between two traces
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .tracer import ExecutionTrace


@dataclass
class DivergencePoint:
    """First point where two traces diverge."""

    timestep: int
    neuron_id: int
    trace_a_spike: int
    trace_b_spike: int
    trace_a_voltage: float
    trace_b_voltage: float
    voltage_diff: float


@dataclass
class CausalEvent:
    """One event in a causal spike chain."""

    timestep: int
    neuron_id: int
    input_current: float
    voltage: float
    spiked: bool


def find_divergence(
    trace_a: ExecutionTrace,
    trace_b: ExecutionTrace,
) -> DivergencePoint | None:
    """Find the first timestep where two traces produce different spikes.

    Useful for comparing ANN-converted SNN vs directly-trained SNN,
    or Python simulation vs hardware output.

    Returns None if traces are identical.
    """
    n_steps = min(trace_a.n_steps, trace_b.n_steps)
    n_neurons = min(trace_a.n_neurons, trace_b.n_neurons)

    for t in range(n_steps):
        for n in range(n_neurons):
            if trace_a.spikes[t, n] != trace_b.spikes[t, n]:
                return DivergencePoint(
                    timestep=t,
                    neuron_id=n,
                    trace_a_spike=int(trace_a.spikes[t, n]),
                    trace_b_spike=int(trace_b.spikes[t, n]),
                    trace_a_voltage=float(trace_a.voltages[t, n]),
                    trace_b_voltage=float(trace_b.voltages[t, n]),
                    voltage_diff=abs(float(trace_a.voltages[t, n]) - float(trace_b.voltages[t, n])),
                )
    return None


def spike_diff(
    trace_a: ExecutionTrace,
    trace_b: ExecutionTrace,
) -> dict[str, Any]:
    """Summary of spike differences between two traces.

    Returns
    -------
    dict with keys:
        total_mismatches: int
        mismatch_rate: float (fraction of timestep*neuron pairs)
        first_divergence: DivergencePoint or None
        per_neuron_mismatches: ndarray
    """
    n_steps = min(trace_a.n_steps, trace_b.n_steps)
    n_neurons = min(trace_a.n_neurons, trace_b.n_neurons)

    diff = trace_a.spikes[:n_steps, :n_neurons] != trace_b.spikes[:n_steps, :n_neurons]
    total = int(diff.sum())
    per_neuron = diff.sum(axis=0)

    return {
        "total_mismatches": total,
        "mismatch_rate": total / max(n_steps * n_neurons, 1),
        "first_divergence": find_divergence(trace_a, trace_b),
        "per_neuron_mismatches": per_neuron,
    }


def causal_chain(
    trace: ExecutionTrace,
    neuron_id: int,
    timestep: int,
    max_depth: int = 10,
) -> list[CausalEvent]:
    """Trace backward from a spike to find causal input events.

    Starting from neuron_id at timestep, finds the chain of spikes
    that contributed current to this neuron in preceding timesteps.

    Parameters
    ----------
    trace : ExecutionTrace
    neuron_id : int
        Target neuron.
    timestep : int
        Timestep of the spike to explain.
    max_depth : int
        Maximum backward steps to trace.

    Returns
    -------
    list of CausalEvent
        Causal chain from target backward to inputs.
    """
    chain = []

    # Start with the target event
    chain.append(
        CausalEvent(
            timestep=timestep,
            neuron_id=neuron_id,
            input_current=float(trace.currents[timestep, neuron_id]),
            voltage=float(trace.voltages[timestep, neuron_id]),
            spiked=bool(trace.spikes[timestep, neuron_id]),
        )
    )

    # Trace backward: at each step, find neurons that spiked and
    # contributed current to the current target
    current_targets = {neuron_id}
    for depth in range(1, max_depth + 1):
        t = timestep - depth
        if t < 0:
            break

        # Find all neurons that spiked at time t
        spiking = np.where(trace.spikes[t] > 0)[0]
        if len(spiking) == 0:
            continue

        # Any spiking neuron could have contributed current to our targets
        # (we don't have the connectivity here, so we report all spikers
        # that temporally precede the target)
        for n in spiking:
            chain.append(
                CausalEvent(
                    timestep=t,
                    neuron_id=int(n),
                    input_current=float(trace.currents[t, n]),
                    voltage=float(trace.voltages[t, n]),
                    spiked=True,
                )
            )

        # Update targets for next depth
        current_targets = set(spiking.tolist())

    return chain
