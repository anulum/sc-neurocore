# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — FitzHugh-Nagumo co-simulation references

"""Independent FitzHugh-Nagumo spike-count and RK4 reference contracts."""

from __future__ import annotations

from sc_neurocore.neurons.equation_builder import EquationNeuron
from sc_neurocore.neurons.models.fitzhugh_nagumo import FitzHughNagumoNeuron
from tests.cosim_reference_statistics import _summarise


def _fitzhugh_nagumo_hand_spike_count(n_steps: int, current: float) -> int:
    """Return the hand-authored FitzHugh-Nagumo (RK4, rising-edge crossing) spike count."""
    neuron = FitzHughNagumoNeuron(
        dt=0.1, v=-1.0, w=-0.5, a=0.7, b=0.8, epsilon=0.08, v_threshold=1.0
    )
    return sum(neuron.step(current) for _ in range(n_steps))


def _fitzhugh_nagumo_substep_neuron(substeps: int) -> EquationNeuron:
    """Build the faithful FitzHugh-Nagumo oscillator with an artificial sub-step count.

    FitzHugh-Nagumo is polynomial, so its Q16.16 datapath is bit-exact against float64; giving
    it ``substeps`` inner steps lets the macro-step lowering be validated on a model whose only
    residual would be a logic error (no look-up-table quantisation to confound the comparison).
    """
    return EquationNeuron(
        equations={
            "v": "v - v * v * v / 3.0 - w + I",
            "w": "epsilon * (v + a - b * w)",
        },
        parameters={"a": 0.7, "b": 0.8, "epsilon": 0.08, "v_threshold": 1.0},
        state={"v": -1.0, "w": -0.5},
        threshold="v >= v_threshold",
        dt=0.1,
        method="rk4",
        detection="crossing",
        substeps=substeps,
    )


def _fitzhugh_nagumo_rk4_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return exact classical-RK4 features for the driven FitzHugh-Nagumo oscillator.

    The FitzHugh (1961) cubic membrane and linear recovery equations are advanced
    with the same four-stage RK4 step and rising-edge spike detection the faithful
    schema runner applies, with **no reset** — the re-enrolled model is a genuine
    relaxation oscillator whose spikes are upward ``v >= 1`` threshold crossings, not
    integrate-and-fire resets. The cube is written ``v * v * v`` (not ``v ** 3``) so
    it is the exact IEEE multiplication the runner and the hand model evaluate. The
    reference is an independent re-derivation of the committed relaxation-oscillation
    trace, not a copy of the runner.

    Parameters
    ----------
    current:
        Constant input current applied at every timestep.
    dt:
        Simulation timestep.
    steps:
        Number of timesteps to advance.

    Returns
    -------
    dict of str to float
        Reference feature map for the ``v`` and ``w`` state variables plus
        spike-count and first-spike-step features.
    """
    a = 0.7
    b = 0.8
    epsilon = 0.08
    threshold = 1.0
    v = -1.0
    w = -0.5
    v_values: list[float] = []
    w_values: list[float] = []
    spikes: list[int] = []

    def deriv(v_state: float, w_state: float) -> tuple[float, float]:
        return (
            v_state - v_state * v_state * v_state / 3.0 - w_state + current,
            epsilon * (v_state + a - b * w_state),
        )

    for _ in range(steps):
        v_prev = v
        k1v, k1w = deriv(v, w)
        k2v, k2w = deriv(v + 0.5 * dt * k1v, w + 0.5 * dt * k1w)
        k3v, k3w = deriv(v + 0.5 * dt * k2v, w + 0.5 * dt * k2w)
        k4v, k4w = deriv(v + dt * k3v, w + dt * k3w)
        v = v + dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0
        w = w + dt * (k1w + 2.0 * k2w + 2.0 * k3w + k4w) / 6.0
        # Rising-edge crossing: fires when the post-step membrane is at/above threshold
        # and the previous committed membrane was below it (matching the hand model's
        # ``v >= thr and v_prev < thr`` edge test); no reset for this oscillator.
        spikes.append(1 if (v >= threshold and v_prev < threshold) else 0)
        v_values.append(v)
        w_values.append(w)

    return _summarise({"v": v_values, "w": w_values}, spikes)
