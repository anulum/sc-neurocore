# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hodgkin-Huxley co-simulation references

"""Independent Hodgkin-Huxley spike-count and macro-step RK4 contracts."""

from __future__ import annotations

from sc_neurocore.neurons.models.hodgkin_huxley import HodgkinHuxleyNeuron
from tests.cosim_reference_conductance_rates import _np_exp, _reference_exprel
from tests.cosim_reference_statistics import _summarise


def _hodgkin_huxley_hand_spike_count(n_macro_steps: int, current: float) -> int:
    """Return the hand-authored Hodgkin-Huxley macro-step (RK4, crossing) spike count.

    ``HodgkinHuxleyNeuron.step`` is a 1 ms macro step of 100 inner sub-steps (``dt=0.01``)
    with a rising-edge ``v >= v_threshold`` crossing on the macro boundary and no reset. The
    bundled ``hodgkin_huxley`` schema mirrors the ``integrator="rk4"`` path exactly
    (``method="rk4"``, ``substeps=100``, ``detection="crossing"``) — the simultaneous RK4,
    not the Gauss-Seidel default ``baseline_euler`` — so one hand ``step()`` corresponds to
    one schema macro ``step()``.
    """
    neuron = HodgkinHuxleyNeuron(integrator="rk4")
    return sum(neuron.step(current) for _ in range(n_macro_steps))


def _hodgkin_huxley_macrostep_rk4_features(
    *, current: float, dt: float, steps: int, substeps: int
) -> dict[str, float]:
    """Return exact macro-step RK4 features for the driven Hodgkin-Huxley oscillator.

    The Hodgkin-Huxley (1952) model is the faithful representation of the maintained
    ``HodgkinHuxleyNeuron(integrator="rk4")``, whose ``step()`` is itself a 100-sub-step
    macro step: each macro step advances ``substeps`` inner four-stage classical RK4
    sub-steps of ``dt`` over the same simultaneous derivative, and the rising-edge
    ``v >= 0`` crossing is evaluated only on the macro boundary against the condition at
    the previous macro boundary, with **no reset**. The four-state membrane and Na/K
    gating rate functions are transcribed verbatim from the schema, reusing
    :func:`_np_exp` and :func:`_reference_exprel` (the exprel-rewritten ``alpha_m`` /
    ``alpha_n``) so the recurrence reproduces the schema runner bit-for-bit. The
    reference is an independent re-derivation of the committed driven-spiking trace, not a
    copy of the runner.

    Parameters
    ----------
    current:
        Constant input current applied at every timestep.
    dt:
        Inner sub-step timestep.
    steps:
        Number of macro steps to advance.
    substeps:
        Number of inner RK4 sub-steps per macro step.

    Returns
    -------
    dict of str to float
        Reference feature map for the ``v``, ``m``, ``h``, and ``n`` state variables
        plus spike-count and first-spike-step features.
    """
    g_na = 120.0
    g_k = 36.0
    g_l = 0.3
    e_na = 50.0
    e_k = -77.0
    e_l = -54.4
    c_m = 1.0
    v_threshold = 0.0
    recorded: dict[str, list[float]] = {"v": [], "m": [], "h": [], "n": []}
    spikes: list[int] = []

    def deriv(sv: tuple[float, ...]) -> tuple[float, ...]:
        v, m, h, n = sv
        dv = (
            -g_na * m**3 * h * (v - e_na) - g_k * n**4 * (v - e_k) - g_l * (v - e_l) + current
        ) / c_m
        dm = 1.0 / _reference_exprel(-(v + 40) / 10) * (1 - m) - 4 * _np_exp(-(v + 65) / 18) * m
        dh = 0.07 * _np_exp(-(v + 65) / 20) * (1 - h) - 1 / (1 + _np_exp(-(v + 35) / 10)) * h
        dn = 0.1 / _reference_exprel(-(v + 55) / 10) * (1 - n) - 0.125 * _np_exp(-(v + 65) / 80) * n
        return dv, dm, dh, dn

    def rk4_substep(sv: tuple[float, ...]) -> tuple[float, ...]:
        k1 = deriv(sv)
        s1 = tuple(sv[i] + 0.5 * dt * k1[i] for i in range(4))
        k2 = deriv(s1)
        s2 = tuple(sv[i] + 0.5 * dt * k2[i] for i in range(4))
        k3 = deriv(s2)
        s3 = tuple(sv[i] + dt * k3[i] for i in range(4))
        k4 = deriv(s3)
        return tuple(sv[i] + dt * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6 for i in range(4))

    state: tuple[float, ...] = (-65.0, 0.05, 0.6, 0.32)
    for _ in range(steps):
        v_prev = state[0]
        for _ in range(substeps):
            state = rk4_substep(state)
        # Macro-boundary rising-edge crossing (matching the hand model / macro runner).
        spikes.append(1 if (state[0] >= v_threshold and v_prev < v_threshold) else 0)
        for index, name in enumerate(("v", "m", "h", "n")):
            recorded[name].append(state[index])

    return _summarise(recorded, spikes)
