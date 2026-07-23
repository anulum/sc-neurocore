# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Connor-Stevens co-simulation references

"""Independent Connor-Stevens spike-count and macro-step RK4 contracts."""

from __future__ import annotations

from sc_neurocore.neurons.models.connor_stevens import ConnorStevensNeuron
from tests.cosim_reference_conductance_rates import _np_exp, _reference_exprel
from tests.cosim_reference_statistics import _summarise


def _connor_stevens_hand_spike_count(n_macro_steps: int, current: float) -> int:
    """Return the hand-authored Connor-Stevens macro-step (RK4, crossing) spike count.

    The maintained ``ConnorStevensNeuron.step`` is a 1 ms macro step of 100 inner
    four-stage RK4 sub-steps (``dt=0.01``) with a rising-edge ``v >= v_threshold`` crossing
    on the macro boundary and no reset. The bundled ``connor_stevens`` schema mirrors this
    exactly (``method="rk4"``, ``substeps=100``, ``detection="crossing"``), so one hand
    ``step()`` corresponds to one schema macro ``step()``.
    """
    neuron = ConnorStevensNeuron()
    return sum(neuron.step(current) for _ in range(n_macro_steps))


def _connor_stevens_macrostep_rk4_features(
    *, current: float, dt: float, steps: int, substeps: int
) -> dict[str, float]:
    """Return exact macro-step RK4 features for the driven Connor-Stevens oscillator.

    The Connor-Stevens (1971) A-current model is the faithful representation of the
    maintained ``ConnorStevensNeuron`` (RK4, sub-stepped): each macro step advances
    ``substeps`` inner four-stage classical RK4 sub-steps of ``dt``, and the rising-edge
    ``v >= 0`` crossing is evaluated only on the macro boundary against the condition at
    the previous macro boundary — matching the hand model's 100-sub-step-per-millisecond
    macro step and ``EquationNeuron.step``'s macro crossing, with **no reset**. The
    six-state membrane and Na/K/A-type gating rate functions are transcribed verbatim from
    the schema, reusing :func:`_np_exp` and :func:`_reference_exprel` (and the cube-root
    ``a``-gate) so the recurrence reproduces the schema runner bit-for-bit. The reference is
    an independent re-derivation of the committed driven-spiking trace, not a copy of the
    runner.

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
        Reference feature map for the ``v``, ``m``, ``h``, ``n``, ``a``, and ``b``
        state variables plus spike-count and first-spike-step features.
    """
    g_na = 120.0
    g_k = 20.0
    g_a = 47.7
    g_l = 0.3
    e_na = 55.0
    e_k = -72.0
    e_a = -75.0
    e_l = -17.0
    c_m = 1.0
    v_threshold = 0.0
    recorded: dict[str, list[float]] = {"v": [], "m": [], "h": [], "n": [], "a": [], "b": []}
    spikes: list[int] = []

    def deriv(sv: tuple[float, ...]) -> tuple[float, ...]:
        v, m, h, n, a, b = sv
        dv = (
            -g_na * m**3 * h * (v - e_na)
            - g_k * n**4 * (v - e_k)
            - g_a * a**3 * b * (v - e_a)
            - g_l * (v - e_l)
            + current
        ) / c_m
        dm = (
            3.8 / _reference_exprel(-(v + 29.7) / 10) * (1 - m)
            - 15.2 * _np_exp(-(v + 54.7) / 18) * m
        )
        dh = 0.266 * _np_exp(-(v + 48) / 20) * (1 - h) - 3.8 / (1 + _np_exp(-(v + 18) / 10)) * h
        dn = (
            0.2 / _reference_exprel(-(v + 45.7) / 10) * (1 - n)
            - 0.25 * _np_exp(-(v + 55.7) / 80) * n
        )
        da = (
            (0.0761 * _np_exp((v + 94.22) / 31.84) / (1 + _np_exp((v + 1.17) / 28.93)))
            ** (1.0 / 3.0)
            - a
        ) / (0.3632 + 1.158 / (1 + _np_exp((v + 55.96) / 20.12)))
        db = (1 / (1 + _np_exp((v + 53.3) / 14.54)) ** 4 - b) / (
            1.24 + 2.678 / (1 + _np_exp((v + 50) / 16.027))
        )
        return dv, dm, dh, dn, da, db

    def rk4_substep(sv: tuple[float, ...]) -> tuple[float, ...]:
        k1 = deriv(sv)
        s1 = tuple(sv[i] + 0.5 * dt * k1[i] for i in range(6))
        k2 = deriv(s1)
        s2 = tuple(sv[i] + 0.5 * dt * k2[i] for i in range(6))
        k3 = deriv(s2)
        s3 = tuple(sv[i] + dt * k3[i] for i in range(6))
        k4 = deriv(s3)
        return tuple(sv[i] + dt * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6 for i in range(6))

    state: tuple[float, ...] = (-68.0, 0.01, 0.99, 0.1, 0.5, 0.1)
    for _ in range(steps):
        v_prev = state[0]
        for _ in range(substeps):
            state = rk4_substep(state)
        # Macro-boundary rising-edge crossing (matching the hand model / macro runner).
        spikes.append(1 if (state[0] >= v_threshold and v_prev < v_threshold) else 0)
        for index, name in enumerate(("v", "m", "h", "n", "a", "b")):
            recorded[name].append(state[index])

    return _summarise(recorded, spikes)
