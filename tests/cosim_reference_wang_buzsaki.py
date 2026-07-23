# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Wang-Buzsaki co-simulation references

"""Independent Wang-Buzsaki spike-count and Gauss-Seidel contracts."""

from __future__ import annotations

from sc_neurocore.neurons.models.wang_buzsaki import WangBuzsakiNeuron
from tests.cosim_reference_conductance_rates import _np_exp, _reference_exprel
from tests.cosim_reference_statistics import _summarise


def _wang_buzsaki_hand_spike_count(n_macro_steps: int, current: float) -> int:
    """Return the hand-authored Wang-Buzsaki macro-step (Gauss-Seidel, crossing) spike count.

    ``WangBuzsakiNeuron.step`` is a 0.5 ms macro step of 50 inner sub-steps (``dt=0.01``)
    advanced sequentially (the gating variables ``h``/``n`` from the old voltage, then the
    membrane voltage ``v`` from the new gates), with a rising-edge ``v >= v_threshold``
    crossing on the macro boundary and no reset. The bundled ``wang_buzsaki`` schema mirrors
    that path exactly (``method="gauss_seidel"``, ``substeps=50``, state ordered ``h, n, v``,
    ``detection="crossing"``), so one hand ``step()`` corresponds to one schema macro
    ``step()``. The neuron is constructed once so the state accumulates across the train.
    """
    neuron = WangBuzsakiNeuron()
    return sum(neuron.step(current) for _ in range(n_macro_steps))


def _wang_buzsaki_macrostep_gauss_seidel_features(
    *, current: float, dt: float, steps: int, substeps: int
) -> dict[str, float]:
    """Return exact macro-step Gauss-Seidel features for the driven Wang-Buzsaki oscillator.

    The Wang-Buzsaki (1996) fast-spiking interneuron is the faithful representation of the
    maintained ``WangBuzsakiNeuron``: each macro step advances ``substeps`` inner sequential
    (Gauss-Seidel) forward-Euler sub-steps of ``dt`` — the gating variables ``h`` and ``n``
    are updated from the old voltage first, then the membrane voltage ``v`` from the
    already-updated gates (the schema declares ``method="gauss_seidel"`` with state ordered
    ``h, n, v``). Sodium activation is instantaneous: ``m_inf = alpha_m/(alpha_m+beta_m)``
    with ``alpha_m = 1/exprel(-(v+35)/10)`` (the exprel rewrite of ``0.1*(v+35)/(1-exp(...))``)
    and ``beta_m = 4*exp(-(v+60)/18)``; the potassium rate ``alpha_n`` is likewise
    ``0.1/exprel(-(v+34)/10)``. The rising-edge ``v >= v_threshold`` crossing is evaluated
    only on the macro boundary against the condition at the previous macro boundary, with
    **no reset**. The rate functions are transcribed verbatim from the schema, reusing
    :func:`_np_exp` and :func:`_reference_exprel` so the recurrence reproduces the schema
    runner bit-for-bit. The reference is an independent re-derivation of the committed
    driven-spiking trace, not a copy of the runner.

    Parameters
    ----------
    current:
        Constant input current applied at every timestep.
    dt:
        Inner sub-step timestep.
    steps:
        Number of macro steps to advance.
    substeps:
        Number of inner Gauss-Seidel sub-steps per macro step.

    Returns
    -------
    dict of str to float
        Reference feature map for the ``h``, ``n``, and ``v`` state variables plus
        spike-count and first-spike-step features.
    """
    phi = 5.0
    g_na = 35.0
    g_k = 9.0
    g_l = 0.1
    e_na = 55.0
    e_k = -90.0
    e_l = -65.0
    capacitance = 1.0
    v_threshold = -20.0
    h = 0.8
    n = 0.1
    v = -65.0
    recorded: dict[str, list[float]] = {"h": [], "n": [], "v": []}
    spikes: list[int] = []
    for _ in range(steps):
        v_prev = v
        for _ in range(substeps):
            # ``h`` (declared first): reads the old voltage and old ``h``.
            h = (
                h
                + phi
                * (0.07 * _np_exp(-(v + 58) / 20) * (1 - h) - 1 / (1 + _np_exp(-(v + 28) / 10)) * h)
                * dt
            )
            # ``n`` (declared second): reads the old voltage and old ``n``.
            n = (
                n
                + phi
                * (
                    0.1 / _reference_exprel(-(v + 34) / 10) * (1 - n)
                    - 0.125 * _np_exp(-(v + 44) / 80) * n
                )
                * dt
            )
            # ``v`` (declared last): reads the already-updated ``h``/``n`` and old ``v``.
            inv_exprel = 1 / _reference_exprel(-(v + 35) / 10)
            m_inf = inv_exprel / (inv_exprel + 4 * _np_exp(-(v + 60) / 18))
            v = (
                v
                + (
                    -g_na * m_inf**3 * h * (v - e_na)
                    - g_k * n**4 * (v - e_k)
                    - g_l * (v - e_l)
                    + current
                )
                / capacitance
                * dt
            )
        # Macro-boundary rising-edge crossing (matching the hand model / macro runner).
        spikes.append(1 if (v >= v_threshold and v_prev < v_threshold) else 0)
        recorded["h"].append(h)
        recorded["n"].append(n)
        recorded["v"].append(v)

    return _summarise(recorded, spikes)
