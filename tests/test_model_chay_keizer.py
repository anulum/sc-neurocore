# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ChayKeizerNeuron (Chay & Keizer 1983, 5-D) behavioural contract tests

"""Contracts for the five-dimensional Chay & Keizer 1983 beta-cell burster.

The central contract is square-wave bursting: with the published parameters the
cytosolic calcium must slowly oscillate by of order one micromolar while fast
spikes ride the active-phase plateau. An earlier reduced implementation only
ever ran a few milliseconds in its tests and silently failed to burst; these
contracts exercise realistic durations so that regression cannot recur.
"""

from __future__ import annotations

import pytest

from sc_neurocore.neurons.models.chay_keizer import ChayKeizerNeuron

#: Probe ladder the Studio behaviour facet drives every model with.
_DRIVE_LADDER = (0.0, 1.0, 4.0, 16.0, 64.0, 256.0, 1024.0)


def _run(neuron: ChayKeizerNeuron, milliseconds: float, current: float = 0.0) -> dict[str, float]:
    """Integrate and summarise the trajectory over a window."""

    steps = int(milliseconds / neuron.dt)
    spikes = 0
    ca_min, ca_max = neuron.ca, neuron.ca
    v_min, v_max = neuron.v, neuron.v
    settle = steps // 4
    for i in range(steps):
        spikes += neuron.step(current)
        if i >= settle:
            ca_min, ca_max = min(ca_min, neuron.ca), max(ca_max, neuron.ca)
            v_min, v_max = min(v_min, neuron.v), max(v_max, neuron.v)
    return {
        "spikes": spikes,
        "ca_span": ca_max - ca_min,
        "ca_min": ca_min,
        "ca_max": ca_max,
        "v_min": v_min,
        "v_max": v_max,
    }


def test_autonomous_calcium_burst_oscillation() -> None:
    """At zero current the cell bursts: calcium slowly oscillates ~1 micromolar.

    This is the model's defining behaviour and the regression the reduced
    predecessor failed — calcium must swing across a wide band, not settle.
    """

    summary = _run(ChayKeizerNeuron(), milliseconds=15000.0)
    assert summary["ca_span"] > 0.3, summary
    assert 0.2 < summary["ca_min"] < 0.6, summary
    assert 0.8 < summary["ca_max"] < 1.5, summary
    assert summary["spikes"] > 20, summary


def test_square_wave_spikes_ride_the_plateau() -> None:
    """Fast spikes sit on the burst plateau in the paper's voltage band.

    Chay & Keizer report ~12 mV spikes with a minimum near -57 mV and an
    overall amplitude near 30 mV; the trajectory must stay in that band rather
    than producing large overshooting action potentials.
    """

    summary = _run(ChayKeizerNeuron(), milliseconds=15000.0)
    assert summary["v_min"] < -50.0, summary
    assert -35.0 < summary["v_max"] < -10.0, summary


def test_calcium_dependent_potassium_paces_the_burst() -> None:
    """The calcium-activated potassium conductance is what makes calcium oscillate.

    With it, calcium repeatedly falls back through a silent phase (a burst-pause
    oscillation); without it nothing terminates the active phase, so the cell
    spikes continuously and calcium only climbs — it never falls back.
    """

    bursting = _run(ChayKeizerNeuron(), milliseconds=12000.0)
    no_kca = _run(ChayKeizerNeuron(g_kca=0.0), milliseconds=12000.0)
    # Bursting calcium drops well below its starting level on each silent phase.
    assert bursting["ca_min"] < 0.5, bursting
    # Without K(Ca) calcium rises monotonically and never falls back.
    assert no_kca["ca_min"] >= 0.79, no_kca


def test_calcium_pump_clears_calcium_without_influx() -> None:
    """With the calcium channel shut, the pump drains calcium toward zero."""

    neuron = ChayKeizerNeuron(g_ca=0.0, ca=1.0)
    for _ in range(int(20000.0 / neuron.dt)):
        neuron.step(0.0)
    assert neuron.ca < 0.05


@pytest.mark.parametrize("current", _DRIVE_LADDER)
def test_drivable_across_the_probe_ladder(current: float) -> None:
    """No drive on the behaviour-probe ladder trips the stability guard.

    The predecessor's calcium envelope was so tight that the model raised at
    rest; the faithful model must simulate cleanly across the whole ladder.
    """

    neuron = ChayKeizerNeuron()
    for _ in range(int(200.0 / neuron.dt)):
        neuron.step(current)
    assert -200.0 <= neuron.v <= 200.0
    assert 0.0 <= neuron.ca <= neuron._CA_MAX


def test_strong_current_depolarises_relative_to_rest() -> None:
    """A large applied current holds the membrane more depolarised than rest."""

    rest = ChayKeizerNeuron()
    driven = ChayKeizerNeuron()
    for _ in range(int(500.0 / rest.dt)):
        rest.step(0.0)
        driven.step(256.0)
    assert driven.v > rest.v


def test_warmer_temperature_speeds_the_gates() -> None:
    """A higher temperature factor advances the gates faster in one step."""

    cold = ChayKeizerNeuron(temp_celsius=6.3, v=-20.0)
    warm = ChayKeizerNeuron(temp_celsius=30.0, v=-20.0)
    cold.step(0.0)
    warm.step(0.0)
    # From the same depolarised start the warmer cell moves its potassium gate
    # further (phi multiplies the gate kinetics).
    assert abs(warm.n - 0.061079) > abs(cold.n - 0.061079)


def test_internal_substeps_match_a_finer_timestep() -> None:
    """Sub-stepping a coarse dt matches stepping a fine dt the same span."""

    coarse = ChayKeizerNeuron(dt=0.05)
    fine = ChayKeizerNeuron(dt=0.01)
    coarse.step(0.0)
    for _ in range(5):
        fine.step(0.0)
    assert coarse.v == pytest.approx(fine.v, abs=1e-6)
    assert coarse.ca == pytest.approx(fine.ca, abs=1e-9)


def test_reset_restores_dynamic_state_keeping_parameters() -> None:
    """Reset returns the gates and calcium to rest without touching parameters."""

    neuron = ChayKeizerNeuron(dt=0.02, g_k=10.0, k_dis=2.0)
    for _ in range(int(2000.0 / neuron.dt)):
        neuron.step(0.0)
    neuron.reset()
    assert (neuron.v, neuron.m, neuron.h, neuron.n, neuron.ca) == (
        -54.774,
        0.029725,
        0.747865,
        0.061079,
        0.8,
    )
    assert neuron.dt == 0.02
    assert neuron.g_k == 10.0
    assert neuron.k_dis == 2.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("v", float("nan")),
        ("v", 250.0),
        ("m", -0.01),
        ("h", 1.01),
        ("n", 1.5),
        ("ca", -1e-6),
        ("g_ca", -1.0),
        ("g_k", -1.0),
        ("g_kca", -1.0),
        ("g_l", -1.0),
        ("c_m", 0.0),
        ("k_dis", 0.0),
        ("radius_cm", 0.0),
        ("faraday", 0.0),
        ("f_ca", -1e-6),
        ("k_ca", -1e-6),
        ("dt", 0.0),
        ("dt", float("inf")),
    ],
)
def test_invalid_state_or_parameter_does_not_mutate(field: str, value: float) -> None:
    """An invalid runtime or parameter value raises and leaves state untouched."""

    neuron = ChayKeizerNeuron()
    setattr(neuron, field, value)
    before = (neuron.v, neuron.m, neuron.h, neuron.n, neuron.ca)
    with pytest.raises(ValueError):
        neuron.step(0.0)
    assert (neuron.v, neuron.m, neuron.h, neuron.n, neuron.ca) == before


def test_non_finite_current_rejected_without_mutation() -> None:
    """A non-finite applied current is rejected before any state changes."""

    neuron = ChayKeizerNeuron()
    before = (neuron.v, neuron.m, neuron.h, neuron.n, neuron.ca)
    with pytest.raises(ValueError, match="current must be finite"):
        neuron.step(float("nan"))
    assert (neuron.v, neuron.m, neuron.h, neuron.n, neuron.ca) == before


def test_diverging_calcium_candidate_is_caught() -> None:
    """An absurd influx scale pushes calcium past the envelope and is rejected."""

    neuron = ChayKeizerNeuron(radius_cm=1e-15, dt=0.01)
    before = (neuron.v, neuron.m, neuron.h, neuron.n, neuron.ca)
    with pytest.raises(ValueError, match="calcium candidate"):
        neuron.step(0.0)
    assert (neuron.v, neuron.m, neuron.h, neuron.n, neuron.ca) == before
