# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ChayKeizerMinimalNeuron (reduced 3-state Chay-Keizer) contract tests

"""Contracts for the reduced three-state Chay-Keizer beta-cell burster.

The reduced model must keep the original's defining behaviour: cytosolic calcium
slowly oscillates as a sawtooth while fast spikes ride an active-phase plateau,
in the voltage band of the reference (Bertram et al. 2023, Fig. 1).
"""

from __future__ import annotations

import pytest

from sc_neurocore.neurons.models.chay_keizer_minimal import ChayKeizerMinimalNeuron

#: Probe ladder the Studio behaviour facet drives every model with.
_DRIVE_LADDER = (0.0, 1.0, 4.0, 16.0, 64.0, 256.0, 1024.0)


def _run(
    neuron: ChayKeizerMinimalNeuron, milliseconds: float, current: float = 0.0
) -> dict[str, float]:
    """Integrate and summarise the trajectory over a window."""

    steps = int(milliseconds / neuron.dt)
    spikes = 0
    c_min, c_max = neuron.c, neuron.c
    v_min, v_max = neuron.v, neuron.v
    settle = steps // 4
    for i in range(steps):
        spikes += neuron.step(current)
        if i >= settle:
            c_min, c_max = min(c_min, neuron.c), max(c_max, neuron.c)
            v_min, v_max = min(v_min, neuron.v), max(v_max, neuron.v)
    return {
        "spikes": spikes,
        "c_span": c_max - c_min,
        "c_min": c_min,
        "c_max": c_max,
        "v_min": v_min,
        "v_max": v_max,
    }


def test_autonomous_calcium_sawtooth_bursting() -> None:
    """At zero current the cell bursts: calcium slowly oscillates as a sawtooth."""

    summary = _run(ChayKeizerMinimalNeuron(), milliseconds=40000.0)
    assert summary["c_span"] > 0.02, summary
    assert summary["spikes"] > 30, summary


def test_spikes_ride_the_plateau_in_the_reference_band() -> None:
    """Fast spikes sit on an active-phase plateau in the reference voltage band."""

    summary = _run(ChayKeizerMinimalNeuron(), milliseconds=40000.0)
    assert summary["v_min"] < -55.0, summary
    assert -40.0 < summary["v_max"] < -10.0, summary


def test_calcium_dependent_potassium_paces_the_burst() -> None:
    """The calcium-activated potassium conductance terminates the active phase.

    With it the cell bursts (silent phases between spike trains) and calcium stays
    bounded; without it nothing stops the active phase, so the cell spikes
    continuously and calcium climbs far higher.
    """

    bursting = _run(ChayKeizerMinimalNeuron(), milliseconds=40000.0)
    no_kca = _run(ChayKeizerMinimalNeuron(g_kca=0.0), milliseconds=40000.0)
    assert no_kca["spikes"] > bursting["spikes"] + 100, (bursting, no_kca)
    assert no_kca["c_max"] > bursting["c_max"] + 0.05, (bursting, no_kca)


def test_calcium_pump_clears_calcium_without_influx() -> None:
    """With the calcium current shut, the pump drains calcium toward zero."""

    neuron = ChayKeizerMinimalNeuron(g_ca=0.0, c=0.3)
    for _ in range(int(120000.0 / neuron.dt)):
        neuron.step(0.0)
    assert neuron.c < 0.05


def test_atp_conductance_sets_excitability() -> None:
    """A larger ATP-sensitive potassium conductance hyperpolarises the cell."""

    excitable = _run(ChayKeizerMinimalNeuron(), milliseconds=20000.0)
    silenced = _run(ChayKeizerMinimalNeuron(g_katp=2000.0), milliseconds=20000.0)
    assert silenced["v_max"] < excitable["v_max"]


@pytest.mark.parametrize("current", _DRIVE_LADDER)
def test_drivable_across_the_probe_ladder(current: float) -> None:
    """No drive on the behaviour-probe ladder trips the stability guard."""

    neuron = ChayKeizerMinimalNeuron()
    for _ in range(int(200.0 / neuron.dt)):
        neuron.step(current)
    assert -200.0 <= neuron.v <= 200.0
    assert 0.0 <= neuron.c <= neuron._C_MAX


def test_internal_substeps_match_a_finer_timestep() -> None:
    """Sub-stepping a coarse dt matches stepping a fine dt the same span."""

    coarse = ChayKeizerMinimalNeuron(dt=0.05)
    fine = ChayKeizerMinimalNeuron(dt=0.01)
    coarse.step(0.0)
    for _ in range(5):
        fine.step(0.0)
    assert coarse.v == pytest.approx(fine.v, abs=1e-6)
    assert coarse.c == pytest.approx(fine.c, abs=1e-9)


def test_reset_restores_dynamic_state_keeping_parameters() -> None:
    """Reset returns the gate and calcium to rest without touching parameters."""

    neuron = ChayKeizerMinimalNeuron(dt=0.02, g_k=2000.0, k_d=0.5)
    for _ in range(int(3000.0 / neuron.dt)):
        neuron.step(0.0)
    neuron.reset()
    assert (neuron.v, neuron.n, neuron.c) == (-60.0, 0.0, 0.1)
    assert neuron.dt == 0.02
    assert neuron.g_k == 2000.0
    assert neuron.k_d == 0.5


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("v", float("nan")),
        ("v", 250.0),
        ("n", -0.01),
        ("n", 1.01),
        ("c", -1e-6),
        ("g_ca", -1.0),
        ("g_k", -1.0),
        ("g_kca", -1.0),
        ("g_katp", -1.0),
        ("c_m", 0.0),
        ("s_m", 0.0),
        ("s_n", 0.0),
        ("k_d", 0.0),
        ("f_c", -1e-6),
        ("k_pmca", -1e-6),
        ("tau_n", 0.0),
        ("dt", 0.0),
        ("dt", float("inf")),
    ],
)
def test_invalid_state_or_parameter_does_not_mutate(field: str, value: float) -> None:
    """An invalid runtime or parameter value raises and leaves state untouched."""

    neuron = ChayKeizerMinimalNeuron()
    setattr(neuron, field, value)
    before = (neuron.v, neuron.n, neuron.c)
    with pytest.raises(ValueError):
        neuron.step(0.0)
    assert (neuron.v, neuron.n, neuron.c) == before


def test_non_finite_current_rejected_without_mutation() -> None:
    """A non-finite applied current is rejected before any state changes."""

    neuron = ChayKeizerMinimalNeuron()
    before = (neuron.v, neuron.n, neuron.c)
    with pytest.raises(ValueError, match="current must be finite"):
        neuron.step(float("nan"))
    assert (neuron.v, neuron.n, neuron.c) == before
