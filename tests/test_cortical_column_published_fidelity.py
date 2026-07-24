# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPublishedFidelity from former test_cortical_column.py

"""Focused suite: TestPublishedFidelity from former test_cortical_column.py."""

from __future__ import annotations

from tests.cortical_column_support import *  # noqa: F403


class TestPublishedFidelity:
    """Pin the qualitative features of the asynchronous-irregular state.

    These tests run the model at the published lower-bound
    `scale=0.1` with full-scale in-degree preservation. Each takes
    ~25 s on a modern CPU.
    """

    def test_no_population_silent(self, rasters):
        col, r = rasters
        rates = col.population_rates(r, dt=0.1, burn_in_ms=200.0)
        for p, rate in rates.items():
            assert rate > 0.1, f"{p} silent at {rate:.3f} Hz"

    def test_no_population_at_refractory_ceiling(self, rasters):
        # T_ref = 2 ms → max sustainable rate ≈ 500 Hz. Asynchronous-
        # irregular Potjans rates should sit well below 80 Hz.
        col, r = rasters
        rates = col.population_rates(r, dt=0.1, burn_in_ms=200.0)
        for p, rate in rates.items():
            assert rate < 80.0, f"{p} saturated at {rate:.1f} Hz"

    def test_inhibitory_faster_than_excitatory_overall(self, rasters):
        col, r = rasters
        rates = col.population_rates(r, dt=0.1, burn_in_ms=200.0)
        e_mean = np.mean([rates[p] for p in POPULATIONS if not p.endswith("i")])
        i_mean = np.mean([rates[p] for p in POPULATIONS if p.endswith("i")])
        assert i_mean > e_mean, f"Potjans E/I asymmetry violated: E={e_mean:.2f} I={i_mean:.2f}"

    def test_l4e_in_published_band(self, rasters):
        # L4e is the main thalamic-input layer in Potjans; its rate
        # is one of the most reproducible (4.51 Hz published).
        col, r = rasters
        rates = col.population_rates(r, dt=0.1, burn_in_ms=200.0)
        assert 1.0 < rates["L4e"] < 15.0, (
            f"L4e rate {rates['L4e']:.2f} Hz outside [1, 15] sanity band"
        )

    def test_per_connection_delays_tighten_rates(self, rasters):
        """With per-connection Gaussian delays (default), at least 5
        of 8 populations should sit within 1.5× of Potjans Table 4.

        This is the verification that `delay_distribution=True` is
        actually doing what it claims — the single-mean-delay path
        gives 2-7× ratios for most populations; per-connection
        Gaussian delays bring the typical ratio down to 1.2-2× by
        breaking the recurrent population synchrony that the single-
        delay path produces.
        """
        published = {
            "L23e": 0.86,
            "L23i": 2.91,
            "L4e": 4.51,
            "L4i": 5.78,
            "L5e": 7.59,
            "L5i": 8.13,
            "L6e": 1.10,
            "L6i": 8.07,
        }
        col, r = rasters
        rates = col.population_rates(r, dt=0.1, burn_in_ms=200.0)
        within_band = 0
        for p, ref in published.items():
            ratio = rates[p] / ref
            if 0.5 <= ratio <= 1.5:
                within_band += 1
        assert within_band >= 5, (
            f"only {within_band}/8 populations within [0.5, 1.5]× of "
            f"Potjans Table 4 — per-connection delay distribution may "
            f"have regressed (rates={rates})"
        )

    def test_zero_background_silent(self):
        # Sanity: with bg_rate = 0 the recurrent network has nothing
        # to bootstrap and stays silent indefinitely.
        col = CorticalColumn(
            scale=0.05,
            scale_correction=True,
            bg_rate=0.0,
            seed=42,
        )
        r = col.simulate(duration_ms=100.0, dt=0.1)
        rates = col.population_rates(r, dt=0.1, burn_in_ms=20.0)
        assert max(rates.values()) == 0.0
