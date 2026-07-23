# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCorticalColumn from former test_cortical_column.py

"""Focused suite: TestCorticalColumn from former test_cortical_column.py."""

from __future__ import annotations

from tests.cortical_column_support import *  # noqa: F403

class TestCorticalColumn:
    def test_creates_with_defaults(self):
        col = CorticalColumn(scale=0.02, scale_correction=False, delay_distribution=False, seed=42)
        assert set(col.sizes.keys()) == set(POPULATIONS)
        assert col.n_total == sum(col.sizes.values())
        # Default scale=0.1 → ~7717 cells; here at 0.02 → ~1544.
        assert 1000 < col.n_total < 2000

    def test_invalid_scale_raises(self):
        with pytest.raises(ValueError, match="scale"):
            CorticalColumn(scale=0.0)
        with pytest.raises(ValueError, match="scale"):
            CorticalColumn(scale=1.5)

    def test_invalid_delay_bins_and_backend_raise(self):
        with pytest.raises(ValueError, match="n_delay_bins"):
            CorticalColumn(scale=0.02, n_delay_bins=0)
        with pytest.raises(ValueError, match="backend must be"):
            CorticalColumn(scale=0.02, backend="fortran")

    @pytest.mark.parametrize(
        "backend,availability_flag,match",
        [
            (
                "rust",
                "_HAS_RUST_CSR_MULTI_SPMV",
                "sc_neurocore_engine.py_parallel_csr_multi_spmv_add",
            ),
            ("julia", "_HAS_JULIA_MULTI_SPMV", "Julia kernel"),
            ("go", "_HAS_GO_MULTI_SPMV", "Go kernel"),
            ("mojo", "_HAS_MOJO_MULTI_SPMV", "Mojo kernel"),
        ],
    )
    def test_explicit_unavailable_backend_fails_closed(
        self,
        monkeypatch,
        backend,
        availability_flag,
        match,
    ):
        monkeypatch.setattr(cortical_column_module, availability_flag, False)
        with pytest.raises(RuntimeError, match=match):
            CorticalColumn(scale=0.02, backend=backend)

    def test_full_scale_sizes(self):
        # At scale=1.0, sizes should match Potjans Table 5 exactly
        # without materialising the full synapse graph.
        sizes = CorticalColumn.population_sizes(scale=1.0)
        for pop, expected in FULL_SIZES.items():
            assert sizes[pop] == expected

    def test_step_returns_per_pop_spike_dict(self):
        col = CorticalColumn(scale=0.02, scale_correction=False, delay_distribution=False, seed=42)
        spikes = col.step(dt=0.1)
        assert set(spikes.keys()) == set(POPULATIONS)
        for p, sp in spikes.items():
            assert sp.shape == (col.sizes[p],)
            assert sp.dtype == bool

    def test_simulate_returns_rasters(self):
        col = CorticalColumn(scale=0.02, scale_correction=False, delay_distribution=False, seed=42)
        rasters = col.simulate(duration_ms=20.0, dt=0.1)
        for p in POPULATIONS:
            assert rasters[p].shape == (200, col.sizes[p])
            assert rasters[p].dtype == bool

    def test_simulate_zero_steps_raises(self):
        col = CorticalColumn(scale=0.02, scale_correction=False, delay_distribution=False, seed=42)
        with pytest.raises(ValueError, match="duration_ms / dt"):
            col.simulate(duration_ms=0.0, dt=0.1)

    def test_dt_change_mid_run_raises(self):
        col = CorticalColumn(scale=0.02, scale_correction=False, delay_distribution=False, seed=42)
        col.step(dt=0.1)
        with pytest.raises(ValueError, match="dt changed mid-run"):
            col.step(dt=0.2)

    def test_no_background_no_spikes(self):
        # Cut both the background drive and let the network alone:
        # nothing should fire because there is no feedforward input.
        col = CorticalColumn(
            scale=0.02,
            scale_correction=False,
            bg_rate=0.0,
            seed=42,
        )
        rasters = col.simulate(duration_ms=100.0, dt=0.1)
        total = sum(int(np.count_nonzero(rasters[p])) for p in POPULATIONS)
        assert total == 0

    def test_reset_state_clears_voltages_and_buffers(self):
        col = CorticalColumn(
            scale=0.02,
            scale_correction=False,
            bg_rate=0.0,
            seed=42,
        )
        col.simulate(duration_ms=20.0, dt=0.1)
        col.reset_state()
        for p in POPULATIONS:
            assert np.all(col.i_syn[p] == 0.0)
            assert np.all(col.refrac[p] == 0.0)
        # dt is dropped so the next step can pick a new dt without raising.
        col.step(dt=0.05)

    def test_population_rates_drops_burn_in(self):
        col = CorticalColumn(
            scale=0.02,
            scale_correction=False,
            bg_rate=0.0,
            seed=42,
        )
        rasters = col.simulate(duration_ms=200.0, dt=0.1)
        rates = col.population_rates(rasters, dt=0.1, burn_in_ms=100.0)
        for r in rates.values():
            assert r == 0.0

    def test_population_rates_burn_in_eats_entire_run(self):
        # When `burn_in_ms` ≥ recorded duration, every per-population
        # slice is empty and the helper must return 0.0 instead of
        # crashing on `arr.shape[1]`.
        col = CorticalColumn(
            scale=0.02,
            scale_correction=False,
            bg_rate=0.0,
            seed=42,
        )
        rasters = col.simulate(duration_ms=20.0, dt=0.1)
        rates = col.population_rates(rasters, dt=0.1, burn_in_ms=200.0)
        assert all(r == 0.0 for r in rates.values())

    def test_repr_is_one_line_summary(self):
        col = CorticalColumn(scale=0.02, scale_correction=False, delay_distribution=False, seed=42)
        s = repr(col)
        assert s.startswith("CorticalColumn(")
        assert "scale=0.02" in s
        assert "n_total=" in s
        assert "\n" not in s

    def test_population_names_property(self):
        col = CorticalColumn(scale=0.02, scale_correction=False, delay_distribution=False, seed=42)
        assert tuple(col.population_names) == POPULATIONS

    def test_total_indegree_matches_potjans_table5(self):
        # With scale_correction=True the per-target indegree should
        # match the FULL-SCALE in-degree per Potjans Table 5
        # (≈ Σ_s p[t,s] · N_s_full). We allow a 5 % tolerance for
        # multapse rounding noise across seeds.
        col = CorticalColumn(
            scale=0.1,
            scale_correction=True,
            delay_distribution=False,
            seed=42,
        )
        for ti, target in enumerate(POPULATIONS):
            expected = sum(
                CONN_PROBS[ti, sj] * FULL_SIZES[POPULATIONS[sj]] for sj in range(len(POPULATIONS))
            )
            measured = col.total_indegree(target)
            assert abs(measured - expected) / expected < 0.05, (
                f"{target}: measured {measured} vs expected {expected:.0f}"
            )
