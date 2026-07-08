# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Potjans & Diesmann 2014 cortical microcircuit

"""Tests for the 8-population cortical microcircuit.

Smoke and determinism tests use `scale=0.02` with
`scale_correction=False` so that they finish in under ~5 s. Fidelity
tests against Potjans Table 4 require the published lower-bound
`scale=0.1` with full-scale in-degree preservation; those tests run
~25 s and are isolated to the `TestPublishedFidelity` class so they
can be filtered with `pytest -k 'not Fidelity'` for fast iteration.
"""

import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from sc_neurocore.network import cortical_column as cortical_column_module
from tests.module_reload import restore_module_namespace, snapshot_module_namespace
from sc_neurocore.network.cortical_column import (
    CONN_PROBS,
    CorticalColumn,
    FULL_SIZES,
    K_BG,
    POPULATIONS,
)


# ── Smoke / property tests ───────────────────────────────────────────


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


# ── Determinism ──────────────────────────────────────────────────────


class TestDeterminism:
    def test_same_seed_same_state(self):
        a = CorticalColumn(scale=0.02, scale_correction=False, delay_distribution=False, seed=99)
        b = CorticalColumn(scale=0.02, scale_correction=False, delay_distribution=False, seed=99)
        for p in POPULATIONS:
            np.testing.assert_array_equal(a.v[p], b.v[p])

    def test_same_seed_same_rasters(self):
        a = CorticalColumn(scale=0.02, scale_correction=False, delay_distribution=False, seed=7)
        b = CorticalColumn(scale=0.02, scale_correction=False, delay_distribution=False, seed=7)
        ra = a.simulate(duration_ms=20.0, dt=0.1)
        rb = b.simulate(duration_ms=20.0, dt=0.1)
        for p in POPULATIONS:
            np.testing.assert_array_equal(ra[p], rb[p])

    def test_different_seed_different_state(self):
        a = CorticalColumn(scale=0.02, scale_correction=False, delay_distribution=False, seed=1)
        b = CorticalColumn(scale=0.02, scale_correction=False, delay_distribution=False, seed=2)
        # At least one population must have different initial voltages.
        differs = any(not np.array_equal(a.v[p], b.v[p]) for p in POPULATIONS)
        assert differs

    def test_global_numpy_seed_does_not_leak(self):
        np.random.seed(0)
        a = CorticalColumn(scale=0.02, scale_correction=False, delay_distribution=False, seed=42)
        ra = a.simulate(duration_ms=10.0, dt=0.1)
        np.random.seed(99999)
        b = CorticalColumn(scale=0.02, scale_correction=False, delay_distribution=False, seed=42)
        rb = b.simulate(duration_ms=10.0, dt=0.1)
        for p in POPULATIONS:
            np.testing.assert_array_equal(ra[p], rb[p])


# ── Connectivity & weights ───────────────────────────────────────────


class TestConnectivity:
    def test_populations_constant_matches_table5(self):
        # Order is significant — many tests / docs rely on it.
        assert POPULATIONS == (
            "L23e",
            "L23i",
            "L4e",
            "L4i",
            "L5e",
            "L5i",
            "L6e",
            "L6i",
        )

    def test_conn_probs_shape_and_known_entries(self):
        assert CONN_PROBS.shape == (8, 8)
        # Spot-check a couple of entries from Potjans Table 5.
        # L23e ← L23e (recurrent superficial-pyramidal): 0.1009
        assert CONN_PROBS[0, 0] == pytest.approx(0.1009)
        # L5e ← L5i (deep-layer reciprocal inhibition): 0.3726
        assert CONN_PROBS[4, 5] == pytest.approx(0.3726)
        # L4i ← L23i: zero per Binzegger
        assert CONN_PROBS[3, 1] == pytest.approx(0.0029)

    def test_k_bg_table5(self):
        # Background in-degree per cell from Potjans Table 5.
        assert K_BG["L23e"] == 1600
        assert K_BG["L6e"] == 2900  # Largest of the 8.

    def test_l4_to_l23e_weight_is_doubled(self):
        # Per Potjans, the L4e → L2/3e edge is boosted to 2 · w_e.
        col = CorticalColumn(scale=0.02, scale_correction=False, delay_distribution=False, seed=42)
        assert col.w_l4_to_l23e == 2.0 * col.w_e

    def test_inhibitory_weight_uses_g_inh(self):
        col = CorticalColumn(
            scale=0.02,
            scale_correction=False,
            g_inh=4.0,
            seed=42,
        )
        assert col.w_i == pytest.approx(-4.0 * col.w_e)

    def test_sparse_adjacency_built_for_every_nonzero_pair(self):
        col = CorticalColumn(scale=0.02, scale_correction=False, delay_distribution=False, seed=42)
        nonzero_pairs = {
            (POPULATIONS[i], POPULATIONS[j])
            for i in range(8)
            for j in range(8)
            if CONN_PROBS[i, j] > 0.0
        }
        assert set(col._W.keys()) == nonzero_pairs

    def test_block_csr_path_builds_and_runs(self):
        """Opt-in `use_block_csr=True` builds 2 × n_delay_bins block
        matrices and produces biologically plausible rates.

        With the batched Rust kernel, the block path is now ON PAR
        with the per-pair scipy path at scale=0.1 (287 s vs 290 s
        for 600 ms; commit `8595c639` measurement). At scale ≥ 0.5
        the per-call sparse work grows linearly while FFI overhead
        stays constant — block path expected to win materially
        from scale=0.5 upward.
        """
        col = CorticalColumn(
            scale=0.02,
            scale_correction=False,
            delay_distribution=True,
            n_delay_bins=5,
            use_block_csr=True,
            seed=42,
        )
        assert len(col._block_e) == 5
        assert len(col._block_i) == 5
        for b in col._block_e:
            assert b.shape[0] == col.n_total
            assert b.shape[1] == col._n_total_e
        for b in col._block_i:
            assert b.shape[0] == col.n_total
            assert b.shape[1] == col._n_total_i
        # Pre-extracted (indptr, indices, data) in Rust dtypes —
        # this is what the batched multi-spmv FFI consumes.
        assert len(col._block_e_arrays) == 5
        assert len(col._block_i_arrays) == 5
        for indptr, indices, data in col._block_e_arrays:
            assert indptr.dtype == np.int32
            assert indices.dtype == np.int32
            assert data.dtype == np.float64
        # Smoke: a few steps complete without raising and emit
        # the expected shape outputs.
        for _ in range(50):
            spikes = col.step(dt=0.1)
            assert set(spikes.keys()) == set(POPULATIONS)
            for p, sp in spikes.items():
                assert sp.shape == (col.sizes[p],)

    def test_empty_block_csr_preserves_shape(self):
        block = CorticalColumn._stack_block([], n_rows=3, n_cols=5)
        assert block.shape == (3, 5)
        assert block.nnz == 0

    def test_block_csr_python_fallback_injects_delayed_spikes(self, monkeypatch):
        col = CorticalColumn(
            scale=0.02,
            scale_correction=False,
            delay_distribution=True,
            n_delay_bins=1,
            use_block_csr=True,
            bg_rate=0.0,
            seed=42,
            backend="auto",
        )
        col._init_buffers(dt=0.1)
        excitatory = [p for p in POPULATIONS if not p.endswith("i")]
        source = excitatory[0]
        target = POPULATIONS[0]
        row = col._target_offsets[target]
        col._block_e = [
            sparse.csr_matrix(([col.w_e], ([row], [0])), shape=(col.n_total, col._n_total_e))
        ]
        col._block_e_arrays = [
            (
                np.ascontiguousarray(col._block_e[0].indptr, dtype=np.int32),
                np.ascontiguousarray(col._block_e[0].indices, dtype=np.int32),
                np.ascontiguousarray(col._block_e[0].data, dtype=np.float64),
            )
        ]
        col._block_i = [sparse.csr_matrix((col.n_total, col._n_total_i), dtype=np.float64)]
        col._block_i_arrays = [
            (
                np.ascontiguousarray(col._block_i[0].indptr, dtype=np.int32),
                np.ascontiguousarray(col._block_i[0].indices, dtype=np.int32),
                np.ascontiguousarray(col._block_i[0].data, dtype=np.float64),
            )
        ]
        delayed_idx = (col._buf_idx - col._global_e_bin_steps[0]) % col._buf_len_e
        col._buf_e[source][delayed_idx, 0] = 1
        before = {p: col.i_syn[p].copy() for p in POPULATIONS}

        monkeypatch.setattr(cortical_column_module, "_HAS_RUST_CSR_SPMV", False)
        monkeypatch.setattr(cortical_column_module, "_rust_csr_spmv_add", None)
        monkeypatch.setattr(cortical_column_module, "_HAS_RUST_CSR_MULTI_SPMV", False)
        monkeypatch.setattr(cortical_column_module, "_rust_csr_multi_spmv_add", None)
        monkeypatch.setattr(cortical_column_module, "_HAS_MOJO_MULTI_SPMV", False)
        monkeypatch.setattr(cortical_column_module, "_mojo_multi_spmv", None)
        monkeypatch.setattr(cortical_column_module, "_HAS_GO_MULTI_SPMV", False)
        monkeypatch.setattr(cortical_column_module, "_go_multi_spmv", None)
        monkeypatch.setattr(cortical_column_module, "_HAS_JULIA_MULTI_SPMV", False)
        monkeypatch.setattr(cortical_column_module, "_julia_multi_spmv", None)
        col._inject_block(dt=0.1)

        assert col.i_syn[target][0] == pytest.approx(before[target][0] + col.w_e)

    def test_block_csr_single_rust_fallback_injects_delayed_spikes(self, monkeypatch):
        col = CorticalColumn(
            scale=0.02,
            scale_correction=False,
            delay_distribution=True,
            n_delay_bins=1,
            use_block_csr=True,
            bg_rate=0.0,
            seed=42,
            backend="python",
        )
        col._init_buffers(dt=0.1)
        target = POPULATIONS[0]
        row = col._target_offsets[target]
        col._block_e = [
            sparse.csr_matrix(([col.w_e], ([row], [0])), shape=(col.n_total, col._n_total_e))
        ]
        col._block_e_arrays = [
            (
                np.ascontiguousarray(col._block_e[0].indptr, dtype=np.int32),
                np.ascontiguousarray(col._block_e[0].indices, dtype=np.int32),
                np.ascontiguousarray(col._block_e[0].data, dtype=np.float64),
            )
        ]
        col._block_i = [sparse.csr_matrix((col.n_total, col._n_total_i), dtype=np.float64)]
        col._block_i_arrays = [
            (
                np.ascontiguousarray(col._block_i[0].indptr, dtype=np.int32),
                np.ascontiguousarray(col._block_i[0].indices, dtype=np.int32),
                np.ascontiguousarray(col._block_i[0].data, dtype=np.float64),
            )
        ]
        delayed_idx = (col._buf_idx - col._global_e_bin_steps[0]) % col._buf_len_e
        col._buf_e[POPULATIONS[0]][delayed_idx, 0] = 1

        def fake_spmv(indptr, indices, data, x, y):
            y += sparse.csr_matrix((data, indices, indptr), shape=(y.size, x.size)).dot(x)

        monkeypatch.setattr(cortical_column_module, "_HAS_RUST_CSR_SPMV", True)
        monkeypatch.setattr(cortical_column_module, "_rust_csr_spmv_add", fake_spmv)
        col._inject_block(dt=0.1)

        assert col.i_syn[target][0] == pytest.approx(col.w_e)

    def test_explicit_python_block_csr_does_not_call_single_rust_fallback(self, monkeypatch):
        col = CorticalColumn(
            scale=0.02,
            scale_correction=False,
            delay_distribution=True,
            n_delay_bins=1,
            use_block_csr=True,
            bg_rate=0.0,
            seed=42,
            backend="python",
        )
        col._init_buffers(dt=0.1)
        target = POPULATIONS[0]
        source = POPULATIONS[0]
        row = col._target_offsets[target]
        col._block_e = [
            sparse.csr_matrix(([col.w_e], ([row], [0])), shape=(col.n_total, col._n_total_e))
        ]
        col._block_e_arrays = [
            (
                np.ascontiguousarray(col._block_e[0].indptr, dtype=np.int32),
                np.ascontiguousarray(col._block_e[0].indices, dtype=np.int32),
                np.ascontiguousarray(col._block_e[0].data, dtype=np.float64),
            )
        ]
        col._block_i = [sparse.csr_matrix((col.n_total, col._n_total_i), dtype=np.float64)]
        col._block_i_arrays = [
            (
                np.ascontiguousarray(col._block_i[0].indptr, dtype=np.int32),
                np.ascontiguousarray(col._block_i[0].indices, dtype=np.int32),
                np.ascontiguousarray(col._block_i[0].data, dtype=np.float64),
            )
        ]
        delayed_idx = (col._buf_idx - col._global_e_bin_steps[0]) % col._buf_len_e
        col._buf_e[source][delayed_idx, 0] = 1

        def forbidden_single_rust(*_args):
            raise AssertionError("backend='python' must not call native Rust fallback")

        monkeypatch.setattr(cortical_column_module, "_HAS_RUST_CSR_SPMV", True)
        monkeypatch.setattr(cortical_column_module, "_rust_csr_spmv_add", forbidden_single_rust)
        monkeypatch.setattr(cortical_column_module, "_HAS_RUST_CSR_MULTI_SPMV", False)
        monkeypatch.setattr(cortical_column_module, "_rust_csr_multi_spmv_add", None)
        monkeypatch.setattr(cortical_column_module, "_HAS_MOJO_MULTI_SPMV", False)
        monkeypatch.setattr(cortical_column_module, "_mojo_multi_spmv", None)
        monkeypatch.setattr(cortical_column_module, "_HAS_GO_MULTI_SPMV", False)
        monkeypatch.setattr(cortical_column_module, "_go_multi_spmv", None)
        monkeypatch.setattr(cortical_column_module, "_HAS_JULIA_MULTI_SPMV", False)
        monkeypatch.setattr(cortical_column_module, "_julia_multi_spmv", None)

        col._inject_block(dt=0.1)

        assert col.i_syn[target][0] == pytest.approx(col.w_e)

    def test_spmv_into_python_and_rust_paths(self, monkeypatch):
        block = sparse.csr_matrix(([2.0], ([1], [0])), shape=(3, 2))
        x = np.array([4.0, 0.0])
        y_python = np.zeros(3)

        monkeypatch.setattr(cortical_column_module, "_HAS_RUST_CSR_SPMV", False)
        monkeypatch.setattr(cortical_column_module, "_rust_csr_spmv_add", None)
        CorticalColumn._spmv_into(block, x, y_python)
        np.testing.assert_allclose(y_python, [0.0, 8.0, 0.0])

        calls = []
        y_rust = np.zeros(3)

        def fake_spmv(indptr, indices, data, x_arg, y_arg):
            calls.append((indptr.dtype, indices.dtype, data.dtype))
            y_arg += sparse.csr_matrix((data, indices, indptr), shape=(y_arg.size, x_arg.size)).dot(
                x_arg
            )

        monkeypatch.setattr(cortical_column_module, "_HAS_RUST_CSR_SPMV", True)
        monkeypatch.setattr(cortical_column_module, "_rust_csr_spmv_add", fake_spmv)
        CorticalColumn._spmv_into(block, x, y_rust)

        np.testing.assert_allclose(y_rust, y_python)
        assert calls == [(np.dtype("int32"), np.dtype("int32"), np.dtype("float64"))]

    @pytest.mark.parametrize(
        ("native_name", "flag_name", "function_name"),
        [
            ("mojo", "_HAS_MOJO_MULTI_SPMV", "_mojo_multi_spmv"),
            ("go", "_HAS_GO_MULTI_SPMV", "_go_multi_spmv"),
            ("julia", "_HAS_JULIA_MULTI_SPMV", "_julia_multi_spmv"),
        ],
    )
    def test_block_csr_auto_native_dispatch_marshals_pointers(
        self,
        monkeypatch,
        native_name,
        flag_name,
        function_name,
    ):
        col = CorticalColumn(
            scale=0.02,
            scale_correction=False,
            delay_distribution=True,
            n_delay_bins=1,
            use_block_csr=True,
            bg_rate=0.0,
            seed=42,
            backend="auto",
        )
        col._init_buffers(dt=0.1)
        target = POPULATIONS[0]
        row = col._target_offsets[target]
        col._block_e = [sparse.csr_matrix((col.n_total, col._n_total_e), dtype=np.float64)]
        col._block_e_arrays = [
            (
                np.ascontiguousarray(col._block_e[0].indptr, dtype=np.int32),
                np.ascontiguousarray(col._block_e[0].indices, dtype=np.int32),
                np.ascontiguousarray(col._block_e[0].data, dtype=np.float64),
            )
        ]
        col._block_i = [
            sparse.csr_matrix(([col.w_i], ([row], [0])), shape=(col.n_total, col._n_total_i))
        ]
        col._block_i_arrays = [
            (
                np.ascontiguousarray(col._block_i[0].indptr, dtype=np.int32),
                np.ascontiguousarray(col._block_i[0].indices, dtype=np.int32),
                np.ascontiguousarray(col._block_i[0].data, dtype=np.float64),
            )
        ]
        inhibitory = [p for p in POPULATIONS if p.endswith("i")]
        delayed_idx = (col._buf_idx - col._global_i_bin_steps[0]) % col._buf_len_i
        col._buf_i[inhibitory[0]][delayed_idx, 0] = 1

        def fake_native(n_blocks, n_rows, _indptrs, _indices, _data, _xs, _x_lens, y_ptr):
            assert n_blocks == 1
            if native_name == "julia":
                ptr = cortical_column_module.ctypes.cast(
                    y_ptr,
                    cortical_column_module.ctypes.POINTER(cortical_column_module.ctypes.c_double),
                )
                y = np.ctypeslib.as_array(ptr, shape=(n_rows,))
            else:
                y = np.ctypeslib.as_array(y_ptr, shape=(n_rows,))
            y[row] += col.w_i

        monkeypatch.setattr(cortical_column_module, "_HAS_RUST_CSR_MULTI_SPMV", False)
        monkeypatch.setattr(cortical_column_module, "_rust_csr_multi_spmv_add", None)
        monkeypatch.setattr(cortical_column_module, "_HAS_MOJO_MULTI_SPMV", True)
        monkeypatch.setattr(cortical_column_module, "_mojo_multi_spmv", None)
        monkeypatch.setattr(cortical_column_module, "_HAS_GO_MULTI_SPMV", False)
        monkeypatch.setattr(cortical_column_module, "_go_multi_spmv", None)
        monkeypatch.setattr(cortical_column_module, "_HAS_JULIA_MULTI_SPMV", False)
        monkeypatch.setattr(cortical_column_module, "_julia_multi_spmv", None)
        monkeypatch.setattr(cortical_column_module, flag_name, True)
        monkeypatch.setattr(cortical_column_module, function_name, fake_native)

        col._inject_block(dt=0.1)

        assert col.i_syn[target][0] == pytest.approx(col.w_i)

    def test_total_indegree_counts_multapses_from_csr_data(self):
        col = CorticalColumn(scale=0.02, scale_correction=False, delay_distribution=False, seed=42)
        target = POPULATIONS[0]
        assert col.total_indegree(target) == sum(
            int(np.add.reduce(col._W[target, source].data))
            for source in POPULATIONS
            if (target, source) in col._W
        ) // max(1, col.sizes[target])


class TestNativeDiscovery:
    def test_rust_discovery_uses_root_package_fallback(self, monkeypatch):
        real_import_module = cortical_column_module._importlib.import_module

        def root_only_engine(name):
            if name == "sc_neurocore_engine.sc_neurocore_engine":
                raise ImportError(name)
            if name == "sc_neurocore_engine":
                return SimpleNamespace(
                    py_parallel_csr_spmv_add=lambda *args: None,
                    py_parallel_csr_multi_spmv_add=lambda *args: None,
                )
            return real_import_module(name)

        monkeypatch.setattr(cortical_column_module._importlib, "import_module", root_only_engine)
        _saved_ns = snapshot_module_namespace(cortical_column_module)
        reloaded = importlib.reload(cortical_column_module)
        try:
            assert reloaded._HAS_RUST_CSR_SPMV is True
            assert reloaded._HAS_RUST_CSR_MULTI_SPMV is True
        finally:
            monkeypatch.undo()
            restore_module_namespace(cortical_column_module, _saved_ns)

    def test_rust_discovery_fails_closed_without_symbols(self, monkeypatch):
        real_import_module = cortical_column_module._importlib.import_module

        def missing_engine(name):
            if name in {"sc_neurocore_engine.sc_neurocore_engine", "sc_neurocore_engine"}:
                raise ImportError(name)
            return real_import_module(name)

        monkeypatch.setattr(cortical_column_module._importlib, "import_module", missing_engine)
        _saved_ns = snapshot_module_namespace(cortical_column_module)
        reloaded = importlib.reload(cortical_column_module)
        try:
            assert reloaded._HAS_RUST_CSR_SPMV is False
            assert reloaded._rust_csr_spmv_add is None
            assert reloaded._HAS_RUST_CSR_MULTI_SPMV is False
            assert reloaded._rust_csr_multi_spmv_add is None
        finally:
            monkeypatch.undo()
            restore_module_namespace(cortical_column_module, _saved_ns)

    def test_julia_discovery_failure_remains_optional(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "juliacall", None)
        _saved_ns = snapshot_module_namespace(cortical_column_module)
        reloaded = importlib.reload(cortical_column_module)
        try:
            assert reloaded._HAS_JULIA_MULTI_SPMV is False
            assert reloaded._julia_multi_spmv is None
        finally:
            monkeypatch.undo()
            restore_module_namespace(cortical_column_module, _saved_ns)

    def test_optional_ctypes_backend_load_failures_remain_optional(self, monkeypatch):
        def fake_exists(path):
            return path.endswith("libcortical_column.so")

        def reject_cdll(path):
            raise OSError(path)

        monkeypatch.setattr(cortical_column_module.os.path, "exists", fake_exists)
        monkeypatch.setattr(cortical_column_module.ctypes, "CDLL", reject_cdll)
        _saved_ns = snapshot_module_namespace(cortical_column_module)
        reloaded = importlib.reload(cortical_column_module)
        try:
            assert reloaded._HAS_GO_MULTI_SPMV is False
            assert reloaded._go_multi_spmv is None
            assert reloaded._HAS_MOJO_MULTI_SPMV is False
            assert reloaded._mojo_multi_spmv is None
        finally:
            monkeypatch.undo()
            restore_module_namespace(cortical_column_module, _saved_ns)

    def test_mojo_ctypes_discovery_configures_symbol(self, monkeypatch):
        class FakeFunction:
            argtypes = None
            restype = object()

        fake_function = FakeFunction()
        fake_lib = SimpleNamespace(py_parallel_csr_multi_spmv_add_c=fake_function)

        def fake_exists(path):
            return path.endswith("libcortical_column.so")

        monkeypatch.setattr(cortical_column_module.os.path, "exists", fake_exists)
        monkeypatch.setattr(cortical_column_module.ctypes, "CDLL", lambda _path: fake_lib)
        _saved_ns = snapshot_module_namespace(cortical_column_module)
        reloaded = importlib.reload(cortical_column_module)
        try:
            assert reloaded._HAS_MOJO_MULTI_SPMV is True
            assert fake_function.argtypes is not None
            assert fake_function.restype is None
        finally:
            monkeypatch.undo()
            restore_module_namespace(cortical_column_module, _saved_ns)


# ── Published fidelity (Potjans 2014 Table 4) ────────────────────────


class TestPublishedFidelity:
    """Pin the qualitative features of the asynchronous-irregular state.

    These tests run the model at the published lower-bound
    `scale=0.1` with full-scale in-degree preservation. Each takes
    ~25 s on a modern CPU.
    """

    @pytest.fixture(scope="class")
    def rasters(self):
        col = CorticalColumn(scale=0.1, scale_correction=True, seed=42)
        return col, col.simulate(duration_ms=600.0, dt=0.1)

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
