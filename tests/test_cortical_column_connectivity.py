# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestConnectivity from former test_cortical_column.py

"""Focused suite: TestConnectivity from former test_cortical_column.py."""

from __future__ import annotations

from tests.cortical_column_support import *  # noqa: F403


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
