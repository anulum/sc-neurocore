# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Focused cortical-column coverage contracts

"""Strict-typed coverage contracts for cortical-column edge branches."""

from __future__ import annotations

import logging
import sys
from typing import Any

import numpy as np
import pytest
from scipy import sparse

from sc_neurocore.network import _cortical_column_backends as backend_discovery
from sc_neurocore.network import cortical_column as cortical_column_module
from sc_neurocore.network.cortical_column import CorticalColumn, POPULATIONS


def _block_arrays(
    block: sparse.csr_matrix,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Return the contiguous block-array triple consumed by native SpMV paths."""
    return (
        np.ascontiguousarray(block.indptr, dtype=np.int32),
        np.ascontiguousarray(block.indices, dtype=np.int32),
        np.ascontiguousarray(block.data, dtype=np.float64),
    )


@pytest.mark.parametrize("scale", [0.0, -0.25, 1.25])
def test_population_sizes_rejects_out_of_domain_scale(scale: float) -> None:
    """Static size queries reject the same scale domain as full construction."""
    with pytest.raises(ValueError, match="scale must be in"):
        CorticalColumn.population_sizes(scale=scale)


def test_native_discovery_handles_absent_ctypes_libraries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing optional shared libraries leave Go and Mojo unavailable."""
    monkeypatch.setitem(sys.modules, "juliacall", None)
    monkeypatch.setattr(backend_discovery.os.path, "exists", lambda _path: False)

    def missing_engine(name: str) -> Any:
        raise ImportError(name)

    discovered = backend_discovery.discover_native_backends(
        __file__,
        logging.getLogger(__name__),
        missing_engine,
    )

    assert discovered.go_multi_spmv is None
    assert discovered.mojo_multi_spmv is None


def test_spmv_into_accepts_precomputed_rust_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Rust SpMV path accepts the constructor's precomputed CSR arrays."""
    block = sparse.csr_matrix(([2.0], ([1], [0])), shape=(3, 2))
    arrays = _block_arrays(block)
    x = np.array([4.0, 0.0])
    y = np.zeros(3)

    def fake_spmv(
        indptr: np.ndarray[Any, Any],
        indices: np.ndarray[Any, Any],
        data: np.ndarray[Any, Any],
        x_arg: np.ndarray[Any, Any],
        y_arg: np.ndarray[Any, Any],
    ) -> None:
        y_arg += sparse.csr_matrix(
            (data, indices, indptr),
            shape=(y_arg.size, x_arg.size),
        ).dot(x_arg)

    monkeypatch.setattr(cortical_column_module, "_HAS_RUST_CSR_SPMV", True)
    monkeypatch.setattr(cortical_column_module, "_rust_csr_spmv_add", fake_spmv)

    CorticalColumn._spmv_into(block, x, y, arrays)

    np.testing.assert_allclose(y, [0.0, 8.0, 0.0])


def test_auto_block_csr_uses_single_rust_fallback_when_multi_backend_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto block injection falls back to the single Rust SpMV bridge when present."""
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
    source = next(pop for pop in POPULATIONS if not pop.endswith("i"))
    row = col._target_offsets[target]
    excitatory_block = sparse.csr_matrix(
        ([col.w_e], ([row], [0])),
        shape=(col.n_total, col._n_total_e),
    )
    inhibitory_block = sparse.csr_matrix((col.n_total, col._n_total_i), dtype=np.float64)
    col._block_e = [excitatory_block]
    col._block_i = [inhibitory_block]
    col._block_e_arrays = [_block_arrays(excitatory_block)]
    col._block_i_arrays = [_block_arrays(inhibitory_block)]

    delayed_idx = (col._buf_idx - col._global_e_bin_steps[0]) % col._buf_len_e
    col._buf_e[source][delayed_idx, 0] = 1
    calls: list[tuple[int, int]] = []

    def fake_spmv(
        indptr: np.ndarray[Any, Any],
        indices: np.ndarray[Any, Any],
        data: np.ndarray[Any, Any],
        x: np.ndarray[Any, Any],
        y: np.ndarray[Any, Any],
    ) -> None:
        calls.append((int(indptr.size), int(x.size)))
        y += sparse.csr_matrix((data, indices, indptr), shape=(y.size, x.size)).dot(x)

    monkeypatch.setattr(cortical_column_module, "_HAS_RUST_CSR_MULTI_SPMV", False)
    monkeypatch.setattr(cortical_column_module, "_rust_csr_multi_spmv_add", None)
    monkeypatch.setattr(cortical_column_module, "_HAS_MOJO_MULTI_SPMV", False)
    monkeypatch.setattr(cortical_column_module, "_mojo_multi_spmv", None)
    monkeypatch.setattr(cortical_column_module, "_HAS_GO_MULTI_SPMV", False)
    monkeypatch.setattr(cortical_column_module, "_go_multi_spmv", None)
    monkeypatch.setattr(cortical_column_module, "_HAS_JULIA_MULTI_SPMV", False)
    monkeypatch.setattr(cortical_column_module, "_julia_multi_spmv", None)
    monkeypatch.setattr(cortical_column_module, "_HAS_RUST_CSR_SPMV", True)
    monkeypatch.setattr(cortical_column_module, "_rust_csr_spmv_add", fake_spmv)

    col._inject_block(dt=0.1)

    assert calls == [(excitatory_block.indptr.size, col._n_total_e)]
    assert col.i_syn[target][0] == pytest.approx(col.w_e)


def test_per_pair_delayed_bin_injects_nonzero_source_spikes() -> None:
    """Non-block delayed-bin injection applies the populated per-bin CSR path."""
    col = CorticalColumn(
        scale=0.02,
        scale_correction=False,
        delay_distribution=True,
        use_block_csr=False,
        bg_rate=0.0,
        seed=42,
        backend="python",
    )
    col._init_buffers(dt=0.1)

    target = "L23e"
    source = "L23e"
    bin_block = sparse.csr_matrix(
        ([1.0], ([0], [0])),
        shape=(col.sizes[target], col.sizes[source]),
    )
    col._W_bin_steps[(target, source)] = [(0, bin_block)]
    col._buf_e[source][col._buf_idx, 0] = 1

    before = float(col.i_syn[target][0])
    col._inject_per_pair(dt=0.1)

    assert col.i_syn[target][0] == pytest.approx(before + col.w_e)
