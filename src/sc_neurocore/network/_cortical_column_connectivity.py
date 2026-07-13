# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Cortical-column sparse connectivity construction

"""Sparse connectivity and delay-bin construction for the cortical column."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse, stats

from ._cortical_column_parameters import (
    CONN_PROBS,
    DELAY_E,
    DELAY_E_SIGMA,
    DELAY_I,
    DELAY_I_SIGMA,
    FULL_SIZES,
    POPULATIONS,
    _is_inhibitory,
)


class _CorticalConnectivity:
    """Build the Potjans sparse graph and its optional delay-bin blocks."""

    scale_correction: bool
    delay_distribution: bool
    n_delay_bins: int
    use_block_csr: bool
    sizes: dict[str, int]
    n_total: int
    w_e: float
    w_i: float
    w_l4_to_l23e: float
    _rng: np.random.Generator

    def _build_connectivity(self) -> None:
        """Materialise per-pair and optional block-CSR connectivity."""
        scale_correction = self.scale_correction
        delay_distribution = self.delay_distribution
        n_delay_bins = self.n_delay_bins
        use_block_csr = self.use_block_csr
        # Build per-target-source SPARSE BINARY adjacency matrices,
        # one per (target, source) pair where the connection
        # probability is > 0. Each matrix W[t, s] has shape
        # (n_t_scaled, n_s_scaled) with entries in {0, 1} sampled
        # i.i.d. Bernoulli(p_eff). At sub-full scale with
        # `scale_correction=True` we keep the FULL-SCALE indegree
        # K_real per target cell by sampling K_real connections
        # uniformly from the smaller scaled source population
        # (van Albada 2015). Without correction we use Bernoulli at
        # the literal connection probability `p`.
        # When `delay_distribution=True`, additionally split each
        # pair's connections into `n_delay_bins` groups by per-
        # connection Gaussian-sampled delay. Each bin gets its own
        # sparse adjacency and its own integer-step delay offset.
        # In `use_block_csr=True` mode (default) the bin grid is
        # GLOBAL: bin centres come from theoretical Gaussian
        # quantiles, all pairs snap to the same set of delays, and
        # per-pair sub-matrices are vertically/horizontally stacked
        # into one block CSR per (source-type, bin) so per-step
        # cost collapses from O(n_pairs × n_bins) to O(2 × n_bins)
        # mat-vecs.
        # In `use_block_csr=False` mode the bin grid is PER-PAIR
        # (quantiles of that pair's specific delay sample), giving
        # a slightly richer model per pair but ~30× slower step.
        self._W: dict[tuple[str, str], sparse.csr_matrix] = {}
        self._W_bins: dict[
            tuple[str, str],
            list[tuple[float, sparse.csr_matrix]],
        ] = {}
        self._W_bin_steps: dict[
            tuple[str, str],
            list[tuple[int, sparse.csr_matrix]],
        ] = {}

        # Global delay-bin centres (in milliseconds), one set per
        # source-type, used when `use_block_csr=True`. Derived from
        # theoretical Gaussian quantiles at midpoints of equal-area
        # bins. Conversion to integer dt steps happens at
        # `_init_buffers(dt)`.
        if delay_distribution and use_block_csr:
            qmid = (np.arange(n_delay_bins) + 0.5) / n_delay_bins
            z = stats.norm.ppf(qmid)
            self._global_e_centers_ms = np.clip(
                DELAY_E + z * DELAY_E_SIGMA,
                0.05,
                None,
            )
            self._global_i_centers_ms = np.clip(
                DELAY_I + z * DELAY_I_SIGMA,
                0.05,
                None,
            )
        else:
            self._global_e_centers_ms = np.array([DELAY_E])
            self._global_i_centers_ms = np.array([DELAY_I])

        # Stacked block CSRs per (source-type, bin_idx). Built only
        # when `delay_distribution=True and use_block_csr=True`.
        # Each block has shape (n_total, n_total_<source-type>) and
        # already has weights (w_e, w_i, w_l4_to_l23e) baked into
        # the data values, so the per-step `dot` returns the
        # weighted contribution directly.
        # Per-source-type cumulative offsets so a pair's
        # (rows, cols) index pair can be lifted into the global
        # block index space at construction.
        self._target_offsets: dict[str, int] = {}
        self._source_e_offsets: dict[str, int] = {}
        self._source_i_offsets: dict[str, int] = {}
        t_off = 0
        e_off = 0
        i_off = 0
        for p in POPULATIONS:
            self._target_offsets[p] = t_off
            t_off += self.sizes[p]
            if _is_inhibitory(p):
                self._source_i_offsets[p] = i_off
                i_off += self.sizes[p]
            else:
                self._source_e_offsets[p] = e_off
                e_off += self.sizes[p]
        self._n_total_e = e_off
        self._n_total_i = i_off
        # Accumulators for block-CSR construction. Each element is
        # (rows_global, cols_global, data_weighted).
        block_e_acc: list[
            list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]]
        ] = [[] for _ in range(n_delay_bins)]
        block_i_acc: list[
            list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]]
        ] = [[] for _ in range(n_delay_bins)]
        for ti, target in enumerate(POPULATIONS):
            n_t = self.sizes[target]
            for sj, source in enumerate(POPULATIONS):
                prob = float(CONN_PROBS[ti, sj])
                if prob <= 0.0:
                    continue
                n_s = self.sizes[source]
                if scale_correction:
                    # Per-target in-degree fixed at the full-scale
                    # value; sample with replacement from the scaled
                    # source population. Multapses are intentionally
                    # ALLOWED — measured 2026-04-18 with a vectorised
                    # `argpartition` no-multapse alternative produced
                    # catastrophic rate inflation at scale=0.1
                    # (E populations 50-100× over published Table 4
                    # vs ~2× over with multapses). The mean per-target
                    # weight is identical between the two approaches,
                    # but the no-multapse variant amplifies population
                    # synchrony in the heavy-recurrent regime where
                    # K approaches N_s. Multapse-with-replacement is
                    # the regime that van Albada 2015 actually
                    # validates; it is also what NEST uses by
                    # default at sub-full scale.
                    k_per_target = max(
                        1,
                        int(round(prob * FULL_SIZES[source])),
                    )
                    rows = np.repeat(
                        np.arange(n_t, dtype=np.int32),
                        k_per_target,
                    )
                    cols = self._rng.integers(
                        0,
                        n_s,
                        size=n_t * k_per_target,
                        dtype=np.int32,
                    )
                    data = np.ones(n_t * k_per_target, dtype=np.float32)
                else:
                    # Bernoulli per pair at the literal probability.
                    mask = self._rng.random((n_t, n_s)) < prob
                    rows_t, cols_t = np.nonzero(mask)
                    rows = rows_t.astype(np.int32, copy=False)
                    cols = cols_t.astype(np.int32, copy=False)
                    data = np.ones(rows.size, dtype=np.float32)
                # Build via CSR directly with explicit indptr/indices
                # to dodge the scipy `get_index_dtype` path that
                # invokes the broken `umr_maximum` reduction under
                # NumPy-reload conditions (coverage instrumentation).
                order = np.argsort(rows, kind="stable")
                rows_s = rows[order]
                cols_s = cols[order]
                data_s = data[order]
                indptr = np.zeros(n_t + 1, dtype=np.int32)
                np.add.at(indptr, rows_s + 1, 1)
                np.cumsum(indptr, out=indptr)
                W = sparse.csr_matrix(
                    (data_s, cols_s, indptr),
                    shape=(n_t, n_s),
                )
                # Multapses (duplicate (row, col) pairs) are summed
                # into a single CSR entry by sum_duplicates, giving
                # integer multapse counts in `W.data`.
                W.sum_duplicates()
                self._W[target, source] = W

                # Per-connection delay distribution. Sample one delay
                # per connection from the source-type Gaussian, bin
                # into `n_delay_bins` groups, and either:
                #   - build one sub-CSR per (per-pair) bin (legacy
                #     `use_block_csr=False`), or
                #   - accumulate per-connection global-row / global-
                #     col / weighted-data triples per global bin to
                #     be assembled into block CSRs later
                #     (`use_block_csr=True`).
                if delay_distribution and self.n_delay_bins > 1:
                    if _is_inhibitory(source):
                        d_mean, d_sigma = DELAY_I, DELAY_I_SIGMA
                        global_centers = self._global_i_centers_ms
                    else:
                        d_mean, d_sigma = DELAY_E, DELAY_E_SIGMA
                        global_centers = self._global_e_centers_ms
                    delays_ms = self._rng.normal(
                        d_mean,
                        d_sigma,
                        size=rows_s.size,
                    )
                    # Strictly positive; clip to avoid same-step
                    # algebraic loops (delay must be ≥ 1 dt step at
                    # the smallest dt the caller might pick — we use
                    # 0.05 ms as a conservative floor).
                    delays_ms = np.clip(delays_ms, 0.05, None)

                    if use_block_csr:
                        # Snap each connection's delay to nearest
                        # GLOBAL bin centre for its source type.
                        bin_idx_global = np.argmin(
                            np.abs(
                                delays_ms[:, None] - global_centers[None, :],
                            ),
                            axis=1,
                        )
                        # Effective per-connection weight (baked).
                        if _is_inhibitory(source):
                            weight_per_conn = self.w_i
                            t_offset = self._target_offsets[target]
                            s_offset = self._source_i_offsets[source]
                            acc = block_i_acc
                        else:
                            if source == "L4e" and target == "L23e":
                                weight_per_conn = self.w_l4_to_l23e
                            else:
                                weight_per_conn = self.w_e
                            t_offset = self._target_offsets[target]
                            s_offset = self._source_e_offsets[source]
                            acc = block_e_acc
                        rows_global = rows_s.astype(np.int64) + t_offset
                        cols_global = cols_s.astype(np.int64) + s_offset
                        data_w = data_s * weight_per_conn
                        for b in range(self.n_delay_bins):
                            mask_b = bin_idx_global == b
                            if not mask_b.any():
                                continue
                            acc[b].append(
                                (
                                    rows_global[mask_b].astype(np.int32),
                                    cols_global[mask_b].astype(np.int32),
                                    data_w[mask_b].astype(np.float64),
                                )
                            )
                        # In block mode we do NOT build per-pair
                        # `_W_bins` — block matrices are built once
                        # below outside the pair loop.
                        continue
                    # Quantile-bin the connections by delay.
                    n_bins = self.n_delay_bins
                    quantiles = np.linspace(
                        0.0,
                        1.0,
                        n_bins + 1,
                    )[1:-1]
                    cuts = np.quantile(delays_ms, quantiles)
                    bin_idx = np.searchsorted(cuts, delays_ms)
                    bins_list: list[tuple[float, sparse.csr_matrix]] = []
                    for b in range(n_bins):
                        mask = bin_idx == b
                        if not mask.any():
                            continue
                        # Bin's representative delay (mean over its
                        # member connections, in milliseconds).
                        bin_delay_ms = float(delays_ms[mask].mean())
                        # Build sub-CSR from the bin's rows / cols.
                        rows_b = rows_s[mask]
                        cols_b = cols_s[mask]
                        data_b = data_s[mask]
                        indptr_b = np.zeros(n_t + 1, dtype=np.int32)
                        np.add.at(indptr_b, rows_b + 1, 1)
                        np.cumsum(indptr_b, out=indptr_b)
                        # Re-sort cols within each row so sum_duplicates
                        # produces a canonical CSR.
                        W_b = sparse.csr_matrix(
                            (data_b, cols_b, indptr_b),
                            shape=(n_t, n_s),
                        )
                        W_b.sum_duplicates()
                        bins_list.append((bin_delay_ms, W_b))
                    self._W_bins[target, source] = bins_list

        # Assemble block CSRs (one per (source-type, bin_idx)) from
        # the global-bin accumulators populated above. Each block
        # has shape (n_total, n_total_<source-type>) and already has
        # weights baked in. Per-step `dot()` returns the weighted
        # contribution directly and the per-pair Python loop
        # collapses to one mat-vec per (source-type, bin).
        self._block_e: list[sparse.csr_matrix] = []
        self._block_i: list[sparse.csr_matrix] = []
        # Pre-extracted (indptr, indices, data) triples per block, all
        # in the dtypes the Rust kernel requires (int32 / int32 /
        # float64). Avoiding `np.ascontiguousarray` in the per-step
        # inner loop is what lets the Rust path actually beat scipy
        # — measured 2026-04-18, the per-call cast overhead alone
        # was eating the per-call speedup.
        self._block_e_arrays: list[
            tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]
        ] = []
        self._block_i_arrays: list[
            tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]
        ] = []
        if delay_distribution and use_block_csr:
            for b in range(self.n_delay_bins):
                blk_e = self._stack_block(
                    block_e_acc[b],
                    self.n_total,
                    self._n_total_e,
                )
                self._block_e.append(blk_e)
                self._block_e_arrays.append(
                    (
                        np.ascontiguousarray(blk_e.indptr, dtype=np.int32),
                        np.ascontiguousarray(blk_e.indices, dtype=np.int32),
                        np.ascontiguousarray(blk_e.data, dtype=np.float64),
                    )
                )
                blk_i = self._stack_block(
                    block_i_acc[b],
                    self.n_total,
                    self._n_total_i,
                )
                self._block_i.append(blk_i)
                self._block_i_arrays.append(
                    (
                        np.ascontiguousarray(blk_i.indptr, dtype=np.int32),
                        np.ascontiguousarray(blk_i.indices, dtype=np.int32),
                        np.ascontiguousarray(blk_i.data, dtype=np.float64),
                    )
                )

    # ── Block-CSR assembly helper ────────────────────────────────

    @staticmethod
    def _stack_block(
        triples: list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]],
        n_rows: int,
        n_cols: int,
    ) -> sparse.csr_matrix:
        """Concatenate per-pair (rows, cols, data) into a CSR.

        Empty triples list yields an all-zero shape-correct CSR so
        the per-step `dot()` still returns a well-shaped vector
        (handles the case where a particular bin is unused, e.g.
        when `n_delay_bins=1`).
        """
        if not triples:
            return sparse.csr_matrix(
                (n_rows, n_cols),
                dtype=np.float64,
            )
        rows = np.concatenate([t[0] for t in triples])
        cols = np.concatenate([t[1] for t in triples])
        data = np.concatenate([t[2] for t in triples])
        # Build CSR via explicit indptr to dodge scipy's
        # `get_index_dtype` reduction-path sensitivity to a NumPy
        # reload (same hardening pattern as the per-pair build).
        order = np.argsort(rows, kind="stable")
        rows = rows[order]
        cols = cols[order]
        data = data[order]
        indptr = np.zeros(n_rows + 1, dtype=np.int32)
        np.add.at(indptr, rows + 1, 1)
        np.cumsum(indptr, out=indptr)
        m = sparse.csr_matrix(
            (data, cols.astype(np.int32), indptr),
            shape=(n_rows, n_cols),
        )
        m.sum_duplicates()
        return m
