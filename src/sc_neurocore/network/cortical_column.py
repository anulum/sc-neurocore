# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Potjans & Diesmann 2014 8-population cortical microcircuit

"""Potjans & Diesmann 2014 canonical cortical microcircuit.

Implements the full 8-population cortical column model from
Potjans, T. C. & Diesmann, M. (2014). *The cell-type specific
cortical microcircuit: relating structure and activity in a full-
scale spiking network model.* Cerebral Cortex 24(3): 785-806,
DOI: 10.1093/cercor/bhs358.

Layers and populations
----------------------

| Population | Type        | Full-scale size (Table 5) |
|------------|-------------|---------------------------|
| L23e       | excitatory  | 20 683                    |
| L23i       | inhibitory  |  5 834                    |
| L4e        | excitatory  | 21 915                    |
| L4i        | inhibitory  |  5 479                    |
| L5e        | excitatory  |  4 850                    |
| L5i        | inhibitory  |  1 065                    |
| L6e        | excitatory  | 14 395                    |
| L6i        | inhibitory  |  2 948                    |

Connectivity
------------

The 8×8 connection-probability matrix is taken verbatim from
Potjans & Diesmann 2014 Table 5. `CONN_PROBS[t, s]` is the
probability that a neuron in target population `t` receives an
input from a randomly drawn neuron in source population `s`.
This matches the Binzegger et al. 2004 anatomical estimate that
the paper is built on.

Background input is independent Poisson at `bg_rate = 8 Hz` per
channel; each cell receives `K_bg[pop]` such channels (Table 5).

Synaptic model
--------------

LIF neurons with exponentially decaying current-based PSCs (NEST
`iaf_psc_exp`):

    dV/dt   = -(V - E_L) / tau_m + I_syn / C_m
    dI_syn/dt = -I_syn / tau_syn

Spikes reset `V → V_reset` and trigger a refractory window of
`t_ref = 2 ms` during which the membrane is clamped. Synaptic
delays are 1.5 ms for excitatory sources and 0.8 ms for
inhibitory sources; sub-step delays are quantised to multiples of
`dt`. Excitatory PSC amplitude is `w = 87.81 pA`; inhibitory PSC
amplitude is `w_in = -g · w` with `g = 4`. The L4e → L2/3e edge
is boosted to `2 · w` per Potjans §"Strengthening of the L4e to
L2/3e connection" (matches Hahne et al. 2017).

Scaling
-------

The class accepts a `scale ∈ (0, 1]` factor that multiplies all
population sizes. Connection probabilities are unchanged but per-
cell synaptic weights are inversely scaled to preserve mean drive
(`scale_correction=True` by default; this matches the "full-scale
in-degree" preservation strategy of van Albada et al. 2015).
`scale=1.0` produces ~77 169 neurons; `scale=0.1` ~7717 neurons,
which is the smallest scale where the published asynchronous-
irregular firing rates (Table 4) are reproduced within tolerance.

API surface
-----------

    col = CorticalColumn(scale=0.1, seed=42)
    spikes = col.simulate(duration_ms=1000.0)
    rates = col.population_rates(spikes)
    # rates['L23e'] ≈ 0.86 Hz   etc., matching Potjans 2014 Table 4.

`step(dt)` is exposed for callers that need single-step control
(e.g. closed-loop with external sensory drive). `reset_state()`
re-randomises voltages from the per-instance RNG without rebuilding
the connectivity (which is expensive at full scale).
"""

from __future__ import annotations

import ctypes
import importlib as _importlib
import logging
import os as os
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import sparse

from ._cortical_column_backends import discover_native_backends
from ._cortical_column_connectivity import _CorticalConnectivity
from ._cortical_column_parameters import (
    BG_RATE as BG_RATE,
    CONN_PROBS as CONN_PROBS,
    C_M as C_M,
    DELAY_E as DELAY_E,
    DELAY_I as DELAY_I,
    E_L as E_L,
    FULL_SIZES as FULL_SIZES,
    G_INH as G_INH,
    K_BG as K_BG,
    N_POPS as N_POPS,
    POPULATIONS as POPULATIONS,
    T_REF as T_REF,
    TAU_M as TAU_M,
    TAU_SYN as TAU_SYN,
    V_RESET as V_RESET,
    V_TH as V_TH,
    W_E as W_E,
    W_I as W_I,
    _is_inhibitory,
    population_sizes as _population_sizes,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


_logger = logging.getLogger(__name__)
_native_backends = discover_native_backends(__file__, _logger, _importlib.import_module)
_rust_csr_spmv_add = _native_backends.rust_spmv
_rust_csr_multi_spmv_add = _native_backends.rust_multi_spmv
_julia_multi_spmv = _native_backends.julia_multi_spmv
_go_multi_spmv = _native_backends.go_multi_spmv
_mojo_multi_spmv = _native_backends.mojo_multi_spmv
_HAS_RUST_CSR_SPMV = _rust_csr_spmv_add is not None
_HAS_RUST_CSR_MULTI_SPMV = _rust_csr_multi_spmv_add is not None
_HAS_JULIA_MULTI_SPMV = _julia_multi_spmv is not None
_HAS_GO_MULTI_SPMV = _go_multi_spmv is not None
_HAS_MOJO_MULTI_SPMV = _mojo_multi_spmv is not None

# ── Main microcircuit class ──────────────────────────────────────────


class CorticalColumn(_CorticalConnectivity):
    """Potjans & Diesmann 2014 8-population cortical microcircuit.

    Parameters
    ----------
    scale : float, optional
        Population-size multiplier in (0, 1]. Default 0.1
        (≈ 7700 neurons), which is the smallest size where the
        published Table 4 firing rates are reproduced within
        tolerance. `scale=1.0` yields the full ~77 000-neuron model.
    bg_rate : float, optional
        Background Poisson rate per channel (Hz). Defaults to the
        published 8.0; setting to 0.0 disengages the drive (useful
        for fidelity tests that confirm cells go silent).
    g_inh : float, optional
        Relative inhibitory weight. Defaults to the published 4.0.
    scale_correction : bool, optional
        When True (default), scales per-synapse weights by 1/scale
        so that mean drive per cell is preserved at sub-full scale
        (van Albada et al. 2015). Disable to study finite-size
        effects directly.
    seed : int or None, optional
        Per-instance RNG seed for connectivity, voltages and
        background Poisson. None → fresh entropy.
    """

    def __init__(
        self,
        scale: float = 0.1,
        bg_rate: float = BG_RATE,
        g_inh: float = G_INH,
        scale_correction: bool = True,
        delay_distribution: bool = True,
        n_delay_bins: int = 5,
        use_block_csr: bool = False,
        seed: int | None = None,
        backend: str = "auto",
    ) -> None:
        if not (0.0 < scale <= 1.0):
            raise ValueError(f"scale must be in (0, 1], got {scale}")
        if n_delay_bins < 1:
            raise ValueError(
                f"n_delay_bins must be ≥ 1, got {n_delay_bins}",
            )
        self.scale = scale
        self.bg_rate = bg_rate
        self.g_inh = g_inh
        self.scale_correction = scale_correction
        self.delay_distribution = delay_distribution
        self.n_delay_bins = n_delay_bins
        self.backend = backend

        if self.backend not in {"auto", "rust", "python", "julia", "go", "mojo"}:
            raise ValueError(
                f"backend must be one of 'auto'|'rust'|'python'|'julia'|'go'|'mojo', got {self.backend!r}"
            )
        if self.backend == "rust" and not _HAS_RUST_CSR_MULTI_SPMV:
            raise RuntimeError(
                "backend='rust' requested but `sc_neurocore_engine.py_parallel_csr_multi_spmv_add` is not available"
            )
        if self.backend == "julia" and not _HAS_JULIA_MULTI_SPMV:
            raise RuntimeError("backend='julia' requested but Julia kernel is not available")
        if self.backend == "go" and not _HAS_GO_MULTI_SPMV:
            raise RuntimeError("backend='go' requested but Go kernel is not available")
        if self.backend == "mojo" and not _HAS_MOJO_MULTI_SPMV:
            raise RuntimeError("backend='mojo' requested but Mojo kernel is not available")

        # `use_block_csr=True` constructs stacked block-CSR matrices
        # per (source-type, global-bin) so the per-step inner loop
        # collapses from `n_pairs * n_delay_bins` (~320) sparse
        # mat-vecs to `2 * n_delay_bins` (~10). Bin centres are
        # GLOBAL — derived from theoretical Gaussian quantiles.
        #
        # Measured 2026-04-18 at scale=0.1: block path is ~2× SLOWER
        # in pure Python than the per-pair path because scipy.sparse
        # CSR mat-vec is compute-bound (FLOPs scale with nnz, which
        # is identical between paths) and the per-pair path's tight
        # inner loop wins on cache locality. Default flipped to
        # False; the block path is preserved as opt-in because it
        # is the natural data layout for any future Rust / Mojo
        # FFI port (10 FFI calls per step vs 320), where fixed FFI
        # overhead per call DOES dominate.
        self.use_block_csr = use_block_csr
        self._rng = np.random.default_rng(seed)

        # Per-population scaled sizes (at least 1 cell per pop to
        # keep matrix shapes well-defined at very low scale).
        self.sizes = self.population_sizes(scale)
        self.n_total = sum(self.sizes.values())

        # Per-source-type weight (pA). With `scale_correction=True`
        # we use the full-scale weight unchanged: van Albada 2015
        # in-degree preservation keeps both K and w at full-scale
        # values, with the scaled source population providing the
        # same number of incoming spikes per dt because the spike
        # rate per cell is invariant. Without correction the weights
        # are unchanged but the literal scaled in-degree is used.
        self.w_e = W_E
        self.w_i = -g_inh * W_E
        self.w_l4_to_l23e = 2.0 * self.w_e  # Potjans boost.

        self._build_connectivity()

        # Per-population state arrays.
        self.v: dict[str, np.ndarray[Any, Any]] = {}
        self.i_syn: dict[str, np.ndarray[Any, Any]] = {}
        self.refrac: dict[str, np.ndarray[Any, Any]] = {}
        for p in POPULATIONS:
            n_p = self.sizes[p]
            # Initial voltages distributed uniformly between V_reset
            # and V_th to dephase the population at t=0 and avoid an
            # initial sync transient.
            self.v[p] = self._rng.uniform(V_RESET, V_TH, size=n_p)
            self.i_syn[p] = np.zeros(n_p, dtype=np.float64)
            self.refrac[p] = np.zeros(n_p, dtype=np.float64)

        # Delay-buffer scaffolding. We delay each source population's
        # spike count proxy by either DELAY_E or DELAY_I depending on
        # source type. The buffer length is set at the first `step()`
        # call once `dt` is known (so callers can pick `dt` freely).
        self._dt: float | None = None
        self._buf_e: dict[str, np.ndarray[Any, Any]] = {}
        self._buf_i: dict[str, np.ndarray[Any, Any]] = {}
        self._buf_idx: int = 0
        self._buf_len_e: int = 0
        self._buf_len_i: int = 0
        # Integer-step delay caches for the global block-CSR path,
        # populated in `_init_buffers(dt)` once `dt` is known.
        self._global_e_bin_steps: list[int] = []
        self._global_i_bin_steps: list[int] = []

    @staticmethod
    def population_sizes(scale: float) -> dict[str, int]:
        """Return Potjans population sizes at ``scale`` without building connectivity."""
        return _population_sizes(scale)

    # ── Time-stepping ────────────────────────────────────────────

    def _init_buffers(self, dt: float) -> None:
        # Convert per-bin float delays to integer step counts (≥ 1
        # so spikes never feed back into the same step that produced
        # them, which would create an algebraic loop).
        if self.delay_distribution and self.use_block_csr:
            self._global_e_bin_steps = [
                max(1, int(round(d / dt))) for d in self._global_e_centers_ms
            ]
            self._global_i_bin_steps = [
                max(1, int(round(d / dt))) for d in self._global_i_centers_ms
            ]
            self._buf_len_e = max(self._global_e_bin_steps)
            self._buf_len_i = max(self._global_i_bin_steps)
        elif self.delay_distribution and self._W_bins:
            max_e = 1
            max_i = 1
            for (target, source), bins in self._W_bins.items():
                steps_bins: list[tuple[int, sparse.csr_matrix]] = []
                for delay_ms, W_b in bins:
                    d_steps = max(1, int(round(delay_ms / dt)))
                    steps_bins.append((d_steps, W_b))
                    if _is_inhibitory(source):
                        max_i = max(max_i, d_steps)
                    else:
                        max_e = max(max_e, d_steps)
                self._W_bin_steps[target, source] = steps_bins
            self._buf_len_e = max_e
            self._buf_len_i = max_i
        else:
            self._buf_len_e = max(1, int(round(DELAY_E / dt)))
            self._buf_len_i = max(1, int(round(DELAY_I / dt)))
        for p in POPULATIONS:
            n_p = self.sizes[p]
            self._buf_e[p] = np.zeros((self._buf_len_e, n_p), dtype=np.int32)
            self._buf_i[p] = np.zeros((self._buf_len_i, n_p), dtype=np.int32)
        self._dt = dt

    def step(self, dt: float = 0.1) -> dict[str, np.ndarray[Any, Any]]:
        """Advance the network by one timestep `dt` (ms).

        Returns a dict mapping population name → boolean spike
        vector for that step. The first call fixes `dt`; later
        calls must use the same value or `ValueError` is raised.
        """
        if self._dt is None:
            self._init_buffers(dt)
        elif dt != self._dt:
            raise ValueError(
                f"dt changed mid-run ({self._dt} → {dt}); call "
                f"reset_state() first if you need a different dt"
            )

        # 1. Decay synaptic currents.
        decay = float(np.exp(-dt / TAU_SYN))
        for p in POPULATIONS:
            self.i_syn[p] *= decay

        # 2. Inject delayed spike contributions. The block-CSR fast
        #    path collapses the per-pair Python loop to one mat-vec
        #    per (source-type, bin); the legacy path keeps the
        #    per-pair loop with per-pair quantile bins.
        if self.delay_distribution and self.use_block_csr:
            self._inject_block(dt)
        else:
            self._inject_per_pair(dt)
        return self._integrate_and_detect(dt)

    def _inject_block(self, dt: float) -> None:
        """Inject block-CSR spikes for one step.

        With the batched Rust kernel available, ONE FFI call per step does all `2 × n_delay_bins`
        spmv at once; otherwise falls back to per-block scipy /
        single-Rust calls.
        """
        contrib_concat = np.zeros(self.n_total, dtype=np.float64)
        e_pops = [p for p in POPULATIONS if not _is_inhibitory(p)]
        i_pops = [p for p in POPULATIONS if _is_inhibitory(p)]

        # Gather spike vectors per bin, dropping empty bins so the
        # batched call only ever sees non-trivial work.
        indptrs: list[np.ndarray[Any, Any]] = []
        indices_list: list[np.ndarray[Any, Any]] = []
        data_list: list[np.ndarray[Any, Any]] = []
        xs: list[np.ndarray[Any, Any]] = []

        for b, d_steps in enumerate(self._global_e_bin_steps):
            block = self._block_e[b]
            if block.nnz == 0:
                continue
            idx = (self._buf_idx - d_steps) % self._buf_len_e
            spike_concat = np.concatenate([self._buf_e[p][idx] for p in e_pops]).astype(np.float64)
            if np.count_nonzero(spike_concat) == 0:
                continue
            indptr_b, indices_b, data_b = self._block_e_arrays[b]
            indptrs.append(indptr_b)
            indices_list.append(indices_b)
            data_list.append(data_b)
            xs.append(spike_concat)

        for b, d_steps in enumerate(self._global_i_bin_steps):
            block = self._block_i[b]
            if block.nnz == 0:
                continue
            idx = (self._buf_idx - d_steps) % self._buf_len_i
            spike_concat = np.concatenate([self._buf_i[p][idx] for p in i_pops]).astype(np.float64)
            if np.count_nonzero(spike_concat) == 0:
                continue
            indptr_b, indices_b, data_b = self._block_i_arrays[b]
            indptrs.append(indptr_b)
            indices_list.append(indices_b)
            data_list.append(data_b)
            xs.append(spike_concat)

        if indptrs:
            bnd = self.backend
            if bnd == "auto":
                if _HAS_RUST_CSR_MULTI_SPMV and _rust_csr_multi_spmv_add is not None:
                    bnd = "rust"
                elif _HAS_MOJO_MULTI_SPMV and _mojo_multi_spmv is not None:
                    bnd = "mojo"
                elif _HAS_GO_MULTI_SPMV and _go_multi_spmv is not None:
                    bnd = "go"
                elif _HAS_JULIA_MULTI_SPMV and _julia_multi_spmv is not None:
                    bnd = "julia"
                else:
                    bnd = "python"

            if bnd in ("mojo", "go", "julia"):
                n_blocks = len(indptrs)
                n_rows = contrib_concat.size
                P_INT32 = ctypes.POINTER(ctypes.c_int32)
                P_FLOAT64 = ctypes.POINTER(ctypes.c_double)

                if bnd == "julia" and _julia_multi_spmv is not None:
                    indptr_ptrs_arr = np.array([arr.ctypes.data for arr in indptrs], dtype=np.uintp)
                    indices_ptrs_arr = np.array(
                        [arr.ctypes.data for arr in indices_list], dtype=np.uintp
                    )
                    data_ptrs_arr = np.array([arr.ctypes.data for arr in data_list], dtype=np.uintp)
                    x_ptrs_arr = np.array([arr.ctypes.data for arr in xs], dtype=np.uintp)
                    x_lens_arr = np.array([arr.size for arr in xs], dtype=int)
                    _julia_multi_spmv(
                        n_blocks,
                        n_rows,
                        indptr_ptrs_arr,
                        indices_ptrs_arr,
                        data_ptrs_arr,
                        x_ptrs_arr,
                        x_lens_arr,
                        contrib_concat.ctypes.data,
                    )
                else:
                    indptr_ptrs = (P_INT32 * n_blocks)(
                        *[arr.ctypes.data_as(P_INT32) for arr in indptrs]
                    )
                    indices_ptrs = (P_INT32 * n_blocks)(
                        *[arr.ctypes.data_as(P_INT32) for arr in indices_list]
                    )
                    data_ptrs = (P_FLOAT64 * n_blocks)(
                        *[arr.ctypes.data_as(P_FLOAT64) for arr in data_list]
                    )
                    x_ptrs = (P_FLOAT64 * n_blocks)(*[arr.ctypes.data_as(P_FLOAT64) for arr in xs])
                    x_lens = (ctypes.c_int32 * n_blocks)(*[arr.size for arr in xs])
                    y_ptr = contrib_concat.ctypes.data_as(P_FLOAT64)

                    if bnd == "mojo" and _mojo_multi_spmv is not None:
                        _mojo_multi_spmv(
                            n_blocks,
                            n_rows,
                            indptr_ptrs,
                            indices_ptrs,
                            data_ptrs,
                            x_ptrs,
                            x_lens,
                            y_ptr,
                        )
                    elif bnd == "go" and _go_multi_spmv is not None:  # pragma: no branch
                        _go_multi_spmv(
                            n_blocks,
                            n_rows,
                            indptr_ptrs,
                            indices_ptrs,
                            data_ptrs,
                            x_ptrs,
                            x_lens,
                            y_ptr,
                        )

            elif (
                bnd == "rust" and _HAS_RUST_CSR_MULTI_SPMV and _rust_csr_multi_spmv_add is not None
            ):
                # ONE batched FFI call replaces the up-to-10
                # per-bin calls. Rust loops internally and shares
                # the rayon thread pool across all bins.
                _rust_csr_multi_spmv_add(
                    indptrs,
                    indices_list,
                    data_list,
                    xs,
                    contrib_concat,
                )
            else:
                for indptr_b, indices_b, data_b, x_b in zip(
                    indptrs,
                    indices_list,
                    data_list,
                    xs,
                    strict=True,
                ):
                    if (
                        self.backend != "python"
                        and _HAS_RUST_CSR_SPMV
                        and _rust_csr_spmv_add is not None
                    ):
                        _rust_csr_spmv_add(
                            indptr_b,
                            indices_b,
                            data_b,
                            x_b,
                            contrib_concat,
                        )
                    else:
                        # Pure-scipy fallback: build a temp CSR view
                        # and use scipy dot.
                        contrib_concat += sparse.csr_matrix(
                            (data_b, indices_b, indptr_b),
                            shape=(self.n_total, x_b.size),
                        ).dot(x_b)

        # Slice back into per-target-pop chunks and add background.
        for target in POPULATIONS:
            n_t = self.sizes[target]
            t_off = self._target_offsets[target]
            chunk = contrib_concat[t_off : t_off + n_t]
            if self.bg_rate > 0.0:
                lam = K_BG[target] * self.bg_rate * dt * 1e-3
                bg_kicks = self._rng.poisson(lam, size=n_t)
                chunk = chunk + bg_kicks * self.w_e
            self.i_syn[target] += chunk

    @staticmethod
    def _spmv_into(
        block: sparse.csr_matrix,
        x: np.ndarray[Any, Any],
        y: np.ndarray[Any, Any],
        arrays: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]
        | None = None,
    ) -> None:
        """Compute ``y += block @ x`` via a sparse block matrix-vector product.

        Uses the Rust rayon-parallel kernel
        when available, falls back to scipy single-threaded.

        `arrays` is the dtype-checked `(indptr, indices, data)` triple
        precomputed at construction so the per-step inner loop avoids
        the per-call cast overhead that otherwise eats the per-call
        Rust speedup. Falls back to deriving from `block` if not
        supplied (slow path used by the parity sanity tests).
        """
        if _HAS_RUST_CSR_SPMV and _rust_csr_spmv_add is not None:
            if arrays is None:
                arrays = (
                    np.ascontiguousarray(block.indptr, dtype=np.int32),
                    np.ascontiguousarray(block.indices, dtype=np.int32),
                    np.ascontiguousarray(block.data, dtype=np.float64),
                )
            indptr, indices, data = arrays
            _rust_csr_spmv_add(indptr, indices, data, x, y)
        else:
            y += block.dot(x)

    def _inject_per_pair(self, dt: float) -> None:
        """Legacy per-pair injection: one mat-vec per (pair, bin)."""
        for ti, target in enumerate(POPULATIONS):
            n_t = self.sizes[target]
            contrib = np.zeros(n_t, dtype=np.float64)
            for sj, source in enumerate(POPULATIONS):
                key = (target, source)
                if key not in self._W:
                    continue
                if _is_inhibitory(source):
                    weight = self.w_i
                    buf = self._buf_i
                    buf_len = self._buf_len_i
                else:
                    buf = self._buf_e
                    buf_len = self._buf_len_e
                    # L4e → L2/3e is boosted; everything else is w_e.
                    if source == "L4e" and target == "L23e":
                        weight = self.w_l4_to_l23e
                    else:
                        weight = self.w_e
                if self.delay_distribution and key in self._W_bin_steps:
                    # Per-bin delayed mat-vec.
                    for d_steps, W_b in self._W_bin_steps[key]:
                        idx = (self._buf_idx - d_steps) % buf_len
                        src_spikes = buf[source][idx]
                        # `np.count_nonzero` instead of `.any()` to
                        # dodge the `_NoValue` sentinel reduction path
                        # under coverage NumPy reload.
                        if np.count_nonzero(src_spikes) == 0:
                            continue
                        hits = W_b.dot(src_spikes.astype(np.float32))
                        contrib += hits * weight
                else:
                    # Single-delay legacy path.
                    d_steps = buf_len
                    idx = (self._buf_idx - d_steps) % buf_len
                    src_spikes = buf[source][idx]
                    if np.count_nonzero(src_spikes) == 0:
                        continue
                    hits = self._W[key].dot(src_spikes.astype(np.float32))
                    contrib += hits * weight
            # Background Poisson channels (excitatory, w_e). Each cell
            # gets K_bg independent Poisson channels each at bg_rate.
            if self.bg_rate > 0.0:
                lam = K_BG[target] * self.bg_rate * dt * 1e-3
                bg_kicks = self._rng.poisson(lam, size=n_t)
                contrib += bg_kicks * self.w_e
            self.i_syn[target] += contrib

    def _integrate_and_detect(
        self,
        dt: float,
    ) -> dict[str, np.ndarray[Any, Any]]:
        """Per-population LIF Euler step + spike detect + buffer push."""
        spikes: dict[str, np.ndarray[Any, Any]] = {}
        for p in POPULATIONS:
            in_refrac = self.refrac[p] > 0.0
            # dV/dt = -(V - E_L)/tau_m + I_syn/C_m  (Euler)
            dv = (-(self.v[p] - E_L) / TAU_M + self.i_syn[p] / C_M) * dt
            self.v[p] = np.where(in_refrac, V_RESET, self.v[p] + dv)

            spk = (self.v[p] >= V_TH) & ~in_refrac
            self.v[p] = np.where(spk, V_RESET, self.v[p])
            self.refrac[p] = np.where(spk, T_REF, self.refrac[p])
            # Element-wise clip; `np.where` instead of `np.maximum`
            # because some scipy/numpy versions resolve `np.maximum`
            # through the broken reduction path under coverage reload.
            new_refrac = self.refrac[p] - dt
            self.refrac[p] = np.where(new_refrac > 0.0, new_refrac, 0.0)
            spikes[p] = spk

            # Push into per-source-type delay buffer.
            if _is_inhibitory(p):
                self._buf_i[p][self._buf_idx % self._buf_len_i] = spk.astype(np.int32)
            else:
                self._buf_e[p][self._buf_idx % self._buf_len_e] = spk.astype(np.int32)

        self._buf_idx += 1
        return spikes

    def simulate(
        self,
        duration_ms: float,
        dt: float = 0.1,
    ) -> dict[str, np.ndarray[Any, Any]]:
        """Run the network for `duration_ms` ms.

        Returns a dict mapping population name → boolean
        `(n_steps, n_pop)` spike raster.
        """
        n_steps = int(round(duration_ms / dt))
        if n_steps <= 0:
            raise ValueError(f"duration_ms / dt must be ≥ 1, got {n_steps}")
        rasters: dict[str, list[np.ndarray[Any, Any]]] = {p: [] for p in POPULATIONS}
        for _ in range(n_steps):
            spikes = self.step(dt=dt)
            for p in POPULATIONS:
                rasters[p].append(spikes[p])
        return {p: np.asarray(rasters[p], dtype=bool) for p in POPULATIONS}

    # ── Analysis helpers ─────────────────────────────────────────

    def population_rates(
        self,
        rasters: dict[str, np.ndarray[Any, Any]],
        dt: float = 0.1,
        burn_in_ms: float = 200.0,
    ) -> dict[str, float]:
        """Return mean firing rate (Hz) per population.

        Drops the initial `burn_in_ms` to remove the construction
        transient. Computed as
        `(spikes after burn-in) / (n_cells · seconds-after-burn-in)`.
        """
        n_burn = int(round(burn_in_ms / dt))
        rates: dict[str, float] = {}
        for p in POPULATIONS:
            arr = rasters[p][n_burn:]
            if arr.size == 0:
                rates[p] = 0.0
                continue
            seconds = arr.shape[0] * dt * 1e-3
            n_cells = max(1, arr.shape[1])
            n_spikes = int(np.count_nonzero(arr))
            rates[p] = n_spikes / (n_cells * seconds)
        return rates

    def total_indegree(self, target: str) -> int:
        """Return mean total synaptic in-degree of a target cell.

        Useful for verification: should approximate
        `sum_s p[t,s] · N_s_full` per Potjans Table 5 when
        `scale_correction=True`.
        """
        n_t = self.sizes[target]
        total = 0
        for source in POPULATIONS:
            key = (target, source)
            if key not in self._W:
                continue
            # Per-target indegree = sum of multapse weights along the row.
            # `nnz` (raw nonzero count) under-counts multapses; the
            # CSR `data` array carries the actual multapse counts.
            total += int(np.add.reduce(self._W[key].data))
        return total // max(1, n_t)

    def reset_state(self) -> None:
        """Re-randomise voltages, currents and refractory state.

        Connectivity (`self._K`) is preserved — rebuilding it costs
        O(N²) at full scale and would defeat the point.
        """
        for p in POPULATIONS:
            n_p = self.sizes[p]
            self.v[p] = self._rng.uniform(V_RESET, V_TH, size=n_p)
            self.i_syn[p][:] = 0.0
            self.refrac[p][:] = 0.0
        # Drop delay buffers and per-bin step caches — caller may
        # pick a different dt, and the bin step counts are dt-derived.
        self._dt = None
        self._buf_e.clear()
        self._buf_i.clear()
        self._buf_idx = 0
        self._buf_len_e = 0
        self._buf_len_i = 0
        self._W_bin_steps.clear()

    # ── Introspection ────────────────────────────────────────────

    def __repr__(self) -> str:
        """Return a concise debug representation of the cortical column."""
        return (
            f"CorticalColumn(scale={self.scale}, n_total={self.n_total}, "
            f"bg_rate={self.bg_rate} Hz, g_inh={self.g_inh})"
        )

    @property
    def population_names(self) -> Sequence[str]:
        """Return the ordered cortical population names."""
        return POPULATIONS
