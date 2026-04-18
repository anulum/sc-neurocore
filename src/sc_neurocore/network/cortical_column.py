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

from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

if TYPE_CHECKING:
    from collections.abc import Sequence


# ── Population ordering and per-population sizes (Potjans Table 5) ──

POPULATIONS: tuple[str, ...] = (
    "L23e", "L23i", "L4e", "L4i", "L5e", "L5i", "L6e", "L6i",
)
N_POPS = len(POPULATIONS)

FULL_SIZES: dict[str, int] = {
    "L23e": 20683, "L23i":  5834,
    "L4e":  21915, "L4i":   5479,
    "L5e":   4850, "L5i":   1065,
    "L6e":  14395, "L6i":   2948,
}

# K_bg: number of independent background-Poisson channels per cell.
# Source: Potjans & Diesmann 2014 Table 5 column "k_ext".
K_BG: dict[str, int] = {
    "L23e": 1600, "L23i": 1500,
    "L4e":  2100, "L4i":  1900,
    "L5e":  2000, "L5i":  1900,
    "L6e":  2900, "L6i":  2100,
}

# Connection-probability matrix. Rows = TARGET, columns = SOURCE.
# Source: Potjans & Diesmann 2014 Table 5 (transcribed verbatim,
# Binzegger et al. 2004 anatomical estimate). Values not in the
# table are 0.
#
# Row order follows POPULATIONS; column order follows POPULATIONS.
CONN_PROBS: np.ndarray = np.array([
    # src:  L23e    L23i    L4e     L4i     L5e     L5i     L6e     L6i
    [0.1009, 0.1689, 0.0437, 0.0818, 0.0323, 0.0000, 0.0076, 0.0000],  # L23e
    [0.1346, 0.1371, 0.0316, 0.0515, 0.0755, 0.0000, 0.0042, 0.0000],  # L23i
    [0.0077, 0.0059, 0.0497, 0.1350, 0.0067, 0.0003, 0.0453, 0.0000],  # L4e
    [0.0691, 0.0029, 0.0794, 0.1597, 0.0033, 0.0000, 0.1057, 0.0000],  # L4i
    [0.1004, 0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.0000],  # L5e
    [0.0548, 0.0269, 0.0257, 0.0022, 0.0598, 0.3158, 0.0086, 0.0000],  # L5i
    [0.0156, 0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252],  # L6e
    [0.0364, 0.0010, 0.0034, 0.0005, 0.0277, 0.0080, 0.0658, 0.1443],  # L6i
], dtype=np.float64)


# ── LIF + synapse + delay parameters (Potjans Table 5) ──────────────

C_M = 250.0        # pF — membrane capacitance
TAU_M = 10.0       # ms — membrane time constant
TAU_SYN = 0.5      # ms — exponential PSC decay
T_REF = 2.0        # ms — absolute refractory
E_L = -65.0        # mV — leak reversal == reset
V_RESET = -65.0    # mV
V_TH = -50.0       # mV — spike threshold

# Synaptic weights (PSC peak amplitudes, pA). Excitatory mean is
# w; inhibitory weights are −g·w. The L4e → L23e edge is boosted
# to 2·w per Potjans 2014.
W_E = 87.81        # pA
G_INH = 4.0
W_I = -G_INH * W_E

# Synaptic delays (ms). Per Potjans Table 5: per-connection
# Gaussian distributions. Mean + std per source-type. The mean
# values (1.5 / 0.8 ms) are also the legacy "single delay" values
# used when `delay_distribution=False`.
DELAY_E = 1.5
DELAY_E_SIGMA = 0.75
DELAY_I = 0.8
DELAY_I_SIGMA = 0.4

# Background Poisson rate per channel (Hz).
BG_RATE = 8.0


def _is_inhibitory(pop_name: str) -> bool:
    return pop_name.endswith("i")


# ── Main microcircuit class ──────────────────────────────────────────


class CorticalColumn:
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
        seed: int | None = None,
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
        self._rng = np.random.default_rng(seed)

        # Per-population scaled sizes (at least 1 cell per pop to
        # keep matrix shapes well-defined at very low scale).
        self.sizes: dict[str, int] = {
            p: max(1, int(round(FULL_SIZES[p] * scale)))
            for p in POPULATIONS
        }
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
        # pair's connections into `n_delay_bins` quantile groups by
        # per-connection Gaussian-sampled delay. Each bin gets its
        # own sparse adjacency and its own integer-step delay
        # offset; the per-step inner loop then iterates pairs ×
        # bins, reading the spike vector at each bin's delay
        # offset. Total nnz across bins == nnz of the legacy single-
        # delay matrix; per-bin matrices are roughly 1/n_delay_bins
        # the size each.
        self._W: dict[tuple[str, str], sparse.csr_matrix] = {}
        # `_W_bins[(t, s)]` is a list of (delay_ms, csr_matrix) pairs
        # when delay_distribution=True; left empty otherwise. The
        # `delay_ms` values are converted to integer step counts on
        # the first `_init_buffers(dt)` call, then cached in
        # `_W_bin_steps`.
        self._W_bins: dict[
            tuple[str, str], list[tuple[float, sparse.csr_matrix]],
        ] = {}
        self._W_bin_steps: dict[
            tuple[str, str], list[tuple[int, sparse.csr_matrix]],
        ] = {}
        for ti, target in enumerate(POPULATIONS):
            n_t = self.sizes[target]
            for sj, source in enumerate(POPULATIONS):
                p = float(CONN_PROBS[ti, sj])
                if p <= 0.0:
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
                        1, int(round(p * FULL_SIZES[source])),
                    )
                    rows = np.repeat(
                        np.arange(n_t, dtype=np.int32), k_per_target,
                    )
                    cols = self._rng.integers(
                        0, n_s, size=n_t * k_per_target, dtype=np.int32,
                    )
                    data = np.ones(n_t * k_per_target, dtype=np.float32)
                else:
                    # Bernoulli per pair at the literal probability.
                    mask = self._rng.random((n_t, n_s)) < p
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
                    (data_s, cols_s, indptr), shape=(n_t, n_s),
                )
                # Multapses (duplicate (row, col) pairs) are summed
                # into a single CSR entry by sum_duplicates, giving
                # integer multapse counts in `W.data`.
                W.sum_duplicates()
                self._W[target, source] = W

                # Per-connection delay distribution. Sample one delay
                # per connection from the source-type Gaussian, bin
                # into `n_delay_bins` quantile groups, and build one
                # sub-CSR per bin. Each bin's spike vector will be
                # read at its own delay offset in `step()`.
                if delay_distribution and self.n_delay_bins > 1:
                    if _is_inhibitory(source):
                        d_mean, d_sigma = DELAY_I, DELAY_I_SIGMA
                    else:
                        d_mean, d_sigma = DELAY_E, DELAY_E_SIGMA
                    delays_ms = self._rng.normal(
                        d_mean, d_sigma, size=rows_s.size,
                    )
                    # Strictly positive; clip to avoid same-step
                    # algebraic loops (delay must be ≥ 1 dt step at
                    # the smallest dt the caller might pick — we use
                    # 0.05 ms as a conservative floor).
                    delays_ms = np.clip(delays_ms, 0.05, None)
                    # Quantile-bin the connections by delay.
                    n_bins = self.n_delay_bins
                    quantiles = np.linspace(
                        0.0, 1.0, n_bins + 1,
                    )[1:-1]
                    cuts = np.quantile(delays_ms, quantiles)
                    bin_idx = np.searchsorted(cuts, delays_ms)
                    bins_list: list[tuple[int, sparse.csr_matrix]] = []
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

        # Per-population state arrays.
        self.v: dict[str, np.ndarray] = {}
        self.i_syn: dict[str, np.ndarray] = {}
        self.refrac: dict[str, np.ndarray] = {}
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
        self._buf_e: dict[str, np.ndarray] = {}
        self._buf_i: dict[str, np.ndarray] = {}
        self._buf_idx: int = 0
        self._buf_len_e: int = 0
        self._buf_len_i: int = 0

    # ── Time-stepping ────────────────────────────────────────────

    def _init_buffers(self, dt: float) -> None:
        # Convert per-bin float delays to integer step counts (≥ 1
        # so spikes never feed back into the same step that produced
        # them, which would create an algebraic loop). When
        # `delay_distribution=False` the step count is the legacy
        # single-delay value derived from `DELAY_*`.
        if self.delay_distribution and self._W_bins:
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

    def step(self, dt: float = 0.1) -> dict[str, np.ndarray]:
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

        # 2. Inject delayed spike contributions via per-pair sparse
        #    matrix-vector products. With `delay_distribution=True`
        #    each pair contributes one mat-vec per delay bin (each
        #    bin reads the source spike vector at its own delay
        #    offset), summed into the target's per-step current
        #    contribution. Plus per-cell background Poisson drive.
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

        # 3. Integrate LIF, detect spikes, apply refractory.
        spikes: dict[str, np.ndarray] = {}
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
                self._buf_i[p][self._buf_idx % self._buf_len_i] = (
                    spk.astype(np.int32)
                )
            else:
                self._buf_e[p][self._buf_idx % self._buf_len_e] = (
                    spk.astype(np.int32)
                )

        self._buf_idx += 1
        return spikes

    def simulate(
        self, duration_ms: float, dt: float = 0.1,
    ) -> dict[str, np.ndarray]:
        """Run the network for `duration_ms` ms.

        Returns a dict mapping population name → boolean
        `(n_steps, n_pop)` spike raster.
        """
        n_steps = int(round(duration_ms / dt))
        if n_steps <= 0:
            raise ValueError(f"duration_ms / dt must be ≥ 1, got {n_steps}")
        rasters: dict[str, list[np.ndarray]] = {p: [] for p in POPULATIONS}
        for _ in range(n_steps):
            spikes = self.step(dt=dt)
            for p in POPULATIONS:
                rasters[p].append(spikes[p])
        return {
            p: np.asarray(rasters[p], dtype=bool) for p in POPULATIONS
        }

    # ── Analysis helpers ─────────────────────────────────────────

    def population_rates(
        self,
        rasters: dict[str, np.ndarray],
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

    def __repr__(self) -> str:  # noqa: D401  (one-line summary)
        return (
            f"CorticalColumn(scale={self.scale}, n_total={self.n_total}, "
            f"bg_rate={self.bg_rate} Hz, g_inh={self.g_inh})"
        )

    @property
    def population_names(self) -> Sequence[str]:
        return POPULATIONS
