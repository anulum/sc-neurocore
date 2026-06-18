# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Projection: synaptic connectivity with CSR storage and delay buffer

"""Projection: synaptic connectivity with per-synapse delay support.

Supports three delay modes:
  - No delay (delay=0): direct propagation, no buffering.
  - Uniform axonal delay (delay=scalar): single circular buffer on
    aggregated current. All synapses in the projection share one delay.
  - Per-synapse delay (delay=array): each connection has its own delay.
    Source spike history is stored and each synapse reads from the
    appropriate past timestep. Enables learnable delays (Masquelier
    DelRec, Hammouamri et al. 2023).
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from . import topology as _topo
from .population import Population


def _csr_matvec(
    indptr: np.ndarray[Any, Any],
    indices: np.ndarray[Any, Any],
    data: np.ndarray[Any, Any],
    x: np.ndarray[Any, Any],
    n_out: int,
    weight_threshold: float = 0.0,
) -> np.ndarray[Any, Any]:
    """CSR matrix-vector product: result[j] += data[k] * x[i] for each (i,j).

    Skips source neurons with x[i]==0 (spike-driven) and optionally
    skips synapses with |data[k]| <= weight_threshold (sparse weights).
    """
    out = np.zeros(n_out, dtype=np.float64)
    n_rows = len(indptr) - 1
    for i in range(n_rows):
        if x[i] == 0:
            continue
        for k in range(indptr[i], indptr[i + 1]):
            if weight_threshold > 0.0 and abs(data[k]) <= weight_threshold:
                continue
            out[indices[k]] += data[k] * x[i]
    return out


def _csr_delayed_matvec(
    indptr: np.ndarray[Any, Any],
    indices: np.ndarray[Any, Any],
    data: np.ndarray[Any, Any],
    delay_steps: np.ndarray[Any, Any],
    spike_history: np.ndarray[Any, Any],
    hist_idx: int,
    n_out: int,
) -> np.ndarray[Any, Any]:
    """CSR matrix-vector product with per-synapse delays.

    For each synapse k connecting source i to target j with delay d_k:
        current[j] += data[k] * spike_history[(hist_idx - d_k) % max_delay, i]
    """
    out = np.zeros(n_out, dtype=np.float64)
    max_delay = spike_history.shape[0]
    n_rows = len(indptr) - 1
    for i in range(n_rows):
        for k in range(indptr[i], indptr[i + 1]):
            d = delay_steps[k]
            read_idx = (hist_idx - d) % max_delay
            spike_val = spike_history[read_idx, i]
            if spike_val == 0:
                continue
            out[indices[k]] += data[k] * spike_val
    return out


def validate_csr_topology(
    indptr: np.ndarray[Any, Any],
    indices: np.ndarray[Any, Any],
    data: np.ndarray[Any, Any],
    n_source: int,
    n_target: int,
    *,
    context: str = "Projection",
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Validate and normalize CSR connectivity arrays."""
    indptr = np.asarray(indptr)
    indices = np.asarray(indices)
    data = np.asarray(data)

    if indptr.ndim != 1:
        raise ValueError(f"{context} indptr must be 1-D")
    if indices.ndim != 1:
        raise ValueError(f"{context} indices must be 1-D")
    if data.ndim != 1:
        raise ValueError(f"{context} data must be 1-D")
    if not np.issubdtype(indptr.dtype, np.integer):
        raise ValueError(f"{context} indptr must contain integers")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError(f"{context} indices must contain integers")
    if indptr.shape[0] != n_source + 1:
        raise ValueError(f"{context} indptr length is invalid")
    if indices.shape[0] != data.shape[0]:
        raise ValueError(f"{context} indices/data lengths differ")
    if indptr[0] != 0:
        raise ValueError(f"{context} indptr must start at 0")
    if not np.all(indptr[1:] >= indptr[:-1]):
        raise ValueError(f"{context} indptr must be monotonic")
    if int(indptr[-1]) != indices.shape[0]:
        raise ValueError(f"{context} indptr terminal length is invalid")
    if indices.size and (np.any(indices < 0) or np.any(indices >= n_target)):
        raise ValueError(f"{context} indices are out of bounds")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{context} data must contain only finite values")

    return (
        indptr.astype(np.int64, copy=False),
        indices.astype(np.int64, copy=False),
        data.astype(np.float64, copy=False),
    )


class Projection:
    """Synaptic projection from source to target population.

    Parameters
    ----------
    delay : float, array-like, or 0
        - 0: no delay (default)
        - scalar > 0: uniform axonal delay (all synapses share one delay)
        - 1-D array of length n_synapses: per-synapse delay in timesteps.
          Enables heterogeneous axonal/synaptic delays.
    """

    TOPOLOGY_MAP = {
        "random": _topo.random_connectivity,
        "all_to_all": _topo.all_to_all,
        "ring": _topo.ring_topology,
        "small_world": _topo.small_world,
        "scale_free": _topo.scale_free,
    }

    def __init__(
        self,
        source: Population,
        target: Population,
        weight: float,
        probability: float = 1.0,
        delay: float | np.ndarray[Any, Any] = 0.0,
        topology: str
        | tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]] = "random",
        plasticity: str | None = None,
        seed: int = 42,
        weight_threshold: float = 0.0,
    ) -> None:
        """Create projection with CSR connectivity and optional delay/plasticity.

        *weight_threshold*: skip synapses with |w| <= threshold during
        propagation. Set > 0 to exploit weight sparsity after pruning.
        """
        self.source = source
        self.target = target
        self.weight = weight
        self.plasticity = plasticity
        self.seed = seed
        self.weight_threshold = weight_threshold
        self.delay: float | np.ndarray[Any, Any] = 0.0
        self._delay_mode: str = "none"
        self._delay_buf: np.ndarray[Any, Any] | None = None
        self._per_syn_delays: np.ndarray[Any, Any] | None = None

        self.indptr, self.indices, self.data = self._build_connectivity(topology, probability, seed)

        self._init_delays(delay)

        if plasticity == "stdp":
            self._pre_trace = np.zeros(source.n, dtype=np.float64)
            self._post_trace = np.zeros(target.n, dtype=np.float64)

    def _init_delays(self, delay: float | np.ndarray[Any, Any]) -> None:
        """Set up delay buffers based on delay specification."""
        delay = np.atleast_1d(np.asarray(delay, dtype=np.float64)).flatten()
        n_synapses = len(self.data)

        if delay.size == 1 and delay[0] == 0.0:
            # No delay
            self._delay_mode = "none"
            self.delay = 0.0
            self._delay_buf = None
            self._per_syn_delays = None
            return

        if delay.size == 1:
            # Uniform axonal delay
            self._delay_mode = "uniform"
            self.delay = float(delay[0])
            steps = max(1, int(round(self.delay)))
            self._delay_buf = np.zeros((steps, self.target.n), dtype=np.float64)
            self._delay_idx = 0
            self._delay_steps_uniform = steps
            self._per_syn_delays = None
            return

        # Per-synapse delays
        if delay.size != n_synapses:
            raise ValueError(
                f"Per-synapse delay array length ({delay.size}) must match "
                f"number of connections ({n_synapses})"
            )
        self._delay_mode = "per_synapse"
        self.delay = delay
        self._per_syn_delays = np.round(delay).astype(np.int64)
        self._per_syn_delays = np.clip(self._per_syn_delays, 0, None)
        max_d = int(self._per_syn_delays.max()) + 1
        # Spike history ring buffer: (max_delay+1, n_source)
        self._spike_history = np.zeros((max_d, self.source.n), dtype=np.float64)
        self._hist_idx = 0
        self._delay_buf = None

    @property
    def n_synapses(self) -> int:
        """Number of synaptic connections."""
        return len(self.data)

    @property
    def delay_mode(self) -> str:
        """Delay mode: 'none', 'uniform', or 'per_synapse'."""
        return self._delay_mode

    @property
    def max_delay(self) -> int:
        """Maximum delay in timesteps across all synapses."""
        if self._delay_mode == "none":
            return 0
        if self._delay_mode == "uniform":
            return self._delay_steps_uniform
        if self._per_syn_delays is None:
            raise RuntimeError("Projection per-synapse delay state is not initialized")
        return int(self._per_syn_delays.max())

    def _build_connectivity(
        self,
        topology: str | tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]],
        probability: float,
        seed: int,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Build CSR arrays from topology name or pre-built tuple."""
        if isinstance(topology, tuple) and len(topology) == 3:
            return validate_csr_topology(
                topology[0],
                topology[1],
                topology[2],
                self.source.n,
                self.target.n,
            )
        if topology == "random":
            return _topo.random_connectivity(
                self.source.n, self.target.n, probability, self.weight, seed
            )
        if topology == "all_to_all":
            return _topo.all_to_all(self.source.n, self.target.n, self.weight)
        if topology in ("ring", "small_world", "scale_free"):
            raise ValueError(
                f"Topology '{topology}' requires same-size source/target; "
                "pass pre-built CSR tuple instead."
            )
        raise ValueError(f"Unknown topology '{topology}'")

    def propagate(self, source_spikes: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Compute target currents from source spikes through CSR connectivity.

        Handles three delay modes:
        - none: direct CSR matvec
        - uniform: aggregated current through circular buffer
        - per_synapse: each synapse reads from spike history at its own delay
        """
        wt = self.weight_threshold
        if self._delay_mode == "none":
            return _csr_matvec(
                self.indptr, self.indices, self.data, source_spikes, self.target.n, wt
            )

        if self._delay_mode == "uniform":
            delay_buf = self._delay_buf
            if delay_buf is None:
                raise RuntimeError("Projection uniform delay buffer is not initialized")
            current = _csr_matvec(
                self.indptr, self.indices, self.data, source_spikes, self.target.n, wt
            )
            output = cast(np.ndarray[Any, Any], delay_buf[self._delay_idx].copy())
            delay_buf[self._delay_idx] = current
            self._delay_idx = (self._delay_idx + 1) % self._delay_steps_uniform
            return output

        # Per-synapse delay
        if self._per_syn_delays is None:
            raise RuntimeError("Projection per-synapse delay state is not initialized")
        self._spike_history[self._hist_idx] = source_spikes.astype(np.float64)
        current = _csr_delayed_matvec(
            self.indptr,
            self.indices,
            self.data,
            self._per_syn_delays,
            self._spike_history,
            self._hist_idx,
            self.target.n,
        )
        self._hist_idx = (self._hist_idx + 1) % self._spike_history.shape[0]
        return current

    def update_plasticity(
        self,
        src_spikes: np.ndarray[Any, Any],
        tgt_spikes: np.ndarray[Any, Any],
        a_plus: float = 0.01,
        a_minus: float = 0.012,
        tau: float = 20.0,
        directional_bias: float = 1.0,
    ) -> None:
        """Trace-based STDP weight update.

        *directional_bias* scales a_plus for this projection. Set to 1.36
        for bottom-up (sensory → higher) projections per quantum-control
        NB19 (measured autonomic → cortical asymmetry = 0.36).
        Default 1.0 = symmetric learning.
        """
        if self.plasticity != "stdp":
            return
        decay = np.exp(-1.0 / tau)
        self._pre_trace = self._pre_trace * decay + src_spikes.astype(np.float64)
        self._post_trace = self._post_trace * decay + tgt_spikes.astype(np.float64)

        n_src = self.source.n
        for i in range(n_src):
            for k in range(self.indptr[i], self.indptr[i + 1]):
                j = self.indices[k]
                if src_spikes[i]:
                    self.data[k] -= a_minus * self._post_trace[j]
                if tgt_spikes[j]:
                    self.data[k] += a_plus * directional_bias * self._pre_trace[i]
                self.data[k] = max(0.0, self.data[k])

        # Enforce K symmetry for self-projections (same source and target).
        # Gradient/STDP updates break W = W^T after ~30 steps (SPO Finding #7).
        # Asymmetric coupling hurts sync by +12% (quantum-control NB24).
        if self.source is self.target:
            self._enforce_symmetry()

    def _enforce_symmetry(self) -> None:
        """Symmetrise CSR weight data: W_ij = W_ji = (W_ij + W_ji) / 2.

        Only meaningful for self-projections (source is target, square matrix).
        Iterates over non-zero entries and averages with their transpose.
        """
        n = self.source.n
        for i in range(n):
            for k in range(self.indptr[i], self.indptr[i + 1]):
                j = self.indices[k]
                if j <= i:
                    continue
                # Find reverse edge j→i
                for k2 in range(self.indptr[j], self.indptr[j + 1]):
                    if self.indices[k2] == i:
                        avg = (self.data[k] + self.data[k2]) / 2.0
                        self.data[k] = avg
                        self.data[k2] = avg
                        break
