# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Projection: synaptic connectivity with CSR storage and delay buffer

"""Projection: synaptic connectivity with CSR storage and delay buffer."""

from __future__ import annotations

import numpy as np

from . import topology as _topo


def _csr_matvec(indptr, indices, data, x, n_out):
    """CSR matrix-vector product: result[j] += data[k] * x[i] for each (i,j)."""
    out = np.zeros(n_out, dtype=np.float64)
    n_rows = len(indptr) - 1
    for i in range(n_rows):
        if x[i] == 0:
            continue
        for k in range(indptr[i], indptr[i + 1]):
            out[indices[k]] += data[k] * x[i]
    return out


class Projection:
    """Synaptic projection from source to target population."""

    TOPOLOGY_MAP = {
        "random": _topo.random_connectivity,
        "all_to_all": _topo.all_to_all,
        "ring": _topo.ring_topology,
        "small_world": _topo.small_world,
        "scale_free": _topo.scale_free,
    }

    def __init__(
        self,
        source,
        target,
        weight,
        probability=1.0,
        delay=0.0,
        topology="random",
        plasticity=None,
        seed=42,
    ):
        """Create projection with CSR connectivity and optional delay/plasticity."""
        self.source = source
        self.target = target
        self.weight = weight
        self.delay = delay
        self.plasticity = plasticity
        self.seed = seed

        self.indptr, self.indices, self.data = self._build_connectivity(topology, probability, seed)

        self._delay_steps = max(1, int(round(delay))) if delay > 0 else 0
        if self._delay_steps > 0:
            self._delay_buf = np.zeros((self._delay_steps, target.n), dtype=np.float64)
            self._delay_idx = 0
        else:
            self._delay_buf = None

        if plasticity == "stdp":
            self._pre_trace = np.zeros(source.n, dtype=np.float64)
            self._post_trace = np.zeros(target.n, dtype=np.float64)

    def _build_connectivity(self, topology, probability, seed):
        """Build CSR arrays from topology name or pre-built tuple."""
        if isinstance(topology, tuple) and len(topology) == 3:
            return topology
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

    def propagate(self, source_spikes) -> np.ndarray:
        """Compute target currents from source spikes through CSR connectivity."""
        current = _csr_matvec(self.indptr, self.indices, self.data, source_spikes, self.target.n)
        if self._delay_buf is not None:
            output = self._delay_buf[self._delay_idx].copy()
            self._delay_buf[self._delay_idx] = current
            self._delay_idx = (self._delay_idx + 1) % self._delay_steps
            return output
        return current

    def update_plasticity(self, src_spikes, tgt_spikes, a_plus=0.01, a_minus=0.012, tau=20.0):
        """Trace-based STDP weight update."""
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
                    self.data[k] += a_plus * self._pre_trace[i]
                self.data[k] = max(0.0, self.data[k])
