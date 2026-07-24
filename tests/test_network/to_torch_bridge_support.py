# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_to_torch_bridge.py

from __future__ import annotations

"""Support extracted from test_to_torch_bridge.py."""

from typing import Any, cast


import numpy as np


import pytest


import torch


from sc_neurocore.network._torch_bridge import NetworkTorchBridge


from sc_neurocore.network import Network, Population, Projection


from sc_neurocore.neurons.stochastic_lif import StochasticLIFNeuron


from sc_neurocore.training.surrogate import atan_surrogate_custom_op


def _all_to_all_topology(
    n_src: int, n_tgt: int, weight: float
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    indptr = np.arange(0, n_src * n_tgt + 1, n_tgt, dtype=np.int64)
    indices = np.tile(np.arange(n_tgt, dtype=np.int64), n_src)
    data = np.full(n_src * n_tgt, weight, dtype=np.float64)
    return indptr, indices, data


def _manual_counts(
    inputs: torch.Tensor,
    proj_in_hid: Projection,
    proj_hid_out: Projection,
) -> torch.Tensor:
    inputs_np = inputs.detach().cpu().numpy()
    batch = inputs_np.shape[1]
    counts = np.zeros((batch, proj_hid_out.target.n), dtype=np.float32)

    for batch_idx in range(batch):
        src = Population("LapicqueNeuron", 2, params={"tau": 5.0, "dt": 1.0}, label="src")
        hid = Population("LapicqueNeuron", 3, params={"tau": 5.0, "dt": 1.0}, label="hid")
        out = Population("LapicqueNeuron", 2, params={"tau": 5.0, "dt": 1.0}, label="out")
        proj1 = Projection(src, hid, weight=0.0, topology=_all_to_all_topology(2, 3, 0.0))
        proj2 = Projection(hid, out, weight=0.0, topology=_all_to_all_topology(3, 2, 0.0))
        proj1.data[:] = proj_in_hid.data
        proj2.data[:] = proj_hid_out.data

        last_src = np.zeros(src.n, dtype=np.int8)
        last_hid = np.zeros(hid.n, dtype=np.int8)
        for t in range(inputs_np.shape[0]):
            src_spikes = src.step_all(inputs_np[t, batch_idx])
            hid_current = proj1.propagate(last_src)
            hid_spikes = hid.step_all(hid_current)
            out_current = proj2.propagate(last_hid)
            out_spikes = out.step_all(out_current)
            counts[batch_idx] += out_spikes.astype(np.float32)
            last_src = src_spikes
            last_hid = hid_spikes

    return torch.from_numpy(counts)


def _projection_edge_values_from_bridge(
    bridge: Any, projection: Projection, name: str
) -> np.ndarray[Any, Any]:
    dense = getattr(bridge, f"{name}_weight").detach().cpu().numpy()
    values = []
    for src_idx in range(projection.source.n):
        for k in range(projection.indptr[src_idx], projection.indptr[src_idx + 1]):
            tgt_idx = projection.indices[k]
            values.append(dense[tgt_idx, src_idx])
    return np.asarray(values, dtype=np.float64)


__all__ = [
    "Any",
    "cast",
    "np",
    "pytest",
    "torch",
    "NetworkTorchBridge",
    "Network",
    "Population",
    "Projection",
    "StochasticLIFNeuron",
    "atan_surrogate_custom_op",
    "_all_to_all_topology",
    "_manual_counts",
    "_projection_edge_values_from_bridge",
]
