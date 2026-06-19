# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Lazarus protocol: save/restore spiking network state

"""Lazarus protocol: save/restore complete spiking network state.

Stores membrane potentials, synaptic weights, spike history,
plasticity traces, and metadata in .npz for compact storage.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np


class Checkpoint:
    """Save and restore complete IdentitySubstrate state."""

    @staticmethod
    def save(substrate: Any, path: str) -> None:
        """Save complete state to .npz file.

        Captures: population voltages, projection CSR arrays,
        STDP traces, spike history, and metadata.
        """
        cortical_v = substrate.cortical.voltages.copy()
        inhibitory_v = substrate.inhibitory.voltages.copy()
        memory_v = substrate.memory.voltages.copy()

        ee_indptr = substrate.proj_ee.indptr.copy()
        ee_indices = substrate.proj_ee.indices.copy()
        ee_data = substrate.proj_ee.data.copy()

        ei_data = substrate.proj_ei.data.copy()
        ie_data = substrate.proj_ie.data.copy()
        em_data = substrate.proj_em.data.copy()
        me_data = substrate.proj_me.data.copy()
        ii_data = substrate.proj_ii.data.copy()

        stdp_pre = getattr(substrate.proj_ee, "_pre_trace", np.array([]))
        stdp_post = getattr(substrate.proj_ee, "_post_trace", np.array([]))

        history_len = min(len(substrate.spike_history), 2000)
        if history_len > 0:
            spike_history = np.array(substrate.spike_history[-history_len:], dtype=np.int8)
        else:
            spike_history = np.zeros((0, substrate.n_cortical), dtype=np.int8)

        metadata = np.array(
            [
                substrate.n_cortical,
                substrate.n_inhibitory,
                substrate.n_memory,
                substrate.seed,
                substrate._total_steps,
                int(time.time()),
            ],
            dtype=np.int64,
        )

        np.savez_compressed(
            path,
            cortical_v=cortical_v,
            inhibitory_v=inhibitory_v,
            memory_v=memory_v,
            ee_indptr=ee_indptr,
            ee_indices=ee_indices,
            ee_data=ee_data,
            ei_data=ei_data,
            ie_data=ie_data,
            em_data=em_data,
            me_data=me_data,
            ii_data=ii_data,
            stdp_pre=stdp_pre,
            stdp_post=stdp_post,
            spike_history=spike_history,
            metadata=metadata,
        )

    @staticmethod
    def load(path: str) -> Any:
        """Restore substrate from checkpoint."""
        from .substrate import IdentitySubstrate

        data = np.load(path, allow_pickle=False)
        meta = data["metadata"]
        n_cortical, n_inhibitory, n_memory, seed = (
            int(meta[0]),
            int(meta[1]),
            int(meta[2]),
            int(meta[3]),
        )
        total_steps = int(meta[4])

        substrate = IdentitySubstrate(n_cortical, n_inhibitory, n_memory, seed)

        _restore_voltages(substrate.cortical, data["cortical_v"])
        _restore_voltages(substrate.inhibitory, data["inhibitory_v"])
        _restore_voltages(substrate.memory, data["memory_v"])

        substrate.proj_ee.indptr[:] = data["ee_indptr"]
        substrate.proj_ee.indices[:] = data["ee_indices"]
        substrate.proj_ee.data[:] = data["ee_data"]
        substrate.proj_ei.data[:] = data["ei_data"]
        substrate.proj_ie.data[:] = data["ie_data"]
        substrate.proj_em.data[:] = data["em_data"]
        substrate.proj_me.data[:] = data["me_data"]
        substrate.proj_ii.data[:] = data["ii_data"]

        stdp_pre = data["stdp_pre"]
        stdp_post = data["stdp_post"]
        if stdp_pre.size > 0 and hasattr(substrate.proj_ee, "_pre_trace"):
            substrate.proj_ee._pre_trace[:] = stdp_pre
        if stdp_post.size > 0 and hasattr(substrate.proj_ee, "_post_trace"):
            substrate.proj_ee._post_trace[:] = stdp_post

        spike_history = data["spike_history"]
        substrate._spike_history = [spike_history[i] for i in range(spike_history.shape[0])]
        substrate._total_steps = total_steps

        return substrate

    @staticmethod
    def merge(paths: list[str]) -> Any:
        """Merge multiple checkpoints by averaging weights and concatenating history."""
        if not paths:
            raise ValueError("No checkpoint paths provided")
        if len(paths) == 1:
            return Checkpoint.load(paths[0])

        base = Checkpoint.load(paths[0])
        ee_data_sum = base.proj_ee.data.copy()
        ei_data_sum = base.proj_ei.data.copy()
        ie_data_sum = base.proj_ie.data.copy()
        em_data_sum = base.proj_em.data.copy()
        me_data_sum = base.proj_me.data.copy()
        ii_data_sum = base.proj_ii.data.copy()
        all_history = list(base.spike_history)

        for p in paths[1:]:
            other = Checkpoint.load(p)
            ee_data_sum += other.proj_ee.data
            ei_data_sum += other.proj_ei.data
            ie_data_sum += other.proj_ie.data
            em_data_sum += other.proj_em.data
            me_data_sum += other.proj_me.data
            ii_data_sum += other.proj_ii.data
            all_history.extend(other.spike_history)

        n = len(paths)
        base.proj_ee.data[:] = ee_data_sum / n
        base.proj_ei.data[:] = ei_data_sum / n
        base.proj_ie.data[:] = ie_data_sum / n
        base.proj_em.data[:] = em_data_sum / n
        base.proj_me.data[:] = me_data_sum / n
        base.proj_ii.data[:] = ii_data_sum / n

        max_history = 2000
        base._spike_history = all_history[-max_history:]
        base._total_steps = sum(int(np.load(p, allow_pickle=False)["metadata"][4]) for p in paths)

        return base


def _restore_voltages(population: Any, voltages: np.ndarray[Any, Any]) -> None:
    """Write voltage array back into individual neuron objects."""
    for i, neuron in enumerate(population.neurons):
        if hasattr(neuron, "v"):
            neuron.v = float(voltages[i])
        elif hasattr(neuron, "x"):
            neuron.x = float(voltages[i])
    population._sync_voltages()
