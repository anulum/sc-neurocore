# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Distributed SC-NeuroCore Driver using MPI

"""MPI scatter/gather driver for distributed stochastic-computing workloads.

The driver keeps single-process execution deterministic when ``mpi4py`` is not
available, while exposing the same workload partitioning, result collection, and
barrier interface used by multi-node stochastic-computing deployments.
"""

import warnings
from typing import Any

import numpy as np

try:
    from mpi4py import MPI  # pragma: no cover  # type: ignore

    HAS_MPI = True  # pragma: no cover
except ImportError:
    MPI = None  # type: ignore[assignment]
    HAS_MPI = False
    warnings.warn("mpi4py not found. Distributed computing disabled. Install 'mpi4py'.")


class MPIDriver:
    """Distributed sc-neurocore driver built on MPI.

    Handles partitioning and synchronisation of bitstreams across cluster
    nodes.
    """

    def __init__(self) -> None:
        self.comm: Any | None
        if HAS_MPI and MPI is not None:  # pragma: no cover
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        else:
            self.comm = None
            self.rank = 0
            self.size = 1

    def scatter_workload(self, global_inputs: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Distribute a large input array across nodes along axis 0.

        Parameters
        ----------
        global_inputs : numpy.ndarray
            Full input array held on the root rank, split along axis 0
            (batch or neuron dimension).

        Returns
        -------
        numpy.ndarray
            This rank's contiguous chunk of the input array.
        """
        if not HAS_MPI or self.size == 1:
            return global_inputs
        comm = self.comm
        if comm is None:  # pragma: no cover
            return global_inputs

        # MPI multi-node path  # pragma: no cover
        total_len = len(global_inputs)  # pragma: no cover
        chunk_size = total_len // self.size  # pragma: no cover
        local_input = np.zeros(chunk_size, dtype=global_inputs.dtype)  # pragma: no cover
        comm.Scatter(global_inputs, local_input, root=0)  # pragma: no cover
        return local_input  # pragma: no cover

    def gather_results(self, local_results: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Collect per-node result arrays back to the root rank."""
        if not HAS_MPI or self.size == 1:
            return local_results
        comm = self.comm
        if comm is None:  # pragma: no cover
            return local_results

        # MPI multi-node path  # pragma: no cover
        total_len = len(local_results) * self.size  # pragma: no cover
        global_results = None  # pragma: no cover
        if self.rank == 0:  # pragma: no cover
            global_results = np.zeros(total_len, dtype=local_results.dtype)  # pragma: no cover
        comm.Gather(local_results, global_results, root=0)  # pragma: no cover
        if global_results is None:
            return np.zeros(0)
        return global_results

    def barrier(self) -> None:
        """Synchronize all nodes."""
        if HAS_MPI and self.comm is not None:  # pragma: no cover
            self.comm.Barrier()
