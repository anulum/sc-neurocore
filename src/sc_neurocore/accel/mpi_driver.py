# SPDX-License-Identifier: AGPL-3.0-or-later
from typing import Any
import numpy as np
import warnings

try:
    from mpi4py import MPI  # pragma: no cover  # type: ignore

    HAS_MPI = True  # pragma: no cover
except ImportError:
    HAS_MPI = False
    warnings.warn("mpi4py not found. Distributed computing disabled. Install 'mpi4py'.")


class MPIDriver:
    """
    Distributed SC-NeuroCore Driver using MPI.
    Handles partitioning and synchronization of bitstreams across cluster nodes.
    """

    def __init__(self) -> None:
        if HAS_MPI:  # pragma: no cover
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        else:
            self.comm = None
            self.rank = 0
            self.size = 1

    def scatter_workload(self, global_inputs: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """
        Distributes a large input array across nodes.
        Splits along axis 0 (Batch or Neurons).
        """
        if not HAS_MPI or self.size == 1:
            return global_inputs

        # MPI multi-node path  # pragma: no cover
        total_len = len(global_inputs)  # pragma: no cover
        chunk_size = total_len // self.size  # pragma: no cover
        local_input = np.zeros(chunk_size, dtype=global_inputs.dtype)  # pragma: no cover
        self.comm.Scatter(global_inputs, local_input, root=0)  # pragma: no cover
        return local_input  # pragma: no cover

    def gather_results(self, local_results: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """
        Collects results from all nodes to Root.
        """
        if not HAS_MPI or self.size == 1:
            return local_results

        # MPI multi-node path  # pragma: no cover
        total_len = len(local_results) * self.size  # pragma: no cover
        global_results = None  # pragma: no cover
        if self.rank == 0:  # pragma: no cover
            global_results = np.zeros(total_len, dtype=local_results.dtype)  # pragma: no cover
        self.comm.Gather(local_results, global_results, root=0)  # pragma: no cover
        if global_results is None:
            return np.zeros(0)
        return global_results

    def barrier(self) -> None:
        """Synchronize all nodes."""
        if HAS_MPI:  # pragma: no cover
            self.comm.Barrier()
