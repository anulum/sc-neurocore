
import numpy as np
import warnings

try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False
    warnings.warn("mpi4py not found. Distributed computing disabled. Install 'mpi4py'.")

class MPIDriver:
    """
    Distributed SC-NeuroCore Driver using MPI.
    Handles partitioning and synchronization of bitstreams across cluster nodes.
    """
    
    def __init__(self):
        if HAS_MPI:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        else:
            self.comm = None
            self.rank = 0
            self.size = 1
            
    def scatter_workload(self, global_inputs: np.ndarray) -> np.ndarray:
        """
        Distributes a large input array across nodes.
        Splits along axis 0 (Batch or Neurons).
        """
        if not HAS_MPI or self.size == 1:
            return global_inputs
            
        # Determine split sizes
        total_len = len(global_inputs)
        chunk_size = total_len // self.size
        # Simple equal split (assumes divisibility for now)
        
        # Buffer for local chunk
        local_input = np.zeros(chunk_size, dtype=global_inputs.dtype)
        
        self.comm.Scatter(global_inputs, local_input, root=0)
        return local_input

    def gather_results(self, local_results: np.ndarray) -> np.ndarray:
        """
        Collects results from all nodes to Root.
        """
        if not HAS_MPI or self.size == 1:
            return local_results
            
        total_len = len(local_results) * self.size
        global_results = None
        if self.rank == 0:
            global_results = np.zeros(total_len, dtype=local_results.dtype)
            
        self.comm.Gather(local_results, global_results, root=0)
        return global_results

    def barrier(self):
        """Synchronize all nodes."""
        if HAS_MPI:
            self.comm.Barrier()
