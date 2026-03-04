from typing import Any, Optional
import numpy as np


class ConnectomeGenerator:
    """
    Generates biologically plausible connectivity matrices.
    """

    @staticmethod
    def generate_watts_strogatz(n_neurons: int, k_neighbors: int, p_rewire: float) -> np.ndarray[Any, Any]:
        """
        Watts-Strogatz Small-World Model.

        1. Start with a regular ring lattice (connect to k neighbors).
        2. Randomly rewire edges with probability p.

        Returns:
            Adjacency Matrix (Binary)
        """
        if k_neighbors >= n_neurons:
            return np.ones((n_neurons, n_neurons)) - np.eye(n_neurons)

        adj = np.zeros((n_neurons, n_neurons), dtype=int)

        # 1. Create Ring Lattice
        for i in range(n_neurons):
            for j in range(1, k_neighbors // 2 + 1):
                # Connect forward
                target = (i + j) % n_neurons
                adj[i, target] = 1
                adj[target, i] = 1  # Undirected for now, or directed?
                # Synapses are usually directed. Let's make it directed ring.

        # 2. Rewire
        for i in range(n_neurons):
            for j in range(1, k_neighbors // 2 + 1):
                target = (i + j) % n_neurons

                if np.random.random() < p_rewire:
                    # Delete old edge
                    adj[i, target] = 0

                    # Find new target (avoid self and existing)
                    new_target = i
                    while new_target == i or adj[i, new_target] == 1:
                        new_target = np.random.randint(0, n_neurons)

                    adj[i, new_target] = 1

        return adj

    @staticmethod
    def generate_scale_free(n_neurons: int) -> np.ndarray[Any, Any]:
        """
        Barabasi-Albert Scale-Free Model (Preferential Attachment).
        """
        # Start with 2 connected nodes
        adj = np.zeros((n_neurons, n_neurons), dtype=int)
        adj[0, 1] = 1
        adj[1, 0] = 1
        degrees = np.zeros(n_neurons)
        degrees[0] = 1
        degrees[1] = 1

        active_nodes = 2

        for i in range(2, n_neurons):
            # Connect to m=1 or m=2 existing nodes based on degree
            # Prob(connect to j) = deg(j) / sum(deg)

            probs = degrees[:active_nodes] / np.sum(degrees[:active_nodes])

            # Select target
            target = np.random.choice(np.arange(active_nodes), p=probs)

            adj[i, target] = 1
            # Directed: i -> target

            degrees[i] += 1
            degrees[target] += 1
            active_nodes += 1

        return adj
