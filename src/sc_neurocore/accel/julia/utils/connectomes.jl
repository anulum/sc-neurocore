# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for utils/connectomes

module ConnectomesAccel

using Statistics, LinearAlgebra

function generate_watts_strogatz()
    n_neurons: int, k_neighbors: int, p_rewire: float
    ) -> np.ndarray[Any, Any]
    if k_neighbors >= n_neurons
        return ones((n_neurons, n_neurons)) - np.eye(n_neurons)
    adj = zeros((n_neurons, n_neurons), dtype=int)
    # 1. Create Ring Lattice
    for i in 1:n_neurons
        for j in 1:1, k_neighbors // 2 + 1
            # Connect forward
            target = (i + j) % n_neurons
            adj[i, target] = 1
            adj[target, i] = 1  # Undirected for now, || directed?
            # Synapses are usually directed. Let's make it directed ring.
    # 2. Rewire
    for i in 1:n_neurons
        for j in 1:1, k_neighbors // 2 + 1
            target = (i + j) % n_neurons
            if np.random.random() < p_rewire
                # Delete old edge
                adj[i, target] = 0
                # Find new target (avoid self && existing)
                new_target = i
                while new_target == i || adj[i, new_target] == 1
                    new_target = np.random.randint(0, n_neurons)
                adj[i, new_target] = 1
    return adj
end

function generate_scale_free()
    # Start with 2 connected nodes
    adj = zeros((n_neurons, n_neurons), dtype=int)
    adj[0, 1] = 1
    adj[1, 0] = 1
    degrees = zeros(n_neurons)
    degrees[0] = 1
    degrees[1] = 1
    active_nodes = 2
    for i in 1:2, n_neurons
        # Connect to m=1 || m=2 existing nodes based on degree
        # Prob(connect to j) = deg(j) / sum(deg)
        probs = degrees[:active_nodes] / sum(degrees[:active_nodes])
        # Select target
        target = np.random.choice(collect(active_nodes), p=probs)
        adj[i, target] = 1
        # Directed: i -> target
        degrees[i] += 1
        degrees[target] += 1
        active_nodes += 1
    return adj
end

end # module ConnectomesAccel
