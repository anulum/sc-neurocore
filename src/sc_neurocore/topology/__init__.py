# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Network topology analysis

"""Graph metrics for SNN connectivity: small-world, modularity, centrality.

Post-hoc analysis of an existing connectivity matrix.

NOT to be confused with :mod:`sc_neurocore.network.topology`, which is
a different module that **generates** connectivity (Erdős–Rényi,
Watts–Strogatz, Barabási–Albert, ring, grid, all-to-all). The two
share the word "topology" but have disjoint roles:

- ``sc_neurocore.topology`` (this module) — measure existing graph
- ``sc_neurocore.network.topology`` — produce graph from parameters

Use this module for:
- Computing clustering coefficient, mean path length, small-world σ
- Newman modularity with optional caller-supplied partition
- Degree statistics + hub identification + assortativity

Use the other module for:
- Building an adjacency matrix from a topology family + parameters
- Feeding the result into :class:`sc_neurocore.network.Projection`

See :doc:`docs/api/graph_topology` for full reference.
"""

from .analyzer import TopologyAnalyzer, TopologyReport

__all__ = ["TopologyAnalyzer", "TopologyReport"]
