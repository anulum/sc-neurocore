# SPDX-License-Identifier: AGPL-3.0-or-later
# Math

Mathematical foundations: category theory, topological observables,
and differential geometry on coupling graphs.

## Category Theory

::: sc_neurocore.math.category_theory

## Topological Observables

Geometric invariants computed on SCPN coupling graphs: winding number
from phase dynamics, Ollivier-Ricci curvature on edges, sheaf
consistency defect, and connection curvature from parallel transport.

`ollivier_ricci_curvature` evaluates the graph-metric definition on the
non-negative coupling support: lazy random-walk measures are transported by
Wasserstein-1 distance over shortest-hop graph distances, and invalid matrices
or node indices fail closed before transport.

::: sc_neurocore.math.topology.winding_number

::: sc_neurocore.math.topology.ollivier_ricci_curvature

::: sc_neurocore.math.topology.sheaf_consistency_defect

::: sc_neurocore.math.topology.connection_curvature
