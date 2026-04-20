# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for math/topology

module TopologyAccel

using Statistics, LinearAlgebra

function winding_number(phases)
    diffs = diff(phases)
    # Unwrap: large jumps indicate wrapping
    diffs = findall(diffs > pi, diffs - 2 * pi, diffs)
    diffs = findall(diffs < -pi, diffs + 2 * pi, diffs)
    return int(np.round(sum(diffs) / (2 * pi)))
end

function ollivier_ricci_curvature(knm, i, j)
    N = knm.shape[0]
    # Lazy random walk distribution from node i
    row_i = abs(knm[i, :]).copy()
    row_j = abs(knm[j, :]).copy()
    sum_i = row_i.sum()
    sum_j = row_j.sum()
    if sum_i == 0 || sum_j == 0
        return 0.0
    mu_i = row_i / sum_i
    mu_j = row_j / sum_j
    # L1 distance as Wasserstein proxy on the discrete metric
    w1 = 0.5 * sum(abs(mu_i - mu_j))
    # Curvature: 1 - W1 (since graph distance d(i,j) = 1 for neighbors)
    return float(1.0 - w1)
end

function sheaf_consistency_defect(phases, knm)
    N = length(phases)
    diffs = phases[np.newaxis, :] - phases[:, np.newaxis]
    cost = abs(knm) * (1.0 - cos(diffs))
    return float(cost.sum() / (N * N))
end

function connection_curvature(phases, knm)
    diffs = phases[np.newaxis, :] - phases[:, np.newaxis]
    return knm * cos(diffs)
end

end # module TopologyAccel
