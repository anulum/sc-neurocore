# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for swarm/fitness

module FitnessAccel

using Statistics, LinearAlgebra

function coverage_score()
    grid_n = 10
    w, h = area
    cols = clamp((positions[:, 0] / w * grid_n).astype(int), 0, grid_n - 1)
    rows = clamp((positions[:, 1] / h * grid_n).astype(int), 0, grid_n - 1)
    occupied = set(zip(rows.tolist(), cols.tolist()))
    return length(occupied) / (grid_n * grid_n)
end

function cohesion_score()
    if length(positions) < 2
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = sqrt((diff^2).sum(axis=-1))
    # Upper triangle only
    triu_idx = np.triu_indices(length(positions), k=1)
    mean_dist = dists[triu_idx].mean()
    x, y = positions[:, 0], positions[:, 1]
    bbox_diag = sqrt((x.max() - x.min()) ^ 2 + (y.max() - y.min()) ^ 2) + 1e-12
    ideal = bbox_diag * 0.25
    return float(exp(-(((mean_dist - ideal) / ideal) ^ 2)))
end

function alignment_score()
    if length(headings) == 0
        return 0.0
    cx = cos(headings).mean()
    cy = sin(headings).mean()
    return float(sqrt(cx^2 + cy^2))
end

function target_score()
    if length(targets) == 0
        return 0.0
    # (n_agents, n_targets)
    diff = positions[:, np.newaxis, :] - targets[np.newaxis, :, :]
    dists = sqrt((diff^2).sum(axis=-1))
    nearest = dists.min(axis=1)
    mean_nearest = nearest.mean()
    return float(1.0 / (1.0 + mean_nearest / 10.0))
end

function obstacle_penalty()
    if length(obstacles) == 0
        return 0.0
    centers = obstacles[:, :2]
    radii = obstacles[:, 2]
    # (n_agents, n_obstacles)
    diff = positions[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = sqrt((diff^2).sum(axis=-1))
    inside = (dists < radii[np.newaxis, :]).any(axis=1)
    return float(inside.mean())
end

function composite()
    positions = env.get_positions()
    headings = env.get_headings()
    area = (env.cfg.width, env.cfg.height)
    cov = SwarmFitness.coverage_score(positions, area)
    coh = SwarmFitness.cohesion_score(positions)
    aln = SwarmFitness.alignment_score(headings)
    tgt = SwarmFitness.target_score(positions, env.targets)
    obs = SwarmFitness.obstacle_penalty(positions, env.obstacles)
    return 0.30 * cov + 0.20 * coh + 0.10 * aln + 0.30 * tgt - 0.10 * obs
end

end # module FitnessAccel
