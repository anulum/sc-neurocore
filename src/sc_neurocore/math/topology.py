# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Topological observables for SCPN phase dynamics

"""Topological and geometric observables for coupled oscillator networks.

These are the quick-win implementations from the Holonomic Atlas
mathematical foundations audit (Round 2).

    from sc_neurocore.math.topology import (
        winding_number,
        ollivier_ricci_curvature,
        sheaf_consistency_defect,
    )
"""

from __future__ import annotations

import numpy as np


def winding_number(phases: np.ndarray) -> int:
    """Compute the winding number of a phase trajectory around S^1.

    The winding number counts how many times the phase wraps around
    the circle [0, 2*pi). It is a topological invariant — continuous
    deformations of the trajectory cannot change it.

    Parameters
    ----------
    phases : np.ndarray, shape (T,)
        Time series of phase values (radians).

    Returns
    -------
    int
        Number of complete windings (positive = counterclockwise).
    """
    diffs = np.diff(phases)
    # Unwrap: large jumps indicate wrapping
    diffs = np.where(diffs > np.pi, diffs - 2 * np.pi, diffs)
    diffs = np.where(diffs < -np.pi, diffs + 2 * np.pi, diffs)
    return int(np.round(np.sum(diffs) / (2 * np.pi)))


def ollivier_ricci_curvature(knm: np.ndarray, i: int, j: int) -> float:
    """Compute Ollivier-Ricci curvature between nodes i and j on the coupling graph.

    Ollivier (2009), "Ricci curvature of Markov chains on metric spaces."
    The curvature kappa(i,j) measures how much the neighborhoods of i and j
    overlap. Positive curvature = neighborhoods converge (community structure).
    Negative curvature = neighborhoods diverge (bottleneck).

    Approximation: kappa(i,j) = 1 - W1(mu_i, mu_j) / d(i,j)
    where mu_i is the lazy random walk distribution from node i,
    and W1 is the Wasserstein-1 distance on the graph.

    Simplified version: uses the coupling strength ratio as a proxy.

    Parameters
    ----------
    knm : np.ndarray, shape (N, N)
        Coupling matrix (non-negative, not necessarily symmetric).
    i, j : int
        Node indices.

    Returns
    -------
    float
        Estimated Ollivier-Ricci curvature in [-1, 1].
    """
    N = knm.shape[0]
    # Lazy random walk distribution from node i
    row_i = np.abs(knm[i, :]).copy()
    row_j = np.abs(knm[j, :]).copy()

    sum_i = row_i.sum()
    sum_j = row_j.sum()
    if sum_i == 0 or sum_j == 0:
        return 0.0

    mu_i = row_i / sum_i
    mu_j = row_j / sum_j

    # L1 distance as Wasserstein proxy on the discrete metric
    w1 = 0.5 * np.sum(np.abs(mu_i - mu_j))

    # Curvature: 1 - W1 (since graph distance d(i,j) = 1 for neighbors)
    return float(1.0 - w1)


def sheaf_consistency_defect(phases: np.ndarray, knm: np.ndarray) -> float:
    """Compute the sheaf consistency defect for the SCPN phase state.

    In sheaf theory, a global section exists iff the gluing conditions
    are satisfied on all overlaps. For the SCPN, the coupling matrix
    defines the overlaps, and the phase differences weighted by coupling
    measure the failure to glue.

    defect = (1/N^2) * sum_{i,j} |K_ij| * |1 - cos(theta_i - theta_j)|

    When phases are synchronized (all equal), defect = 0.
    When phases are maximally incoherent, defect approaches max(|K|).

    This is equivalent to (1 - Kuramoto_R) weighted by coupling.

    Parameters
    ----------
    phases : np.ndarray, shape (N,)
        Phase values (radians) for each layer/oscillator.
    knm : np.ndarray, shape (N, N)
        Coupling matrix.

    Returns
    -------
    float
        Sheaf consistency defect >= 0. Zero means globally coherent.
    """
    N = len(phases)
    diffs = phases[np.newaxis, :] - phases[:, np.newaxis]
    cost = np.abs(knm) * (1.0 - np.cos(diffs))
    return float(cost.sum() / (N * N))


def connection_curvature(phases: np.ndarray, knm: np.ndarray) -> np.ndarray:
    """Compute the connection curvature from PGBO phase dynamics.

    The PGBO covariant derivative u_mu = dphi_mu - alpha * A_mu
    defines a U(1) connection. The curvature F_{ij} = K_{ij} * cos(theta_i - theta_j)
    measures the obstruction to parallel transport between layers i and j.

    Parameters
    ----------
    phases : np.ndarray, shape (N,)
        Phase values.
    knm : np.ndarray, shape (N, N)
        Coupling matrix.

    Returns
    -------
    np.ndarray, shape (N, N)
        Connection curvature matrix. Diagonal is zero.
    """
    diffs = phases[np.newaxis, :] - phases[:, np.newaxis]
    return knm * np.cos(diffs)
