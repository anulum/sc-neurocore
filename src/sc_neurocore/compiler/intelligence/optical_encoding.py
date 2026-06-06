# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Photonic MZI weight encoding

"""Mach-Zehnder interferometer (MZI) phase-shift encoding for photonic arrays."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class MZIWeightEncoding:
    """Encoded weights for a Mach-Zehnder interferometer photonic array.

    Attributes
    ----------
    phases_theta : list[list[float]]
        Phase-shift θ values (radians).
    phases_phi : list[list[float]]
        Phase-shift φ values (radians).
    transmission : list[list[float]]
        Effective transmission coefficients.
    mesh_size : int
        Number of MZI columns.
    """

    phases_theta: list[list[float]]
    phases_phi: list[list[float]]
    transmission: list[list[float]]
    mesh_size: int


def encode_mzi_weights(
    weights: list[list[float | int]],
    *,
    mesh_type: str = "clements",
    loss_db_per_mzi: float = 0.1,
) -> MZIWeightEncoding:
    """Encode a weight matrix as MZI phase-shift parameters."""
    rows = len(weights)
    cols = len(weights[0]) if weights else 0
    mesh_size = max(rows, cols)

    flat = [abs(w) for row in weights for w in row]
    max_abs = max(flat) if flat else 1.0
    if max_abs == 0:
        max_abs = 1.0

    norm = [[w / max_abs for w in row] for row in weights]

    phases_theta = []
    phases_phi = []
    transmission = []
    loss_factor = 10.0 ** (-loss_db_per_mzi / 10.0)

    for row in norm:
        row_theta = []
        row_phi = []
        row_trans = []
        for w in row:
            clamped = max(-1.0, min(1.0, w))
            theta = 2.0 * math.asin(abs(clamped))
            phi = math.pi if clamped < 0 else 0.0
            trans = abs(clamped) * loss_factor
            row_theta.append(round(theta, 6))
            row_phi.append(round(phi, 6))
            row_trans.append(round(trans, 6))
        phases_theta.append(row_theta)
        phases_phi.append(row_phi)
        transmission.append(row_trans)

    return MZIWeightEncoding(
        phases_theta=phases_theta,
        phases_phi=phases_phi,
        transmission=transmission,
        mesh_size=mesh_size,
    )


def generate_mzi_config(
    encoding: MZIWeightEncoding,
    *,
    output_format: str = "json",
) -> str:
    """Generate a photonic chip configuration file from MZI weights."""
    if output_format == "json":
        import json

        return json.dumps(
            {
                "mesh_size": encoding.mesh_size,
                "phases_theta": encoding.phases_theta,
                "phases_phi": encoding.phases_phi,
                "transmission": encoding.transmission,
            },
            indent=2,
        )
    else:  # CSV
        lines = ["row,col,theta,phi,transmission"]
        for i, (t_row, p_row, tr_row) in enumerate(
            zip(encoding.phases_theta, encoding.phases_phi, encoding.transmission)
        ):
            for j, (t, p, tr) in enumerate(zip(t_row, p_row, tr_row)):
                lines.append(f"{i},{j},{t:.6f},{p:.6f},{tr:.6f}")
        return "\n".join(lines)
