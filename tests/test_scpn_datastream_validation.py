# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN datastream validation contracts

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from sc_neurocore.scpn import SCPNDatastream, generate_scpn_datastream, validate_scpn_datastream


def test_validation_rejects_non_binary_spikes() -> None:
    stream = generate_scpn_datastream(n_steps=4, seed=1)
    bad = SCPNDatastream(
        dt_s=stream.dt_s,
        seed=stream.seed,
        probabilities=stream.probabilities,
        spike_train=stream.spike_train.astype(np.uint8).copy(),
        omega_rad_s=stream.omega_rad_s,
        knm=stream.knm,
    )
    bad.spike_train[0, 0] = 2

    with pytest.raises(ValueError, match="binary"):
        validate_scpn_datastream(bad)


def test_validation_rejects_shape_and_bound_violations() -> None:
    stream = generate_scpn_datastream(n_steps=4, seed=3)

    probabilities_out_of_bounds = stream.probabilities.copy()
    probabilities_out_of_bounds[0, 0] = 1.1
    asymmetric_knm = stream.knm.copy()
    asymmetric_knm[0, 1] += 0.5
    nonzero_diagonal_knm = stream.knm.copy()
    nonzero_diagonal_knm[0, 0] = 0.5

    cases: list[tuple[str, Callable[[], SCPNDatastream]]] = [
        (
            "matching shapes",
            lambda: SCPNDatastream(
                dt_s=stream.dt_s,
                seed=stream.seed,
                probabilities=stream.probabilities[:-1],
                spike_train=stream.spike_train,
                omega_rad_s=stream.omega_rad_s,
                knm=stream.knm,
            ),
        ),
        (
            "2-D",
            lambda: SCPNDatastream(
                dt_s=stream.dt_s,
                seed=stream.seed,
                probabilities=stream.probabilities.reshape(-1),
                spike_train=stream.spike_train.reshape(-1),
                omega_rad_s=stream.omega_rad_s,
                knm=stream.knm,
            ),
        ),
        (
            "layer columns",
            lambda: SCPNDatastream(
                dt_s=stream.dt_s,
                seed=stream.seed,
                probabilities=stream.probabilities[:, :-1],
                spike_train=stream.spike_train[:, :-1],
                omega_rad_s=stream.omega_rad_s,
                knm=stream.knm,
            ),
        ),
        (
            "omega_rad_s",
            lambda: SCPNDatastream(
                dt_s=stream.dt_s,
                seed=stream.seed,
                probabilities=stream.probabilities,
                spike_train=stream.spike_train,
                omega_rad_s=stream.omega_rad_s[:-1],
                knm=stream.knm,
            ),
        ),
        (
            "knm must have shape",
            lambda: SCPNDatastream(
                dt_s=stream.dt_s,
                seed=stream.seed,
                probabilities=stream.probabilities,
                spike_train=stream.spike_train,
                omega_rad_s=stream.omega_rad_s,
                knm=stream.knm[:-1, :],
            ),
        ),
        (
            "probabilities must be in",
            lambda: SCPNDatastream(
                dt_s=stream.dt_s,
                seed=stream.seed,
                probabilities=probabilities_out_of_bounds,
                spike_train=stream.spike_train,
                omega_rad_s=stream.omega_rad_s,
                knm=stream.knm,
            ),
        ),
        (
            "knm must be symmetric",
            lambda: SCPNDatastream(
                dt_s=stream.dt_s,
                seed=stream.seed,
                probabilities=stream.probabilities,
                spike_train=stream.spike_train,
                omega_rad_s=stream.omega_rad_s,
                knm=asymmetric_knm,
            ),
        ),
        (
            "knm diagonal",
            lambda: SCPNDatastream(
                dt_s=stream.dt_s,
                seed=stream.seed,
                probabilities=stream.probabilities,
                spike_train=stream.spike_train,
                omega_rad_s=stream.omega_rad_s,
                knm=nonzero_diagonal_knm,
            ),
        ),
    ]

    for match, make_invalid_stream in cases:
        with pytest.raises(ValueError, match=match):
            validate_scpn_datastream(make_invalid_stream())


def test_validation_rejects_non_positive_dt() -> None:
    stream = generate_scpn_datastream(n_steps=3, seed=6)
    bad = SCPNDatastream(
        dt_s=0.0,
        seed=stream.seed,
        probabilities=stream.probabilities,
        spike_train=stream.spike_train,
        omega_rad_s=stream.omega_rad_s,
        knm=stream.knm,
    )

    with pytest.raises(ValueError, match="dt_s"):
        validate_scpn_datastream(bad)


def test_validation_rejects_non_finite_arrays() -> None:
    stream = generate_scpn_datastream(n_steps=4, seed=5)

    probs = stream.probabilities.copy()
    probs[0, 0] = np.inf
    with pytest.raises(ValueError, match="probabilities must be finite"):
        validate_scpn_datastream(
            SCPNDatastream(
                dt_s=stream.dt_s,
                seed=stream.seed,
                probabilities=probs,
                spike_train=stream.spike_train,
                omega_rad_s=stream.omega_rad_s,
                knm=stream.knm,
            )
        )

    omega = stream.omega_rad_s.copy()
    omega[0] = np.inf
    with pytest.raises(ValueError, match="omega_rad_s must be finite"):
        validate_scpn_datastream(
            SCPNDatastream(
                dt_s=stream.dt_s,
                seed=stream.seed,
                probabilities=stream.probabilities,
                spike_train=stream.spike_train,
                omega_rad_s=omega,
                knm=stream.knm,
            )
        )

    knm = stream.knm.copy()
    knm[0, 1] = np.inf
    knm[1, 0] = np.inf
    with pytest.raises(ValueError, match="knm must be finite"):
        validate_scpn_datastream(
            SCPNDatastream(
                dt_s=stream.dt_s,
                seed=stream.seed,
                probabilities=stream.probabilities,
                spike_train=stream.spike_train,
                omega_rad_s=stream.omega_rad_s,
                knm=knm,
            )
        )
