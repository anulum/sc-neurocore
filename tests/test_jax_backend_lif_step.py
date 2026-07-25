# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — JAX LIF step contracts

"""Verify JAX LIF state updates and scalar/array validation."""

import numpy as np
import pytest

from tests.jax_backend_support import jax_lif_step, to_host


def test_jax_lif_step_updates_voltage_and_spikes() -> None:
    v_next, spikes = jax_lif_step(
        np.array([0.0, 0.9], dtype=np.float64),
        np.array([0.4, 0.8], dtype=np.float64),
        v_rest=0.0,
        v_reset=-0.1,
        v_threshold=1.0,
        alpha=0.5,
        resistance=1.0,
        noise=np.array([0.0, 0.0], dtype=np.float64),
    )
    assert np.allclose(to_host(v_next), np.array([0.4, -0.1]))
    assert np.array_equal(to_host(spikes), np.array([0, 1], dtype=np.uint8))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"alpha": 0.0}, "alpha"),
        ({"resistance": np.inf}, "resistance"),
        ({"v_threshold": np.nan}, "v_threshold"),
    ],
)
def test_jax_lif_step_rejects_invalid_scalar_parameters(kwargs, match) -> None:
    params = {
        "v_rest": 0.0,
        "v_reset": 0.0,
        "v_threshold": 1.0,
        "alpha": 0.5,
        "resistance": 1.0,
        "noise": np.zeros(2, dtype=np.float64),
    }
    params.update(kwargs)
    with pytest.raises(ValueError, match=match):
        jax_lif_step(
            np.array([0.0, 0.1], dtype=np.float64),
            np.array([0.2, 0.3], dtype=np.float64),
            **params,
        )


@pytest.mark.parametrize(
    ("v", "current", "noise", "match"),
    [
        (
            np.array([0, 1], dtype=np.int64),
            np.array([0.2, 0.3], dtype=np.float64),
            np.zeros(2, dtype=np.float64),
            "floating-point",
        ),
        (
            np.array([0.0, 0.1], dtype=np.float64),
            np.array([[0.2, 0.3]], dtype=np.float64),
            np.zeros(2, dtype=np.float64),
            "shape",
        ),
        (
            np.array([0.0, np.nan], dtype=np.float64),
            np.array([0.2, 0.3], dtype=np.float64),
            np.zeros(2, dtype=np.float64),
            "finite",
        ),
    ],
)
def test_jax_lif_step_rejects_invalid_array_contracts(v, current, noise, match) -> None:
    with pytest.raises(ValueError, match=match):
        jax_lif_step(
            v,
            current,
            v_rest=0.0,
            v_reset=0.0,
            v_threshold=1.0,
            alpha=0.5,
            resistance=1.0,
            noise=noise,
        )
