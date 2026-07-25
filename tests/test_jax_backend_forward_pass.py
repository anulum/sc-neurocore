# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — JAX multilayer forward-pass contracts

"""Verify multilayer output shapes and forward-pass validation."""

import numpy as np
import pytest

from tests.jax_backend_support import jax_forward_pass, to_host


def test_jax_forward_pass_returns_layer_spikes_and_final_voltage() -> None:
    x = np.array([[0.5, 0.25], [0.0, 1.0]], dtype=np.float64)
    weights = [
        np.array([[0.6, 0.1], [0.2, 0.4]], dtype=np.float64),
        np.array([[0.5, 0.3]], dtype=np.float64),
    ]
    all_spikes, final_v = jax_forward_pass(weights, x, n_steps=3)
    assert len(all_spikes) == 2
    assert to_host(all_spikes[0]).shape == (3, 2, 2)
    assert to_host(all_spikes[1]).shape == (3, 2, 1)
    assert to_host(final_v).shape == (2, 1)


@pytest.mark.parametrize(
    ("weights", "x", "kwargs", "match"),
    [
        ([], np.ones((1, 2), dtype=np.float64), {}, "weights"),
        (
            [np.ones((1, 2), dtype=np.float64)],
            np.ones((1, 2), dtype=np.float64),
            {"n_steps": 0},
            "n_steps",
        ),
        ([np.ones((1, 2), dtype=np.float64)], np.ones(2, dtype=np.float64), {}, "2-D"),
        (
            [np.ones((1, 3), dtype=np.float64)],
            np.ones((1, 2), dtype=np.float64),
            {},
            "input dimension",
        ),
        (
            [np.ones((1, 2), dtype=np.float64)],
            np.array([[np.nan, 0.0]], dtype=np.float64),
            {},
            "finite",
        ),
        (
            [np.ones((1, 2), dtype=np.float64)],
            np.ones((1, 2), dtype=np.float64),
            {"alpha": 0.0},
            "alpha",
        ),
    ],
)
def test_jax_forward_pass_rejects_invalid_contracts(weights, x, kwargs, match) -> None:
    params = {
        "n_steps": 2,
        "v_rest": 0.0,
        "v_reset": 0.0,
        "v_threshold": 1.0,
        "alpha": 0.9,
    }
    params.update(kwargs)
    with pytest.raises(ValueError, match=match):
        jax_forward_pass(weights, x, **params)
