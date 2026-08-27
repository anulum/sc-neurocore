# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — PING circuit engine-binding contracts

"""Installed-extension contracts for the PING circuit step kernel."""

from __future__ import annotations

import importlib
from typing import cast

import numpy as np
import numpy.typing as npt

from tests.engine_requirement import require_engine

require_engine()
import sc_neurocore_engine as engine

extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")

FloatArray = npt.NDArray[np.float64]
SpikeArray = npt.NDArray[np.uint8]
PopulationState = tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    SpikeArray,
]


def _population_state(size: int) -> PopulationState:
    return (
        np.full(size, -67.0, dtype=np.float64),
        np.zeros(size, dtype=np.float64),
        np.zeros(size, dtype=np.float64),
        np.zeros(size, dtype=np.float64),
        np.zeros(size, dtype=np.float64),
        np.zeros(size, dtype=np.float64),
        np.zeros(size, dtype=np.uint8),
    )


def _step(
    excitatory: PopulationState,
    inhibitory: PopulationState,
) -> tuple[int, int]:
    return cast(
        tuple[int, int],
        extension.py_ping_step(
            *excitatory,
            *inhibitory,
            -67.0,
            0.0,
            -80.0,
            0.05,
            1.0,
            -52.0,
            -67.0,
            2.0,
            5.0,
            5.0,
            0.0,
            0.0,
            0.1,
        ),
    )


def test_exported_name_signature_and_top_level_identity_are_stable() -> None:
    function = extension.py_ping_step

    assert function.__name__ == "py_ping_step"
    assert function.__module__ == "sc_neurocore_engine.sc_neurocore_engine"
    assert function.__text_signature__ == (
        "(v_e, g_ampa_e, g_gaba_e, refrac_e, i_drive_e, xi_e, spikes_e_out, "
        "v_i, g_ampa_i, g_gaba_i, refrac_i, i_drive_i, xi_i, spikes_i_out, "
        "e_l, e_ampa, e_gaba, g_l, c_m, v_threshold, v_reset, t_refrac, "
        "tau_ampa, tau_gaba, sigma_e, sigma_i, dt)"
    )
    assert engine.py_ping_step is function


def test_no_drive_rest_state_is_updated_in_place_without_spikes() -> None:
    excitatory = _population_state(2)
    inhibitory = _population_state(1)

    spike_counts = _step(excitatory, inhibitory)

    assert spike_counts == (0, 0)
    np.testing.assert_array_equal(excitatory[0], [-67.0, -67.0])
    np.testing.assert_array_equal(inhibitory[0], [-67.0])
    np.testing.assert_array_equal(excitatory[6], [0, 0])
    np.testing.assert_array_equal(inhibitory[6], [0])


def test_refractory_state_is_clamped_and_decremented_in_place() -> None:
    excitatory = _population_state(1)
    inhibitory = _population_state(1)
    excitatory[0][0] = -40.0
    excitatory[3][0] = 0.2

    spike_counts = _step(excitatory, inhibitory)

    assert spike_counts == (0, 0)
    np.testing.assert_array_equal(excitatory[0], [-67.0])
    np.testing.assert_allclose(excitatory[3], [0.1], rtol=0.0, atol=1e-15)
