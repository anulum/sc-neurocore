# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native public coefficient mutation admission

"""Exercise genuine native domain refusal and recovery with independently owned parameters."""

from pathlib import Path

import numpy as np
import pytest

from tests.test_conversion_native_replay import library as library
from sc_neurocore.conversion.if_native import load_native, replay_native
from sc_neurocore.conversion.if_parameters import IFParameters


def test_native_replay_refuses_mutated_coefficients_and_recovers(library: Path) -> None:
    """Map real kernel refusal without touching caller state, then replay repaired owned coefficients."""
    api = load_native(str(library))
    parameters = IFParameters(
        (np.ones((1, 1), dtype=np.float64),), (None,), (1.0,), 0.0, (0.0,), "spikes"
    )
    frames = np.ones((1, 1, 1), dtype=np.float64)
    state = [np.zeros((1, 1), dtype=np.float64)]
    parameters.weights[0].fill(np.nan)
    with pytest.raises(ValueError, match="native dense IF domain admission refused"):
        replay_native(api, parameters, frames, state, True, True, 128)
    assert np.isnan(parameters.weights[0]).all()
    assert frames.tolist() == [[[1.0]]] and state[0].tolist() == [[0.0]]
    parameters.weights[0].fill(1.0)
    result = replay_native(api, parameters, frames, state, True, True, 128)
    assert result.output.tolist() == [[1.0]]
    assert result.final_state[0].tolist() == [[0.0]]
    assert result.state_trace[0].tolist() == [[[0.0]]]
    assert result.spike_trace[0].tolist() == [[[1.0]]]
    assert state[0].tolist() == [[0.0]]
