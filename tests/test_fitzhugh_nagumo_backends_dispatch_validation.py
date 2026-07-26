# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — FitzHugh-Nagumo dispatch and validation tests

"""Dispatch, input validation, and algorithm selection contracts."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.fitzhugh_nagumo import FitzHughNagumoNeuron
from tests.fitzhugh_nagumo_backends_support import _run


def test_auto_matches_python_bit_exact() -> None:
    ref, ref_spikes, _rv, _rw = _run("python")
    got, spikes, _vf, _wf = _run("auto")
    np.testing.assert_array_equal(got, ref)
    assert spikes == ref_spikes


def test_invalid_backend_raises() -> None:
    with pytest.raises(ValueError, match="backend must be"):
        FitzHughNagumoNeuron().simulate(10, 0.0, backend="cuda")


def test_negative_n_steps_raises() -> None:
    with pytest.raises(ValueError, match="n_steps must be non-negative"):
        FitzHughNagumoNeuron().simulate(-1, 0.0)


def test_non_finite_current_raises() -> None:
    with pytest.raises(ValueError, match="current must be finite"):
        FitzHughNagumoNeuron().simulate(10, np.inf)


def test_non_rk4_integrator_rejected() -> None:
    # simulate accelerates RK4 only; other integrators must raise, not silently
    # produce RK4 results.
    for integ in ("baseline_euler", "rosenbrock"):
        with pytest.raises(ValueError, match="RK4 integrator only"):
            FitzHughNagumoNeuron(integrator=integ).simulate(10, 0.5)
