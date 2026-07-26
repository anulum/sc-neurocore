# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Izhikevich-2007 dispatch and validation tests

"""Dispatch, input validation, and algorithm selection contracts."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.izhikevich2007 import Izhikevich2007Neuron
from tests.izhikevich2007_backends_support import _run


def test_auto_matches_python_bit_exact() -> None:
    ref, ref_spikes, _rv, _ru = _run("python")
    got, spikes, _vf, _uf = _run("auto")
    np.testing.assert_array_equal(got, ref)
    assert spikes == ref_spikes


def test_invalid_backend_raises() -> None:
    with pytest.raises(ValueError, match="backend must be"):
        Izhikevich2007Neuron().simulate(10, 0.0, backend="cuda")


def test_negative_n_steps_raises() -> None:
    with pytest.raises(ValueError, match="n_steps must be non-negative"):
        Izhikevich2007Neuron().simulate(-1, 0.0)


def test_non_finite_current_raises() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        Izhikevich2007Neuron().simulate(10, np.inf)


def test_non_rk4_integrator_rejected() -> None:
    with pytest.raises(ValueError, match="RK4 integrator only"):
        Izhikevich2007Neuron(integrator="euler").simulate(10, 300.0)
