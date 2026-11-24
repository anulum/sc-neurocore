# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCNParameters from former test_model_courage_nekorkin_map.py

"""Focused suite: TestCNParameters from former test_model_courage_nekorkin_map.py."""

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.neurons.models.courage_nekorkin_map import (
    CourageNekorkinMapNeuron,
)


class TestCNParameters:
    """Parameter behaviour of the CourageNekorkinMapNeuron."""

    @pytest.mark.parametrize("m1", [0.5, 0.65, 0.8])
    def test_m1_sweep(self, m1: float) -> None:
        """The trace stays finite across m1 values."""
        n = CourageNekorkinMapNeuron(m1=m1)
        trace, _ = n.simulate(5000, backend="python")
        assert np.all(np.isfinite(trace))

    @pytest.mark.parametrize("beta", [0.08, 0.085, 0.09])
    def test_beta_sweep(self, beta: float) -> None:
        """The trace stays finite across beta values."""
        n = CourageNekorkinMapNeuron(beta=beta)
        trace, _ = n.simulate(5000, backend="python")
        assert np.all(np.isfinite(trace))

    @pytest.mark.parametrize("eps", [0.01, 0.02, 0.04])
    def test_eps_sweep(self, eps: float) -> None:
        """The trace stays finite across epsilon values."""
        n = CourageNekorkinMapNeuron(eps=eps)
        trace, _ = n.simulate(5000, backend="python")
        assert np.all(np.isfinite(trace))
