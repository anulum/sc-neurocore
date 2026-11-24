# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSTGNernst from former test_model_marder_stg.py

"""Focused suite: TestSTGNernst from former test_model_marder_stg.py."""

from __future__ import annotations

import math
from sc_neurocore.neurons.models.marder_stg import (
    MarderSTGNeuron,
)


class TestSTGNernst:
    """The calcium Nernst potential of the MarderSTGNeuron."""

    def test_e_ca_positive_at_rest(self) -> None:
        """E Ca positive at rest."""
        assert MarderSTGNeuron()._nernst_e_ca(0.05) > 100.0

    def test_e_ca_decreases_with_calcium(self) -> None:
        """E Ca decreases with calcium."""
        n = MarderSTGNeuron()
        assert n._nernst_e_ca(50.0) < n._nernst_e_ca(0.05)

    def test_e_ca_handles_zero_calcium(self) -> None:
        """E Ca handles zero calcium."""
        assert math.isfinite(MarderSTGNeuron()._nernst_e_ca(0.0))
