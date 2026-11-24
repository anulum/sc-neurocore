# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSiegertIsolation from former test_model_siegert.py

"""Focused suite: TestSiegertIsolation from former test_model_siegert.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.neurons.models.siegert import (
    SiegertTransferFunction,
)


class TestSiegertIsolation:
    """The SiegertTransferFunction in isolation."""

    def test_defaults(self) -> None:
        """The defaults match the published model."""
        n = SiegertTransferFunction()
        assert n.tau_m == 20.0 and n.tau_rp == 2.0
        assert n.v_threshold == -50.0 and n.v_reset == -70.0

    def test_step_returns_float(self) -> None:
        """One step returns a floating-point rate."""
        n = SiegertTransferFunction()
        assert isinstance(n.step(20.0), (float, np.floating))

    def test_reset_noop(self) -> None:
        """A reset after a step is accepted as a no-op."""
        n = SiegertTransferFunction()
        n.step(20.0)
        n.reset()  # should not raise
