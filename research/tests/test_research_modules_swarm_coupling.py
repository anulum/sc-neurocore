# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSwarmCoupling from former test_research_modules.py

"""Focused suite: TestSwarmCoupling from former test_research_modules.py."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Ensure same-dir support module is importable under pytest importlib mode.
sys.path.insert(0, str(_Path(__file__).resolve().parent))

from research_modules_support import *  # noqa: F403

class TestSwarmCoupling:
    def _make_agent(self, n_inputs, n_neurons, seed=42):
        return SCLearningLayer(
            n_inputs=n_inputs,
            n_neurons=n_neurons,
            w_min=0.0,
            w_max=1.0,
            length=64,
            base_seed=seed,
        )

    def test_synchronize_shifts_weights(self, capsys):
        a = self._make_agent(3, 2, seed=0)
        b = self._make_agent(3, 2, seed=99)
        wa_before = a.get_weights().copy()
        wb_before = b.get_weights().copy()

        sc = SwarmCoupling(coupling_strength=0.5)
        sc.synchronize(a, b)

        wa_after = a.get_weights()
        wb_after = b.get_weights()
        # Weights should have changed
        assert not np.array_equal(wa_before, wa_after)
        assert not np.array_equal(wb_before, wb_after)

    def test_size_mismatch_raises(self):
        a = self._make_agent(2, 3)
        b = self._make_agent(2, 4)
        sc = SwarmCoupling()
        with pytest.raises(ValueError):
            sc.synchronize(a, b)
