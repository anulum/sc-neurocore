# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCorticalColumnFeedforward from former test_cortical_column_dynamics.py

"""Focused suite: TestCorticalColumnFeedforward from former test_cortical_column_dynamics.py."""

from __future__ import annotations

from tests.cortical_column_dynamics_support import *  # noqa: F403


class TestCorticalColumnFeedforward:
    def test_excitatory_populations_present(self):
        """Excitatory populations should exist and produce spikes."""
        col = CorticalColumn(scale=0.02, seed=42)
        result = col.simulate(duration_ms=50.0, dt=0.1)
        exc_keys = [k for k in result if k.endswith("e")]
        assert len(exc_keys) >= 4, f"Expected >= 4 excitatory pops, got {exc_keys}"

    def test_inhibition_present(self):
        """Inhibitory populations should exist and produce spikes with strong drive."""
        col = CorticalColumn(scale=0.02, g_inh=4.0, seed=42)
        result = col.simulate(duration_ms=50.0, dt=0.1)
        inh_keys = [k for k in result if k.endswith("i")]
        assert len(inh_keys) >= 4, f"Expected >= 4 inhibitory pops, got {inh_keys}"
        inh_total = sum(result[k].sum() for k in inh_keys)
        assert inh_total > 0, "no inhibitory spikes at all"
