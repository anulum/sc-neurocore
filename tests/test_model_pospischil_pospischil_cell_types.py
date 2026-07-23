# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPospischilCellTypes from former test_model_pospischil.py

"""Focused suite: TestPospischilCellTypes from former test_model_pospischil.py."""

from __future__ import annotations

from tests.model_pospischil_support import *  # noqa: F403

class TestPospischilCellTypes:
    @pytest.mark.parametrize(
        "g_m,label",
        [
            (0.07, "RS"),
            (0.0, "FS"),
            (0.03, "IB"),
        ],
    )
    def test_cell_type_fires(self, g_m: float, label: str):
        """All cell types should fire at sufficient current."""
        n = PospischilNeuron(g_m=g_m)
        spikes = _run(n, current=10.0, steps=50000)
        assert len(spikes) >= 50, f"{label} (g_m={g_m}): only {len(spikes)} spikes"

    def test_fs_faster_than_rs(self):
        """FS (fast-spiking) has higher rate than RS at same current."""
        n_fs = PospischilNeuron(g_m=0.0)
        n_rs = PospischilNeuron(g_m=0.07)
        s_fs = len(_run(n_fs, current=10.0, steps=50000))
        s_rs = len(_run(n_rs, current=10.0, steps=50000))
        assert s_fs > s_rs
