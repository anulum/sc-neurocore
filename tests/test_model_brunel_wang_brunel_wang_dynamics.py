# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBrunelWangDynamics from former test_model_brunel_wang.py

"""Focused suite: TestBrunelWangDynamics from former test_model_brunel_wang.py."""

from __future__ import annotations

from tests.model_brunel_wang_support import *  # noqa: F403


class TestBrunelWangDynamics:
    def test_subthreshold_silent(self):
        n = BrunelWangNeuron()
        assert len(_run(n, current=0.0, steps=10000)) == 0

    def test_suprathreshold_fires(self):
        n = BrunelWangNeuron()
        assert len(_run(n, current=1.0, steps=10000)) >= 100

    def test_isi_regularity(self):
        n = BrunelWangNeuron()
        spikes = _run(n, current=1.0, steps=10000)
        if len(spikes) >= 20:
            isis = np.diff(spikes[5:]).astype(float)
            cv = np.std(isis) / np.mean(isis)
            assert cv < 0.3

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = BrunelWangNeuron()
            trace = [(n.step(1.0), n.v) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]
