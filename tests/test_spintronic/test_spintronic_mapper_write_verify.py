# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWriteVerify from former test_spintronic_mapper.py

"""Focused suite: TestWriteVerify from former test_spintronic_mapper.py."""

from __future__ import annotations

from spintronic_mapper_support import *  # noqa: F403

class TestWriteVerify:
    def test_success(self):
        dev = SpintronicDeviceConfig.from_tech(SpintronicTech.SOT_MRAM)
        cell = SpintronicCell(0, 0, dev)
        result = write_verify(cell, 200)
        assert result.success
        assert result.error <= 4

    def test_with_noise(self):
        dev = SpintronicDeviceConfig.from_tech(SpintronicTech.SOT_MRAM)
        cell = SpintronicCell(0, 0, dev)
        rng = np.random.default_rng(42)
        result = write_verify(cell, 200, rng=rng)
        assert result.attempts >= 1

    def test_exhausts_attempts_when_noise_never_settles(self):
        # A noise source that always overshoots the tolerance forces every
        # attempt to miss, so the loop reports failure after max_attempts.
        class _AlwaysFarNoise:
            def normal(self, _mean: float, _std: float) -> float:
                return 100.0

        dev = SpintronicDeviceConfig.from_tech(SpintronicTech.SOT_MRAM)
        cell = SpintronicCell(0, 0, dev)
        result = write_verify(cell, 200, max_attempts=3, rng=_AlwaysFarNoise())
        assert result.success is False
        assert result.attempts == 3
