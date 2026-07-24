# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestArtifactRejection from former test_acquisition.py

"""Focused suite: TestArtifactRejection from former test_acquisition.py."""

from __future__ import annotations

from tests.test_bioware.acquisition_support import *  # noqa: F403


class TestArtifactRejection:
    def test_blanking(self) -> None:
        data = np.ones((1000, 5))
        ar = ArtifactRejector(blanking_pre_ms=0.5, blanking_post_ms=2.0)
        blanked = ar.blank(data, stim_times_s=[0.025], sample_rate_hz=20000.0)
        # Centre at sample 500, pre=10 post=40 → blanked
        assert blanked[500, 0] == 0.0

    def test_no_stim_no_blanking(self) -> None:
        data = np.ones((100, 3))
        ar = ArtifactRejector()
        blanked = ar.blank(data, stim_times_s=[], sample_rate_hz=20000.0)
        np.testing.assert_array_equal(blanked, data)
