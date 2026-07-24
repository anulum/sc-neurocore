# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRiscVSovereign from former test_platforms.py

"""Focused suite: TestRiscVSovereign from former test_platforms.py."""

from __future__ import annotations

from tests.platforms_support import *  # noqa: F403


class TestRiscVSovereign:
    @pytest.mark.parametrize(
        "name",
        [
            "sifive_x280_ai",
            "esperanto_et_soc",
            "ventana_veyron_ai",
            "tenstorrent_ascalon",
            "andes_ax45mpv",
        ],
    )
    def test_risc_v(self, name):
        from sc_neurocore.compiler.platforms import get_profile

        assert get_profile(name).platform_class == "risc_v_sovereign"
