# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCreateSessionConfig from former test_ccw_bridge.py

"""Focused suite: TestCreateSessionConfig from former test_ccw_bridge.py."""

from __future__ import annotations

from tests.ccw_bridge_support import *  # noqa: F403


class TestCreateSessionConfig:
    def test_default_meditation_session(self):
        bridge = create_bridge()
        cfg = bridge.create_session_config()
        assert cfg["session"]["mode"] == "meditation"
        assert cfg["session"]["duration_minutes"] == 20
        # MODE_FREQUENCIES[MEDITATION] = (4.0, 7.83)
        assert cfg["audio"]["base_frequency"] == pytest.approx(4.0)
        assert cfg["audio"]["harmonic_frequency"] == pytest.approx(7.83)
        assert cfg["scpn_integration"]["enabled"] is True

    def test_explicit_mode_and_duration(self):
        bridge = create_bridge()
        cfg = bridge.create_session_config(mode=CCWMode.COSMIC, duration_minutes=45)
        assert cfg["session"]["mode"] == "cosmic"
        assert cfg["session"]["duration_minutes"] == 45
        # MODE_FREQUENCIES[COSMIC] = (136.1, 272.2) (OM)
        assert cfg["audio"]["base_frequency"] == pytest.approx(136.1)
        assert cfg["audio"]["harmonic_frequency"] == pytest.approx(272.2)
        assert cfg["visual"]["color_scheme"] == "cosmic"
