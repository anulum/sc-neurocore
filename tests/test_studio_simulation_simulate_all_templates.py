# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio simulation simulate all templates

"""Focused suite: TestSimulateAllTemplates from former test_studio_simulation.py."""

from __future__ import annotations

from tests.studio_simulation_support import *  # noqa: F403

class TestSimulateAllTemplates:
    @pytest.mark.parametrize("name", list(TEMPLATES.keys()))
    def test_template_runs_without_error(self, name):
        t = TEMPLATES[name]
        result = simulate(
            equations=t["equations"],
            threshold=t["threshold"],
            reset=t["reset"],
            params=t["params"],
            init=t["init"],
            dt=t["dt"],
            duration=t["duration"],
            current=t["current"],
        )
        assert len(result["time"]) > 0
        assert all(isinstance(v, list) for v in result["states"].values())

