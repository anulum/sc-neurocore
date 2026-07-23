# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio advanced characterize

"""Focused suite: TestCharacterize from former test_studio_advanced.py."""

from __future__ import annotations

from tests.studio_advanced_support import *  # noqa: F403

class TestCharacterize:
    def test_characterize_lif(self):
        from sc_neurocore.studio.models import simulate_model

        def sim_fn(**kw):
            cur = kw.pop("current", 500)
            kw.pop("params", None)
            kw.pop("init", None)
            kw.pop("dt", None)
            kw.pop("duration", None)
            kw.pop("protocol", None)
            return simulate_model("COBALIFNeuron", duration=50, current=cur)

        base = {"params": {}, "dt": 0.1, "duration": 50, "current": 500, "protocol": "constant"}
        r = characterize_model(sim_fn, base)
        assert "pattern" in r
        assert "fi_curve" in r
        assert len(r["fi_curve"]["currents"]) == 20
        assert "top_sensitivities" in r
        assert "state_ranges" in r

    def test_characterize_endpoint(self, client):
        r = client.post(
            "/api/characterize",
            json={
                "name": "HodgkinHuxleyNeuron",
                "current": 10,
                "duration": 50,
                "dt": 0.01,
            },
        )
        assert r.status_code == 200
        d = r.json()
        assert d["pattern"]["pattern"] in (
            "tonic",
            "bursting",
            "adapting",
            "irregular",
            "chaotic",
            "silent",
            "single_spike",
        )

