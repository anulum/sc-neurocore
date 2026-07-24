# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWilsonCowanPipeline from former test_model_wilson_cowan.py

"""Focused suite: TestWilsonCowanPipeline from former test_model_wilson_cowan.py."""

from __future__ import annotations

from tests.model_wilson_cowan_support import *  # noqa: F403


class TestWilsonCowanPipeline:
    def test_population_creates(self):
        assert Population(WilsonCowanUnit, n=10, label="wc").n == 10

    def test_network_returns_float_not_spike(self):
        """WilsonCowanUnit.step() returns float (E rate), not int.

        Network.step_all expects int return for spike detection.
        The model runs in the network but spike counts will be wrong
        (every non-zero E registers as spike). Document this limitation.
        """
        n = WilsonCowanUnit()
        result = n.step(5.0)
        assert isinstance(result, float)
        # The model is a RATE model, not a spiking model

    def test_paired_declarative_schemas_match_runtime_defaults(self):
        schema_root = _REPOSITORY / "src/sc_neurocore/neurons/model_schemas"
        toml_payload = tomllib.loads((schema_root / "wilson_cowan.toml").read_text())
        json_payload = json.loads((schema_root / "wilson_cowan.json").read_text())
        assert toml_payload == json_payload
        unit = WilsonCowanUnit()
        assert toml_payload["state"] == {"e": unit.e, "i": unit.i}
        for name, value in toml_payload["parameters"].items():
            assert getattr(unit, name) == value
        assert toml_payload["integration"] == {"dt": unit.dt, "method": "rk4"}
