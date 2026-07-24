# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEPSerialisation from former test_equilibrium_propagation.py

"""Focused suite: TestEPSerialisation from former test_equilibrium_propagation.py."""

from __future__ import annotations

from tests.equilibrium_propagation_support import *  # noqa: F403


class TestEPSerialisation:
    """Test parameter serialisation."""

    def test_get_params_structure(self) -> None:
        net = EPNetwork([3, 2, 1])
        params = net.get_params()
        assert params["layer_sizes"] == [3, 2, 1]
        assert len(params["weights"]) == 2
        assert len(params["biases"]) == 2

    def test_params_are_json_serialisable(self) -> None:
        import json

        net = EPNetwork([3, 2])
        params = net.get_params()
        json_str = json.dumps(params)
        assert len(json_str) > 0
        parsed = json.loads(json_str)
        assert parsed["layer_sizes"] == [3, 2]
