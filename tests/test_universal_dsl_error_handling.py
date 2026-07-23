# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestErrorHandling from former test_universal_dsl.py

"""Focused suite: TestErrorHandling from former test_universal_dsl.py."""

from __future__ import annotations

from tests.universal_dsl_support import *  # noqa: F403

class TestErrorHandling:
    """Test error handling for invalid schemas."""

    def test_empty_dynamics_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one ODE"):
            UniversalNeuron.from_dict(
                {
                    "metadata": {"schema_version": 1, "name": "Empty"},
                    "state": {"v": 0.0},
                    "dynamics": {},
                }
            )

    def test_unsupported_version_in_file(self) -> None:
        """Version gate fires when loading a schema with unsupported version."""
        import tempfile

        bad_schema = {
            "metadata": {"schema_version": 999, "name": "Future"},
            "state": {"v": 0.0},
            "dynamics": {"v": "I"},
        }
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(bad_schema, f)
            path = f.name
        with pytest.raises(ValueError, match="Schema version.*not supported"):
            load_schema(path)
        Path(path).unlink()

    def test_from_dict_works(self) -> None:
        neuron = UniversalNeuron.from_dict(
            {
                "metadata": {"schema_version": 1, "name": "TestModel"},
                "state": {"v": 0.0},
                "parameters": {},
                "dynamics": {"v": "I"},
                "integration": {"dt": 0.1, "method": "euler"},
            }
        )
        neuron.step(I=1.0)
        assert neuron.state["v"] != 0.0
