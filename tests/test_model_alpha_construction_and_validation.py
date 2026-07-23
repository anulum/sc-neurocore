# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestConstructionAndValidation from former test_model_alpha.py

"""Focused suite: TestConstructionAndValidation from former test_model_alpha.py."""

from __future__ import annotations

from tests.model_alpha_support import *  # noqa: F403

class TestConstructionAndValidation:
    """Construction normalises fields and rejects invalid configurations."""

    def test_catalogue_defaults(self) -> None:
        n = AlphaNeuron()
        assert (n.v, n.a_exc, n.i_exc, n.a_inh, n.i_inh) == (0.0, 0.0, 0.0, 0.0, 0.0)
        assert (n.v_rest, n.v_threshold) == (0.0, 1.0)
        assert (n.tau_v, n.tau_exc, n.tau_inh, n.dt) == (20.0, 5.0, 10.0, 1.0)

    def test_scalar_fields_are_normalised_to_float(self) -> None:
        n = AlphaNeuron(v=1, tau_v=15)
        assert isinstance(n.v, float) and isinstance(n.tau_v, float)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"v": float("nan")},
            {"a_exc": float("inf")},
            {"i_exc": float("nan")},
            {"a_inh": float("inf")},
            {"i_inh": float("nan")},
            {"v_rest": float("nan")},
            {"v_threshold": float("inf")},
            {"v_threshold": -1.0},
            {"tau_v": 0.0},
            {"tau_exc": -1.0},
            {"tau_inh": 0.0},
            {"dt": 0.0},
            {"dt": float("nan")},
        ],
    )
    def test_rejects_non_physical_configuration(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError):
            AlphaNeuron(**kwargs)

    @pytest.mark.parametrize("field", ["v", "a_exc", "tau_v"])
    def test_rejects_non_numeric_fields(self, field: str) -> None:
        with pytest.raises(ValueError, match="must be numeric"):
            AlphaNeuron(**cast(dict[str, float], {field: "fast"}))
