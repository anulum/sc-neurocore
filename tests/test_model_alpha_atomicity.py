# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAtomicity from former test_model_alpha.py

"""Focused suite: TestAtomicity from former test_model_alpha.py."""

from __future__ import annotations

from tests.model_alpha_support import *  # noqa: F403

class TestAtomicity:
    """Rejected steps leave every dynamic state unchanged."""

    @pytest.mark.parametrize(
        ("field", "message"),
        (("v", "state must be numeric"), ("tau_v", "parameters must be numeric")),
    )
    def test_rejects_non_numeric_runtime_fields(self, field: str, message: str) -> None:
        n = AlphaNeuron()
        setattr(n, field, "invalid")
        with pytest.raises(ValueError, match=message):
            n.step(0.0)

    def test_rejects_non_numeric_runtime_current(self) -> None:
        n = AlphaNeuron()
        before = (n.v, n.a_exc, n.i_exc, n.a_inh, n.i_inh)
        with pytest.raises(ValueError, match="current values must be numeric"):
            n.step(cast(float, "invalid"))
        assert (n.v, n.a_exc, n.i_exc, n.a_inh, n.i_inh) == before

    @pytest.mark.parametrize("current", [float("nan"), float("inf"), -float("inf")])
    def test_rejects_non_finite_current(self, current: float) -> None:
        n = AlphaNeuron()
        before = (n.v, n.a_exc, n.i_exc, n.a_inh, n.i_inh)
        with pytest.raises(ValueError, match="current"):
            n.step(current)
        assert (n.v, n.a_exc, n.i_exc, n.a_inh, n.i_inh) == before

    def test_rejects_non_finite_runtime_state_before_update(self) -> None:
        n = AlphaNeuron()
        n.v = float("nan")
        with pytest.raises(ValueError, match="state"):
            n.step(0.0)
        assert np.isnan(n.v)

    def test_rejects_invalid_runtime_configuration_before_mutation(self) -> None:
        n = AlphaNeuron()
        n.tau_exc = 0.0
        before = (n.v, n.a_exc, n.i_exc, n.a_inh, n.i_inh)
        with pytest.raises(ValueError, match="tau_exc"):
            n.step(1.0)
        assert (n.v, n.a_exc, n.i_exc, n.a_inh, n.i_inh) == before

    def test_rejects_non_finite_update_before_state_mutation(self) -> None:
        n = AlphaNeuron(v=-1.0e308)
        before = (n.v, n.a_exc, n.i_exc, n.a_inh, n.i_inh)
        with pytest.raises((FloatingPointError, ValueError), match="finite"):
            n.step(1.0e308)
        assert (n.v, n.a_exc, n.i_exc, n.a_inh, n.i_inh) == before
