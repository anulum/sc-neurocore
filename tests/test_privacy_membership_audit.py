# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMembershipAudit from former test_privacy.py

"""Focused suite: TestMembershipAudit from former test_privacy.py."""

from __future__ import annotations

from typing import Any
import numpy as np
from sc_neurocore.privacy import (
    MembershipAudit,
)


class TestMembershipAudit:
    """The membership-inference privacy audit."""

    def test_basic(self) -> None:
        """The audit reports an accuracy in the unit interval and a verdict."""

        def model(s: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            return s.sum(axis=0).astype(np.float64)

        members = [np.ones((10, 4), dtype=np.int8) for _ in range(5)]
        non_members = [np.zeros((10, 4), dtype=np.int8) for _ in range(5)]

        audit = MembershipAudit(run_fn=model)
        result = audit.audit(members, non_members)

        assert "accuracy" in result
        assert "vulnerable" in result
        assert 0 <= result["accuracy"] <= 1

    def test_indistinguishable(self) -> None:
        """A constant model gives an audit accuracy of one half."""

        def constant_model(s: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            return np.ones(4)

        members = [np.random.randint(0, 2, (10, 4), dtype=np.int8) for _ in range(5)]
        non_members = [np.random.randint(0, 2, (10, 4), dtype=np.int8) for _ in range(5)]

        audit = MembershipAudit(run_fn=constant_model)
        result = audit.audit(members, non_members)
        # Constant model → 50% accuracy (no leakage)
        assert result["accuracy"] == 0.5
