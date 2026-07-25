# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Compiler-service update policy contracts

"""Live-update classification tests for the compiler boundary."""

from __future__ import annotations

import pytest

from sc_neurocore.compiler_service import LiveUpdateKind, LiveUpdatePolicy, plan_live_update

from .compiler_service_support import _request


def test_live_update_policy_classifies_hot_swap() -> None:
    package = plan_live_update(_request("weights", "lfsr_seeds"))

    assert package.kind is LiveUpdateKind.HOT_SWAP
    assert package.requires_full_resynthesis is False
    assert "digital_twin_replay" in package.validation_gates


def test_live_update_policy_classifies_partial_reconfiguration() -> None:
    package = plan_live_update(_request("routing_table", "aer_route_overlay"))

    assert package.kind is LiveUpdateKind.PARTIAL_RECONFIGURATION
    assert package.requires_full_resynthesis is False


def test_live_update_policy_classifies_full_resynthesis() -> None:
    package = plan_live_update(_request("weights", "clock"))

    assert package.kind is LiveUpdateKind.FULL_RESYNTHESIS_REQUIRED
    assert package.requires_full_resynthesis is True


def test_live_update_policy_rejects_unknown_field() -> None:
    with pytest.raises(ValueError, match="unknown update fields"):
        LiveUpdatePolicy().classify(("unknown",))


def test_classify_rejects_empty_changed_fields() -> None:
    """Classifying no changed fields is a contract violation, raised before the
    known-field set is consulted."""
    with pytest.raises(ValueError, match="changed_fields must be non-empty"):
        LiveUpdatePolicy().classify(())
