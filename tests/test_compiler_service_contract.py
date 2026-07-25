# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Compiler-service manifest contracts

"""Service manifest construction tests for the compiler boundary."""

from __future__ import annotations

import pytest

from sc_neurocore.compiler_service import DigitalTwinSyncContract, build_compiler_service_contract

from .compiler_service_support import _target


def test_build_compiler_service_contract_is_deterministic() -> None:
    manifest = build_compiler_service_contract(
        targets=(_target("z-target"), _target("a-target")),
        sync=DigitalTwinSyncContract(session_id="session-1"),
    )

    assert list(manifest) == [
        "schema_version",
        "service_status",
        "supported_targets",
        "sync_contract",
        "update_policy",
    ]
    assert manifest["service_status"] == "contract_only"
    assert [target["name"] for target in manifest["supported_targets"]] == [
        "a-target",
        "z-target",
    ]
    assert manifest["sync_contract"]["session_id"] == "session-1"


def test_compiler_service_contract_rejects_duplicate_targets() -> None:
    with pytest.raises(ValueError, match="target names"):
        build_compiler_service_contract(
            targets=(_target("dup"), _target("dup")),
            sync=DigitalTwinSyncContract(session_id="session-1"),
        )


def test_build_compiler_service_contract_rejects_empty_targets() -> None:
    """A service boundary with no supported targets is a misconfiguration."""
    with pytest.raises(ValueError, match="targets must be non-empty"):
        build_compiler_service_contract(
            targets=(),
            sync=DigitalTwinSyncContract(session_id="session-1"),
        )
