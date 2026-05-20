# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

from typing import Any

import pytest

import sc_neurocore
from sc_neurocore.exceptions import SCDependencyError
from sc_neurocore.license import (
    POLAR_LICENSE_VALIDATION_ENDPOINT,
    CommercialLicenseStatus,
    get_license_status,
    load_license_from_env,
    reset_license_status,
    set_license_key,
    validate_license_key,
)


def teardown_function() -> None:
    reset_license_status()


def test_default_license_status_keeps_agpl_users_unblocked() -> None:
    status = get_license_status()

    assert status.mode == "agpl"
    assert status.valid is True
    assert status.commercial_enabled is False
    assert status.priority_support is False
    assert status.suppress_agpl_notice is False


def test_valid_polar_key_enables_commercial_mode_without_storing_key() -> None:
    calls: list[tuple[str, dict[str, str], float]] = []

    def transport(endpoint: str, payload: dict[str, str], timeout: float) -> dict[str, Any]:
        calls.append((endpoint, payload, timeout))
        return {
            "valid": True,
            "status": "active",
            "license_key": {"id": "lk_live_123", "status": "active"},
            "benefits": {"priority_support": True},
        }

    status = set_license_key("scn_live_key", organization_id="org_123", transport=transport)

    assert calls == [
        (
            POLAR_LICENSE_VALIDATION_ENDPOINT,
            {"key": "scn_live_key", "organization_id": "org_123"},
            10.0,
        )
    ]
    assert status == CommercialLicenseStatus(
        mode="commercial",
        valid=True,
        status="active",
        license_id="lk_live_123",
        organization_id="org_123",
        commercial_enabled=True,
        priority_support=True,
        suppress_agpl_notice=True,
        message="active",
    )
    assert get_license_status().license_id == "lk_live_123"
    assert "scn_live_key" not in repr(status)


def test_invalid_or_expired_polar_key_does_not_enable_commercial_mode() -> None:
    def transport(_endpoint: str, _payload: dict[str, str], _timeout: float) -> dict[str, Any]:
        return {
            "valid": False,
            "status": "expired",
            "error": "license key expired",
            "license_key": {"id": "lk_old", "status": "expired"},
        }

    status = validate_license_key("scn_old_key", transport=transport)

    assert status.mode == "commercial"
    assert status.valid is False
    assert status.commercial_enabled is False
    assert status.priority_support is False
    assert status.suppress_agpl_notice is False
    assert status.message == "license key expired"


def test_env_license_validation_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SC_NEUROCORE_LICENSE_KEY", raising=False)

    assert load_license_from_env() is None
    assert get_license_status().mode == "agpl"

    monkeypatch.setenv("SC_NEUROCORE_LICENSE_KEY", "scn_env_key")
    monkeypatch.setenv("SC_NEUROCORE_POLAR_ORGANIZATION_ID", "org_env")

    def transport(_endpoint: str, payload: dict[str, str], _timeout: float) -> dict[str, Any]:
        assert payload == {"key": "scn_env_key", "organization_id": "org_env"}
        return {"valid": True, "status": "active", "license_key": {"id": "lk_env"}}

    status = load_license_from_env(transport=transport)

    assert status is not None
    assert status.commercial_enabled is True
    assert get_license_status().license_id == "lk_env"


def test_missing_http_dependency_is_reported_only_when_validation_is_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sc_neurocore.license as license_module

    def missing_httpx(
        _endpoint: str,
        _payload: dict[str, str],
        *,
        timeout: float,
    ) -> dict[str, Any]:
        del timeout
        raise SCDependencyError("Install sc-neurocore[license] to validate keys")

    monkeypatch.setattr(license_module, "_post_json_with_httpx", missing_httpx)

    with pytest.raises(SCDependencyError, match="\\[license\\]"):
        validate_license_key("scn_key")


def test_root_api_exports_license_helpers() -> None:
    assert sc_neurocore.set_license_key is set_license_key
    assert sc_neurocore.get_license_status is get_license_status
