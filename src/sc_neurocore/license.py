#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Opt-in commercial licence validation for SC-NeuroCore."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from sc_neurocore.exceptions import SCDependencyError

POLAR_LICENSE_VALIDATION_ENDPOINT = (
    "https://api.polar.sh/v1/customer-portal/license-keys/validate"
)
LICENSE_KEY_ENV = "SC_NEUROCORE_LICENSE_KEY"
POLAR_ORGANIZATION_ENV = "SC_NEUROCORE_POLAR_ORGANIZATION_ID"

LicenseMode = Literal["agpl", "commercial"]
LicenseTransport = Callable[[str, dict[str, str], float], Mapping[str, Any]]


@dataclass(frozen=True)
class CommercialLicenseStatus:
    """Current AGPL/commercial licence state.

    The raw licence key is intentionally never stored in this object.
    """

    mode: LicenseMode
    valid: bool
    status: str
    license_id: str | None = None
    organization_id: str | None = None
    commercial_enabled: bool = False
    priority_support: bool = False
    suppress_agpl_notice: bool = False
    message: str = ""


_AGPL_STATUS = CommercialLicenseStatus(
    mode="agpl",
    valid=True,
    status="agpl",
    commercial_enabled=False,
    priority_support=False,
    suppress_agpl_notice=False,
    message="AGPL-3.0-or-later mode; no commercial licence key configured",
)
_CURRENT_STATUS = _AGPL_STATUS


def get_license_status() -> CommercialLicenseStatus:
    """Return the current local licence state without contacting a network."""

    return _CURRENT_STATUS


def reset_license_status() -> None:
    """Reset process-local licence state to the default AGPL mode."""

    global _CURRENT_STATUS
    _CURRENT_STATUS = _AGPL_STATUS


def set_license_key(
    key: str,
    *,
    organization_id: str | None = None,
    endpoint: str = POLAR_LICENSE_VALIDATION_ENDPOINT,
    timeout: float = 10.0,
    transport: LicenseTransport | None = None,
) -> CommercialLicenseStatus:
    """Validate and install an explicit commercial licence key for this process."""

    status = validate_license_key(
        key,
        organization_id=organization_id,
        endpoint=endpoint,
        timeout=timeout,
        transport=transport,
    )
    global _CURRENT_STATUS
    _CURRENT_STATUS = status
    return status


def load_license_from_env(
    *,
    endpoint: str = POLAR_LICENSE_VALIDATION_ENDPOINT,
    timeout: float = 10.0,
    transport: LicenseTransport | None = None,
) -> CommercialLicenseStatus | None:
    """Validate ``SC_NEUROCORE_LICENSE_KEY`` when it is explicitly configured."""

    key = os.environ.get(LICENSE_KEY_ENV)
    if not key:
        return None
    return set_license_key(
        key,
        organization_id=os.environ.get(POLAR_ORGANIZATION_ENV),
        endpoint=endpoint,
        timeout=timeout,
        transport=transport,
    )


def validate_license_key(
    key: str,
    *,
    organization_id: str | None = None,
    endpoint: str = POLAR_LICENSE_VALIDATION_ENDPOINT,
    timeout: float = 10.0,
    transport: LicenseTransport | None = None,
) -> CommercialLicenseStatus:
    """Validate a Polar customer-portal licence key.

    Validation is opt-in. AGPL users who never call this function, never call
    ``set_license_key()``, and do not set ``SC_NEUROCORE_LICENSE_KEY`` are not
    blocked and do not need the HTTP dependency.
    """

    stripped_key = key.strip()
    if not stripped_key:
        raise ValueError("Licence key must be a non-empty string")

    payload = {"key": stripped_key}
    if organization_id:
        payload["organization_id"] = organization_id

    response = (
        dict(transport(endpoint, payload, timeout))
        if transport is not None
        else _post_json_with_httpx(endpoint, payload, timeout=timeout)
    )
    return _status_from_polar_response(response, organization_id=organization_id)


def _status_from_polar_response(
    response: Mapping[str, Any],
    *,
    organization_id: str | None,
) -> CommercialLicenseStatus:
    valid = bool(response.get("valid"))
    license_key = response.get("license_key")
    license_payload = license_key if isinstance(license_key, Mapping) else {}
    status = _string_value(
        response.get("status")
        or license_payload.get("status")
        or ("active" if valid else "invalid")
    )
    benefits = response.get("benefits")
    benefit_payload = benefits if isinstance(benefits, Mapping) else {}
    message = _string_value(response.get("error") or response.get("message") or status)
    license_id = _optional_string(license_payload.get("id") or response.get("license_key_id"))
    priority_support = valid and bool(benefit_payload.get("priority_support"))

    return CommercialLicenseStatus(
        mode="commercial",
        valid=valid,
        status=status,
        license_id=license_id,
        organization_id=organization_id,
        commercial_enabled=valid,
        priority_support=priority_support,
        suppress_agpl_notice=valid,
        message=message,
    )


def _post_json_with_httpx(
    endpoint: str,
    payload: dict[str, str],
    *,
    timeout: float,
) -> dict[str, Any]:
    try:
        import httpx
    except ImportError as exc:
        raise SCDependencyError(
            "Install sc-neurocore[license] to validate commercial licence keys"
        ) from exc

    response = httpx.post(endpoint, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("Licence validation response must be a JSON object")
    return data


def _string_value(value: Any) -> str:
    return value if isinstance(value, str) else str(value)


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


__all__ = [
    "CommercialLicenseStatus",
    "LICENSE_KEY_ENV",
    "POLAR_LICENSE_VALIDATION_ENDPOINT",
    "POLAR_ORGANIZATION_ENV",
    "get_license_status",
    "load_license_from_env",
    "reset_license_status",
    "set_license_key",
    "validate_license_key",
]
