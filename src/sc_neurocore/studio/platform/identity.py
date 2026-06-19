# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio identity store

"""Persistent service-account identity contracts for SC-NeuroCore Studio."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sc_neurocore.studio.platform.policy import Principal

IDENTITY_SCHEMA_VERSION = "sc-neurocore.studio.identity.v1"
UTC = timezone.utc


@dataclass(frozen=True, slots=True)
class StudioIdentityRecord:
    """Persistent Studio service-account identity.

    Parameters
    ----------
    principal_id:
        Stable service-account identifier recorded in policy audit events.
    roles:
        Role names granted to the service account.
    token_sha256:
        Lowercase SHA-256 hex digest of the bearer token. Raw tokens are never
        stored in the identity file.
    expires_at_utc:
        Optional UTC expiry instant. Expired records fail authentication.
    active:
        Whether the identity is currently allowed to authenticate.
    """

    principal_id: str
    roles: frozenset[str]
    token_sha256: str
    expires_at_utc: datetime | None = None
    active: bool = True


@dataclass(frozen=True, slots=True)
class StudioIdentityStore:
    """Validated Studio identity store loaded from a local JSON file."""

    service_accounts: tuple[StudioIdentityRecord, ...]


@dataclass(frozen=True, slots=True)
class StudioIdentityResult:
    """Result of authenticating one Studio authorization header."""

    principal: Principal | None
    failure_reason: str | None = None


class StudioIdentityAuthenticator:
    """Authenticate Studio bearer tokens against a validated identity store."""

    def __init__(
        self,
        identity_store: StudioIdentityStore,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._identity_store = identity_store
        self._clock = clock or self._utc_now

    def authenticate_authorization_header(self, authorization: str | None) -> StudioIdentityResult:
        """Authenticate an HTTP ``Authorization`` header.

        Parameters
        ----------
        authorization:
            Raw HTTP authorization header value. Missing values return a neutral
            result so callers can fall back to development-preview identities.

        Returns
        -------
        StudioIdentityResult
            Authenticated principal or a stable failure reason for policy
            auditing.
        """

        if authorization is None or not authorization.strip():
            return StudioIdentityResult(principal=None)
        scheme, separator, token = authorization.strip().partition(" ")
        if separator != " " or scheme.lower() != "bearer" or not token.strip():
            return StudioIdentityResult(principal=None, failure_reason="invalid_identity_token")
        token_hash = hashlib.sha256(token.strip().encode("utf-8")).hexdigest()
        for record in self._identity_store.service_accounts:
            if not hmac.compare_digest(record.token_sha256, token_hash):
                continue
            if not record.active:
                return StudioIdentityResult(
                    principal=None, failure_reason="disabled_identity_token"
                )
            if record.expires_at_utc is not None and self._utc_now_value() >= record.expires_at_utc:
                return StudioIdentityResult(principal=None, failure_reason="expired_identity_token")
            return StudioIdentityResult(
                principal=Principal(principal_id=record.principal_id, roles=record.roles)
            )
        return StudioIdentityResult(principal=None, failure_reason="invalid_identity_token")

    def _utc_now_value(self) -> datetime:
        return self._clock().astimezone(UTC)

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(UTC)


def load_studio_identity_store(path: Path) -> StudioIdentityStore:
    """Load and validate a Studio identity store from JSON.

    Parameters
    ----------
    path:
        JSON file containing ``sc-neurocore.studio.identity.v1`` service
        account records.

    Returns
    -------
    StudioIdentityStore
        Validated immutable service-account records.

    Raises
    ------
    ValueError
        If the file cannot be parsed into the supported identity schema.
    """

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError("Studio identity file cannot be read.") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("Studio identity file must be valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Studio identity schema must be a JSON object.")
    if payload.get("schema_version") != IDENTITY_SCHEMA_VERSION:
        raise ValueError("Studio identity schema version is not supported.")
    raw_accounts = payload.get("service_accounts")
    if not isinstance(raw_accounts, list):
        raise ValueError("Studio identity service_accounts must be a list.")
    records = tuple(_parse_identity_record(index, item) for index, item in enumerate(raw_accounts))
    return StudioIdentityStore(service_accounts=records)


def _parse_identity_record(index: int, item: object) -> StudioIdentityRecord:
    if not isinstance(item, dict):
        raise ValueError(f"Studio identity service account {index} must be an object.")
    principal_id = item.get("principal_id")
    if not isinstance(principal_id, str) or not principal_id.strip():
        raise ValueError("Studio identity principal_id must be a non-empty string.")
    raw_roles = item.get("roles")
    if not isinstance(raw_roles, list) or not raw_roles:
        raise ValueError("Studio identity roles must be a non-empty list.")
    roles = frozenset(_parse_role(role) for role in raw_roles)
    token_sha256 = item.get("token_sha256")
    if not isinstance(token_sha256, str) or not _is_sha256_hex(token_sha256):
        raise ValueError("Studio identity token_sha256 must be a SHA-256 hex digest.")
    raw_active = item.get("active", True)
    if not isinstance(raw_active, bool):
        raise ValueError("Studio identity active flag must be boolean.")
    expires_at_utc = _parse_expiry(item.get("expires_at_utc"))
    return StudioIdentityRecord(
        principal_id=principal_id.strip(),
        roles=roles,
        token_sha256=token_sha256.lower(),
        expires_at_utc=expires_at_utc,
        active=raw_active,
    )


def _parse_role(role: object) -> str:
    if not isinstance(role, str) or not role.strip():
        raise ValueError("Studio identity roles must be non-empty strings.")
    return role.strip()


def _parse_expiry(value: object) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Studio identity expires_at_utc must be a UTC timestamp.")
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("Studio identity expires_at_utc must be an ISO timestamp.") from exc
    if parsed.tzinfo is None:
        raise ValueError("Studio identity expires_at_utc must include a timezone.")
    return parsed.astimezone(UTC)


def _is_sha256_hex(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdefABCDEF" for character in value)


__all__ = [
    "IDENTITY_SCHEMA_VERSION",
    "StudioIdentityAuthenticator",
    "StudioIdentityRecord",
    "StudioIdentityResult",
    "StudioIdentityStore",
    "load_studio_identity_store",
]
