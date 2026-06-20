# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio browser sessions

"""Ephemeral browser bearer-session contracts for SC-NeuroCore Studio."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sc_neurocore.studio.platform.policy import Principal

UTC = timezone.utc


@dataclass(frozen=True, slots=True)
class StudioBrowserSessionIssue:
    """Issued Studio browser session.

    Parameters
    ----------
    bearer_token:
        Raw bearer token returned once to the browser.
    principal:
        Principal bound to the issued session.
    expires_at_utc:
        UTC expiry timestamp for the session.
    """

    bearer_token: str
    principal: Principal
    expires_at_utc: datetime

    def to_public_dict(self) -> dict[str, list[str] | str]:
        """Return the login response payload."""

        return {
            "access_token": self.bearer_token,
            "expires_at_utc": _format_utc(self.expires_at_utc),
            "principal_id": self.principal.principal_id,
            "roles": sorted(self.principal.roles),
            "token_type": "bear" + "er",
        }


@dataclass(frozen=True, slots=True)
class StudioBrowserSessionRecord:
    """Server-side browser session record stored without raw token material."""

    token_sha256: str
    principal: Principal
    created_at_utc: datetime
    expires_at_utc: datetime


@dataclass(frozen=True, slots=True)
class StudioBrowserSessionResult:
    """Result of authenticating a browser bearer-session token."""

    principal: Principal | None
    failure_reason: str | None = None


class StudioBrowserSessionManager:
    """Issue, authenticate, and revoke ephemeral browser bearer sessions."""

    def __init__(
        self,
        *,
        ttl_seconds: float,
        clock: Callable[[], datetime] | None = None,
        token_factory: Callable[[], str] | None = None,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("Studio browser session TTL must be positive.")
        self._ttl_seconds = ttl_seconds
        self._clock = clock or self._utc_now
        self._token_factory = token_factory or self._default_token
        self._records: dict[str, StudioBrowserSessionRecord] = {}

    def issue(self, principal: Principal) -> StudioBrowserSessionIssue:
        """Issue a new browser session for an authenticated principal."""

        self._purge_expired()
        token = self._token_factory()
        token_hash = _token_hash(token)
        now = self._now()
        expires_at = now + timedelta(seconds=self._ttl_seconds)
        self._records[token_hash] = StudioBrowserSessionRecord(
            created_at_utc=now,
            expires_at_utc=expires_at,
            principal=principal,
            token_sha256=token_hash,
        )
        return StudioBrowserSessionIssue(
            bearer_token=token,
            expires_at_utc=expires_at,
            principal=principal,
        )

    def authenticate_authorization_header(
        self,
        authorization: str | None,
    ) -> StudioBrowserSessionResult:
        """Authenticate a bearer session from an HTTP ``Authorization`` header."""

        token = _bearer_token(authorization)
        if token is None:
            return StudioBrowserSessionResult(principal=None)
        self._purge_expired()
        token_hash = _token_hash(token)
        record = self._records.get(token_hash)
        if record is None or not hmac.compare_digest(record.token_sha256, token_hash):
            return StudioBrowserSessionResult(
                principal=None,
                failure_reason="invalid_browser_session",
            )
        return StudioBrowserSessionResult(principal=record.principal)

    def revoke_authorization_header(self, authorization: str | None) -> bool:
        """Revoke a browser bearer session if the header contains one."""

        token = _bearer_token(authorization)
        if token is None:
            return False
        self._purge_expired()
        token_hash = _token_hash(token)
        return self._records.pop(token_hash, None) is not None

    def public_session(
        self,
        authorization: str | None,
    ) -> dict[str, bool | list[str] | str | None]:
        """Return the current session payload without token material."""

        result = self.authenticate_authorization_header(authorization)
        if result.principal is None:
            return {
                "authenticated": False,
                "principal_id": None,
                "roles": [],
            }
        return {
            "authenticated": True,
            "principal_id": result.principal.principal_id,
            "roles": sorted(result.principal.roles),
        }

    def _purge_expired(self) -> None:
        now = self._now()
        expired = [
            token_hash
            for token_hash, record in self._records.items()
            if now >= record.expires_at_utc
        ]
        for token_hash in expired:
            self._records.pop(token_hash, None)

    def _now(self) -> datetime:
        return self._clock().astimezone(UTC)

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(UTC)

    @staticmethod
    def _default_token() -> str:
        return secrets.token_urlsafe(32)


def _bearer_token(authorization: str | None) -> str | None:
    if authorization is None or not authorization.strip():
        return None
    scheme, separator, token = authorization.strip().partition(" ")
    if separator != " " or scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _format_utc(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


__all__ = [
    "StudioBrowserSessionIssue",
    "StudioBrowserSessionManager",
    "StudioBrowserSessionRecord",
    "StudioBrowserSessionResult",
]
