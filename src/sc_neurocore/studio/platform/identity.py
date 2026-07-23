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
import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sc_neurocore.studio.platform.policy import Principal
from sc_neurocore.studio.platform.identity_passwords import (
    DEFAULT_BROWSER_USER_PASSWORD_ITERATIONS,
    make_browser_user_password_verifier,
    verify_browser_user_password,
    _parse_password_verifier,
)

IDENTITY_SCHEMA_VERSION = "sc-neurocore.studio.identity.v1"
UTC = timezone.utc
_ADMIN_ROLE = "studio.admin"


class StudioIdentityLifecycleError(ValueError):
    """Raised when an identity mutation would break lifecycle invariants."""


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

    def to_public_record(self) -> "StudioIdentityPublicRecord":
        """Return a token-free operator representation of this identity."""

        expiry = (
            None
            if self.expires_at_utc is None
            else self.expires_at_utc.isoformat().replace("+00:00", "Z")
        )
        return StudioIdentityPublicRecord(
            active=self.active,
            expires_at_utc=expiry,
            principal_id=self.principal_id,
            roles=tuple(sorted(self.roles)),
        )


@dataclass(frozen=True, slots=True)
class StudioIdentityPublicRecord:
    """Path-free and token-free service-account record for operators.

    Parameters
    ----------
    principal_id:
        Stable service-account identifier.
    roles:
        Roles granted to the service account.
    expires_at_utc:
        Optional UTC expiry instant formatted as an ISO timestamp.
    active:
        Whether the service account can authenticate.
    """

    principal_id: str
    roles: tuple[str, ...]
    expires_at_utc: str | None
    active: bool

    def to_public_dict(self) -> dict[str, bool | list[str] | str | None]:
        """Return an API payload without token hashes or local paths."""

        return {
            "active": self.active,
            "expires_at_utc": self.expires_at_utc,
            "principal_id": self.principal_id,
            "roles": list(self.roles),
        }


@dataclass(frozen=True, slots=True)
class StudioBrowserUserRecord:
    """Persistent browser-login identity for Studio operators.

    Parameters
    ----------
    username:
        Stable browser-login username.
    principal_id:
        Principal identifier recorded in audit events after login.
    roles:
        Role names granted to the browser user.
    password_pbkdf2_sha256:
        Encoded PBKDF2-HMAC-SHA256 password verifier.
    expires_at_utc:
        Optional UTC expiry instant. Expired users cannot log in.
    active:
        Whether the user can currently authenticate.
    """

    username: str
    principal_id: str
    roles: frozenset[str]
    password_pbkdf2_sha256: str
    expires_at_utc: datetime | None = None
    active: bool = True

    def to_public_record(self) -> "StudioBrowserUserPublicRecord":
        """Return a password-free operator representation of this user."""

        expiry = (
            None
            if self.expires_at_utc is None
            else self.expires_at_utc.isoformat().replace("+00:00", "Z")
        )
        return StudioBrowserUserPublicRecord(
            active=self.active,
            expires_at_utc=expiry,
            principal_id=self.principal_id,
            roles=tuple(sorted(self.roles)),
            username=self.username,
        )


@dataclass(frozen=True, slots=True)
class StudioBrowserUserPublicRecord:
    """Path-free and password-free browser-user record for operators."""

    username: str
    principal_id: str
    roles: tuple[str, ...]
    expires_at_utc: str | None
    active: bool

    def to_public_dict(self) -> dict[str, bool | list[str] | str | None]:
        """Return an API payload without password verifier material."""

        return {
            "active": self.active,
            "expires_at_utc": self.expires_at_utc,
            "principal_id": self.principal_id,
            "roles": list(self.roles),
            "username": self.username,
        }


@dataclass(frozen=True, slots=True)
class StudioIdentityStore:
    """Validated Studio identity store loaded from a local JSON file."""

    service_accounts: tuple[StudioIdentityRecord, ...]
    browser_users: tuple[StudioBrowserUserRecord, ...] = ()

    def public_records(self) -> tuple[StudioIdentityPublicRecord, ...]:
        """Return token-free service-account records for admin APIs."""

        return tuple(record.to_public_record() for record in self.service_accounts)

    def public_browser_users(self) -> tuple[StudioBrowserUserPublicRecord, ...]:
        """Return password-free browser-user records for admin APIs."""

        return tuple(record.to_public_record() for record in self.browser_users)


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

    def authenticate_browser_user(self, username: str, password: str) -> StudioIdentityResult:
        """Authenticate one browser user with username and password.

        Parameters
        ----------
        username:
            Browser-login username.
        password:
            Raw password supplied by the browser login form.

        Returns
        -------
        StudioIdentityResult
            Authenticated principal or a stable failure reason for audit rows.
        """

        clean_username = username.strip()
        if not clean_username or not password:
            return StudioIdentityResult(principal=None, failure_reason="invalid_browser_login")
        for record in self._identity_store.browser_users:
            if not hmac.compare_digest(record.username, clean_username):
                continue
            if not record.active:
                return StudioIdentityResult(
                    principal=None,
                    failure_reason="disabled_browser_user",
                )
            if record.expires_at_utc is not None and self._utc_now_value() >= record.expires_at_utc:
                return StudioIdentityResult(
                    principal=None,
                    failure_reason="expired_browser_user",
                )
            if not verify_browser_user_password(password, record.password_pbkdf2_sha256):
                return StudioIdentityResult(
                    principal=None,
                    failure_reason="invalid_browser_login",
                )
            return StudioIdentityResult(
                principal=Principal(principal_id=record.principal_id, roles=record.roles)
            )
        return StudioIdentityResult(principal=None, failure_reason="invalid_browser_login")

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
    raw_browser_users = payload.get("browser_users", [])
    if not isinstance(raw_browser_users, list):
        raise ValueError("Studio identity browser_users must be a list.")
    browser_users = tuple(
        _parse_browser_user_record(index, item) for index, item in enumerate(raw_browser_users)
    )
    return StudioIdentityStore(service_accounts=records, browser_users=browser_users)


def list_studio_identity_public_records(path: Path) -> tuple[StudioIdentityPublicRecord, ...]:
    """Load token-free Studio service-account records from an identity file.

    Parameters
    ----------
    path:
        Persistent identity JSON file.

    Returns
    -------
    tuple[StudioIdentityPublicRecord, ...]
        Public service-account records sorted by principal identifier.
    """

    return tuple(
        sorted(
            load_studio_identity_store(path).public_records(),
            key=lambda record: record.principal_id,
        )
    )


def list_studio_browser_user_public_records(
    path: Path,
) -> tuple[StudioBrowserUserPublicRecord, ...]:
    """Load password-free Studio browser-user records from an identity file.

    Parameters
    ----------
    path:
        Persistent identity JSON file.

    Returns
    -------
    tuple[StudioBrowserUserPublicRecord, ...]
        Public browser-user records sorted by username.
    """

    return tuple(
        sorted(
            load_studio_identity_store(path).public_browser_users(),
            key=lambda record: record.username,
        )
    )


def update_studio_identity_record(
    path: Path,
    *,
    principal_id: str,
    roles: Sequence[str],
    active: bool,
    expires_at_utc: str | None,
) -> StudioIdentityPublicRecord:
    """Atomically update mutable service-account metadata.

    The raw bearer-token SHA-256 digest is preserved and never returned.

    Parameters
    ----------
    path:
        Persistent identity JSON file.
    principal_id:
        Existing service-account identifier to update.
    roles:
        Replacement role set. The set must be non-empty.
    active:
        Replacement active flag.
    expires_at_utc:
        Optional replacement UTC expiry timestamp.

    Returns
    -------
    StudioIdentityPublicRecord
        Updated token-free service-account record.

    Raises
    ------
    KeyError
        If ``principal_id`` is not present in the store.
    ValueError
        If replacement metadata is malformed.
    """

    store = load_studio_identity_store(path)
    clean_principal_id = _parse_principal_id(principal_id)
    clean_roles = frozenset(_parse_roles(roles))
    clean_expires_at = _parse_expiry(expires_at_utc)
    updated: StudioIdentityRecord | None = None
    records: list[StudioIdentityRecord] = []
    for record in store.service_accounts:
        if record.principal_id == clean_principal_id:
            updated = StudioIdentityRecord(
                active=active,
                expires_at_utc=clean_expires_at,
                principal_id=record.principal_id,
                roles=clean_roles,
                token_sha256=record.token_sha256,
            )
            records.append(updated)
        else:
            records.append(record)
    if updated is None:
        raise KeyError(clean_principal_id)
    _require_active_admin_principal(
        service_accounts=tuple(records),
        browser_users=store.browser_users,
    )
    _write_identity_store(path, service_accounts=tuple(records), browser_users=store.browser_users)
    return updated.to_public_record()


def update_studio_browser_user_record(
    path: Path,
    *,
    username: str,
    roles: Sequence[str],
    active: bool,
    expires_at_utc: str | None,
) -> StudioBrowserUserPublicRecord:
    """Atomically update mutable browser-user metadata.

    The stored PBKDF2-HMAC-SHA256 password verifier is preserved and never
    returned.

    Parameters
    ----------
    path:
        Persistent identity JSON file.
    username:
        Existing browser-login username to update.
    roles:
        Replacement role set. The set must be non-empty.
    active:
        Replacement active flag.
    expires_at_utc:
        Optional replacement UTC expiry timestamp.

    Returns
    -------
    StudioBrowserUserPublicRecord
        Updated password-free browser-user record.

    Raises
    ------
    KeyError
        If ``username`` is not present in the store.
    ValueError
        If replacement metadata is malformed.
    """

    store = load_studio_identity_store(path)
    clean_username = _parse_username(username)
    clean_roles = frozenset(_parse_roles(roles))
    clean_expires_at = _parse_expiry(expires_at_utc)
    updated: StudioBrowserUserRecord | None = None
    records: list[StudioBrowserUserRecord] = []
    for record in store.browser_users:
        if record.username == clean_username:
            updated = StudioBrowserUserRecord(
                active=active,
                expires_at_utc=clean_expires_at,
                password_pbkdf2_sha256=record.password_pbkdf2_sha256,
                principal_id=record.principal_id,
                roles=clean_roles,
                username=record.username,
            )
            records.append(updated)
        else:
            records.append(record)
    if updated is None:
        raise KeyError(clean_username)
    _require_active_admin_principal(
        service_accounts=store.service_accounts,
        browser_users=tuple(records),
    )
    _write_identity_store(
        path,
        service_accounts=store.service_accounts,
        browser_users=tuple(records),
    )
    return updated.to_public_record()


def rotate_studio_browser_user_password(
    path: Path,
    *,
    username: str,
    password: str,
) -> StudioBrowserUserPublicRecord:
    """Atomically rotate one browser user's password verifier.

    Parameters
    ----------
    path:
        Persistent identity JSON file.
    username:
        Existing browser-login username to update.
    password:
        New raw password supplied through an authenticated admin request.

    Returns
    -------
    StudioBrowserUserPublicRecord
        Password-free browser-user record after verifier rotation.

    Raises
    ------
    KeyError
        If ``username`` is not present in the store.
    ValueError
        If the username or replacement password is malformed.
    """

    store = load_studio_identity_store(path)
    clean_username = _parse_username(username)
    updated: StudioBrowserUserRecord | None = None
    records: list[StudioBrowserUserRecord] = []
    for record in store.browser_users:
        if record.username == clean_username:
            updated = StudioBrowserUserRecord(
                active=record.active,
                expires_at_utc=record.expires_at_utc,
                password_pbkdf2_sha256=make_browser_user_password_verifier(password),
                principal_id=record.principal_id,
                roles=record.roles,
                username=record.username,
            )
            records.append(updated)
        else:
            records.append(record)
    if updated is None:
        raise KeyError(clean_username)
    _write_identity_store(
        path,
        service_accounts=store.service_accounts,
        browser_users=tuple(records),
    )
    return updated.to_public_record()


def add_studio_browser_user_record(
    path: Path,
    *,
    username: str,
    principal_id: str,
    roles: Sequence[str],
    password: str,
    active: bool = True,
    expires_at_utc: str | None = None,
) -> StudioBrowserUserPublicRecord:
    """Atomically add one persistent browser-login user to an identity file.

    Parameters
    ----------
    path:
        Persistent identity JSON file.
    username:
        Unique browser-login username.
    principal_id:
        Stable principal identifier recorded in policy audit events.
    roles:
        Non-empty role set granted after login.
    password:
        Raw password read from an operator-controlled secret channel.
    active:
        Whether the new browser user can authenticate immediately.
    expires_at_utc:
        Optional UTC expiry timestamp.

    Returns
    -------
    StudioBrowserUserPublicRecord
        Password-free representation of the new browser user.

    Raises
    ------
    ValueError
        If the new user metadata is malformed or conflicts with an existing
        browser username.
    """

    store = load_studio_identity_store(path)
    clean_username = _parse_username(username)
    if any(record.username == clean_username for record in store.browser_users):
        raise ValueError("Studio browser user username already exists.")
    record = StudioBrowserUserRecord(
        active=active,
        expires_at_utc=_parse_expiry(expires_at_utc),
        password_pbkdf2_sha256=make_browser_user_password_verifier(password),
        principal_id=_parse_principal_id(principal_id),
        roles=frozenset(_parse_roles(roles)),
        username=clean_username,
    )
    _write_identity_store(
        path,
        service_accounts=store.service_accounts,
        browser_users=(*store.browser_users, record),
    )
    return record.to_public_record()


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


def _parse_browser_user_record(index: int, item: object) -> StudioBrowserUserRecord:
    if not isinstance(item, dict):
        raise ValueError(f"Studio identity browser user {index} must be an object.")
    username = item.get("username")
    if not isinstance(username, str) or not username.strip():
        raise ValueError("Studio browser user username must be a non-empty string.")
    principal_id = item.get("principal_id")
    if not isinstance(principal_id, str) or not principal_id.strip():
        raise ValueError("Studio browser user principal_id must be a non-empty string.")
    raw_roles = item.get("roles")
    if not isinstance(raw_roles, list) or not raw_roles:
        raise ValueError("Studio browser user roles must be a non-empty list.")
    roles = frozenset(_parse_role(role) for role in raw_roles)
    password_verifier = item.get("password_pbkdf2_sha256")
    if (
        not isinstance(password_verifier, str)
        or _parse_password_verifier(password_verifier) is None
    ):
        raise ValueError("Studio browser user password verifier is invalid.")
    raw_active = item.get("active", True)
    if not isinstance(raw_active, bool):
        raise ValueError("Studio browser user active flag must be boolean.")
    expires_at_utc = _parse_expiry(item.get("expires_at_utc"))
    return StudioBrowserUserRecord(
        active=raw_active,
        expires_at_utc=expires_at_utc,
        password_pbkdf2_sha256=password_verifier,
        principal_id=principal_id.strip(),
        roles=roles,
        username=username.strip(),
    )


def _parse_principal_id(principal_id: str) -> str:
    cleaned = principal_id.strip()
    if not cleaned:
        raise ValueError("Studio identity principal_id must be a non-empty string.")
    return cleaned


def _parse_username(username: str) -> str:
    cleaned = username.strip()
    if not cleaned:
        raise ValueError("Studio browser user username must be a non-empty string.")
    if any(character.isspace() for character in cleaned):
        raise ValueError("Studio browser user username must not contain whitespace.")
    return cleaned


def _parse_roles(roles: Sequence[str]) -> tuple[str, ...]:
    if not roles:
        raise ValueError("Studio identity roles must be a non-empty list.")
    cleaned: list[str] = []
    for role in roles:
        parsed = _parse_role(role)
        if parsed not in cleaned:
            cleaned.append(parsed)
    return tuple(cleaned)


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


def _write_identity_store(
    path: Path,
    *,
    service_accounts: tuple[StudioIdentityRecord, ...],
    browser_users: tuple[StudioBrowserUserRecord, ...],
) -> None:
    payload = {
        "browser_users": [_browser_user_to_json(record) for record in browser_users],
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "service_accounts": [_record_to_json(record) for record in service_accounts],
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    parent = path.parent
    if parent.exists() and not parent.is_dir():
        raise ValueError("Studio identity parent path must be a directory.")
    parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp_path = tempfile.mkstemp(
        dir=parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    tmp_path = Path(raw_tmp_path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            tmp_file.write(encoded)
            tmp_file.write("\n")
        path_permissions = path.stat().st_mode & 0o777 if path.exists() else 0o600
        tmp_path.chmod(path_permissions)
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _require_active_admin_principal(
    *,
    service_accounts: tuple[StudioIdentityRecord, ...],
    browser_users: tuple[StudioBrowserUserRecord, ...],
    now: datetime | None = None,
) -> None:
    clock = datetime.now(UTC) if now is None else now.astimezone(UTC)
    if any(_active_admin_service_account(record, now=clock) for record in service_accounts):
        return
    if any(_active_admin_browser_user(record, now=clock) for record in browser_users):
        return
    raise StudioIdentityLifecycleError(
        "Studio identity updates must leave at least one active unexpired studio.admin principal."
    )


def _active_admin_service_account(record: StudioIdentityRecord, *, now: datetime) -> bool:
    if not record.active or _ADMIN_ROLE not in record.roles:
        return False
    return record.expires_at_utc is None or now < record.expires_at_utc


def _active_admin_browser_user(record: StudioBrowserUserRecord, *, now: datetime) -> bool:
    if not record.active or _ADMIN_ROLE not in record.roles:
        return False
    return record.expires_at_utc is None or now < record.expires_at_utc


def _record_to_json(record: StudioIdentityRecord) -> dict[str, bool | list[str] | str | None]:
    expiry = (
        None
        if record.expires_at_utc is None
        else record.expires_at_utc.isoformat().replace("+00:00", "Z")
    )
    return {
        "active": record.active,
        "expires_at_utc": expiry,
        "principal_id": record.principal_id,
        "roles": sorted(record.roles),
        "token_sha256": record.token_sha256,
    }


def _browser_user_to_json(
    record: StudioBrowserUserRecord,
) -> dict[str, bool | list[str] | str | None]:
    expiry = (
        None
        if record.expires_at_utc is None
        else record.expires_at_utc.isoformat().replace("+00:00", "Z")
    )
    return {
        "active": record.active,
        "expires_at_utc": expiry,
        "password_pbkdf2_sha256": record.password_pbkdf2_sha256,
        "principal_id": record.principal_id,
        "roles": sorted(record.roles),
        "username": record.username,
    }


__all__ = [
    "DEFAULT_BROWSER_USER_PASSWORD_ITERATIONS",
    "IDENTITY_SCHEMA_VERSION",
    "StudioBrowserUserPublicRecord",
    "StudioBrowserUserRecord",
    "StudioIdentityAuthenticator",
    "StudioIdentityLifecycleError",
    "StudioIdentityPublicRecord",
    "StudioIdentityRecord",
    "StudioIdentityResult",
    "StudioIdentityStore",
    "add_studio_browser_user_record",
    "list_studio_browser_user_public_records",
    "list_studio_identity_public_records",
    "load_studio_identity_store",
    "make_browser_user_password_verifier",
    "rotate_studio_browser_user_password",
    "update_studio_browser_user_record",
    "update_studio_identity_record",
    "verify_browser_user_password",
]
