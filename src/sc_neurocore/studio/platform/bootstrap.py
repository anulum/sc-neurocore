# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio identity bootstrap

"""Offline identity bootstrap for SC-NeuroCore Studio."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sc_neurocore.studio.platform.identity import (
    IDENTITY_SCHEMA_VERSION,
    load_studio_identity_store,
)

DEFAULT_STUDIO_ADMIN_PRINCIPAL_ID = "svc-studio-admin"
DEFAULT_STUDIO_ADMIN_ROLES = ("studio.admin", "studio.viewer")
MIN_BOOTSTRAP_TOKEN_BYTES = 32
UTC = timezone.utc


@dataclass(frozen=True, slots=True)
class StudioIdentityBootstrapResult:
    """Result returned after creating a Studio service-account identity file.

    Parameters
    ----------
    identity_file_path:
        Destination JSON identity file path.
    principal_id:
        Service-account principal written to the identity file.
    roles:
        Roles granted to the bootstrap service account.
    bearer_token:
        One-time bearer token returned to the operator. It is never written to
        disk by the bootstrap routine.
    token_sha256:
        SHA-256 digest persisted in the identity file.
    expires_at_utc:
        Optional UTC expiry timestamp written to the identity file.
    file_permissions_hardened:
        Whether the routine successfully applied owner-only file permissions on
        platforms that expose POSIX file modes.
    parent_directory_created:
        Whether the routine created the identity file parent directory.
    """

    identity_file_path: Path
    principal_id: str
    roles: tuple[str, ...]
    bearer_token: str
    token_sha256: str
    expires_at_utc: str | None
    file_permissions_hardened: bool
    parent_directory_created: bool

    def to_public_dict(self) -> dict[str, bool | list[str] | str | None]:
        """Return bootstrap metadata without the bearer token."""

        return {
            "expires_at_utc": self.expires_at_utc,
            "file_permissions_hardened": self.file_permissions_hardened,
            "identity_file_path": str(self.identity_file_path),
            "parent_directory_created": self.parent_directory_created,
            "principal_id": self.principal_id,
            "roles": list(self.roles),
            "schema_version": IDENTITY_SCHEMA_VERSION,
            "token_sha256": self.token_sha256,
        }


def bootstrap_studio_admin_identity(
    identity_file_path: Path,
    *,
    principal_id: str = DEFAULT_STUDIO_ADMIN_PRINCIPAL_ID,
    roles: Sequence[str] = DEFAULT_STUDIO_ADMIN_ROLES,
    token_bytes: int = MIN_BOOTSTRAP_TOKEN_BYTES,
    expires_at_utc: str | None = None,
    overwrite: bool = False,
    token_factory: Callable[[int], str] = secrets.token_urlsafe,
) -> StudioIdentityBootstrapResult:
    """Create a local Studio admin identity file for first deployment.

    Parameters
    ----------
    identity_file_path:
        Destination for the JSON identity file consumed by
        ``SC_NEUROCORE_STUDIO_IDENTITY_FILE``.
    principal_id:
        Stable service-account principal recorded in audit rows.
    roles:
        Non-empty role set granted to the service account.
    token_bytes:
        Entropy bytes requested from ``token_factory``. Values below
        ``MIN_BOOTSTRAP_TOKEN_BYTES`` are rejected.
    expires_at_utc:
        Optional ISO-8601 timestamp. Values are normalised to UTC with a ``Z``
        suffix before being written.
    overwrite:
        Whether an existing identity file may be replaced atomically.
    token_factory:
        Token generator hook. Production callers use ``secrets.token_urlsafe``;
        tests may inject a deterministic generator.

    Returns
    -------
    StudioIdentityBootstrapResult
        Bootstrap metadata plus the one-time bearer token.

    Raises
    ------
    FileExistsError
        If the destination exists and ``overwrite`` is false.
    ValueError
        If identity metadata is malformed.
    OSError
        If the destination cannot be written.
    """

    destination = identity_file_path.expanduser()
    if destination.exists() and not overwrite:
        raise FileExistsError("Studio identity file already exists.")

    clean_principal_id = _parse_principal_id(principal_id)
    clean_roles = _parse_roles(roles)
    if token_bytes < MIN_BOOTSTRAP_TOKEN_BYTES:
        raise ValueError(
            f"Studio bootstrap token requires at least {MIN_BOOTSTRAP_TOKEN_BYTES} bytes."
        )
    normalised_expiry = _normalise_expiry(expires_at_utc)
    parent_created = _ensure_parent_directory(destination.parent)
    bearer_token = token_factory(token_bytes)
    if not isinstance(bearer_token, str) or not bearer_token.strip():
        raise ValueError("Studio bootstrap token factory returned an invalid token.")
    token = bearer_token.strip()
    token_sha256 = hashlib.sha256(token.encode("utf-8")).hexdigest()
    payload = _build_identity_payload(
        principal_id=clean_principal_id,
        roles=clean_roles,
        token_sha256=token_sha256,
        expires_at_utc=normalised_expiry,
    )
    permissions_hardened = _write_identity_payload(destination, payload, overwrite=overwrite)
    load_studio_identity_store(destination)
    return StudioIdentityBootstrapResult(
        identity_file_path=destination,
        principal_id=clean_principal_id,
        roles=clean_roles,
        bearer_token=token,
        token_sha256=token_sha256,
        expires_at_utc=normalised_expiry,
        file_permissions_hardened=permissions_hardened,
        parent_directory_created=parent_created,
    )


def _parse_principal_id(principal_id: str) -> str:
    cleaned = principal_id.strip()
    if not cleaned:
        raise ValueError("Studio bootstrap principal_id must not be empty.")
    if any(character.isspace() for character in cleaned):
        raise ValueError("Studio bootstrap principal_id must not contain whitespace.")
    return cleaned


def _parse_roles(roles: Sequence[str]) -> tuple[str, ...]:
    cleaned_roles: list[str] = []
    for role in roles:
        cleaned = role.strip()
        if not cleaned:
            raise ValueError("Studio bootstrap roles must not be empty.")
        if cleaned not in cleaned_roles:
            cleaned_roles.append(cleaned)
    if not cleaned_roles:
        raise ValueError("Studio bootstrap requires at least one role.")
    if "studio.admin" not in cleaned_roles:
        raise ValueError("Studio bootstrap admin identity requires the studio.admin role.")
    return tuple(cleaned_roles)


def _normalise_expiry(expires_at_utc: str | None) -> str | None:
    if expires_at_utc is None:
        return None
    raw_value = expires_at_utc.strip()
    if not raw_value:
        raise ValueError("Studio bootstrap expiry must not be empty.")
    normalised = raw_value[:-1] + "+00:00" if raw_value.endswith("Z") else raw_value
    try:
        parsed = datetime.fromisoformat(normalised)
    except ValueError as exc:
        raise ValueError("Studio bootstrap expiry must be an ISO-8601 timestamp.") from exc
    if parsed.tzinfo is None:
        raise ValueError("Studio bootstrap expiry must include a timezone.")
    return parsed.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _ensure_parent_directory(parent: Path) -> bool:
    if parent.exists():
        if not parent.is_dir():
            raise ValueError("Studio identity parent path must be a directory.")
        return False
    parent.mkdir(mode=0o700, parents=True, exist_ok=False)
    _chmod_owner_only(parent, directory=True)
    return True


def _build_identity_payload(
    *,
    principal_id: str,
    roles: tuple[str, ...],
    token_sha256: str,
    expires_at_utc: str | None,
) -> dict[str, object]:
    account: dict[str, object] = {
        "active": True,
        "principal_id": principal_id,
        "roles": list(roles),
        "token_sha256": token_sha256,
    }
    if expires_at_utc is not None:
        account["expires_at_utc"] = expires_at_utc
    return {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "service_accounts": [account],
    }


def _write_identity_payload(
    destination: Path,
    payload: dict[str, object],
    *,
    overwrite: bool,
) -> bool:
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if overwrite:
        return _replace_identity_payload(destination, encoded)
    return _create_identity_payload(destination, encoded)


def _create_identity_payload(destination: Path, encoded: str) -> bool:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    descriptor = os.open(destination, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.write("\n")
    except BaseException:
        try:
            destination.unlink()
        finally:
            raise
    return _chmod_owner_only(destination, directory=False)


def _replace_identity_payload(destination: Path, encoded: str) -> bool:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.write("\n")
        permissions_hardened = _chmod_owner_only(temporary_path, directory=False)
        os.replace(temporary_path, destination)
        return permissions_hardened and _chmod_owner_only(destination, directory=False)
    except BaseException:
        try:
            temporary_path.unlink(missing_ok=True)
        finally:
            raise


def _chmod_owner_only(path: Path, *, directory: bool) -> bool:
    if os.name != "posix":
        return False
    path.chmod(0o700 if directory else 0o600)
    return True


__all__ = [
    "DEFAULT_STUDIO_ADMIN_PRINCIPAL_ID",
    "DEFAULT_STUDIO_ADMIN_ROLES",
    "MIN_BOOTSTRAP_TOKEN_BYTES",
    "StudioIdentityBootstrapResult",
    "bootstrap_studio_admin_identity",
]
