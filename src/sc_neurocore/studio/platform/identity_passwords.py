# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio browser-user password verifiers

"""PBKDF2-HMAC-SHA256 password verifiers for Studio browser users.

Separated from identity store lifecycle so credential hashing can be tested and
evolved without loading the full identity persistence surface.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

from sc_neurocore.studio.platform.identity_refusals import StudioIdentityRefused

DEFAULT_BROWSER_USER_PASSWORD_ITERATIONS = 390_000
MIN_BROWSER_USER_PASSWORD_ITERATIONS = 100_000
MIN_BROWSER_USER_PASSWORD_SALT_BYTES = 16


def _is_sha256_hex(value: str) -> bool:
    """Return whether ``value`` is a 64-character lowercase/upper hex digest."""
    return len(value) == 64 and all(character in "0123456789abcdefABCDEF" for character in value)


def make_browser_user_password_verifier(password: str) -> str:
    """Create an encoded PBKDF2-HMAC-SHA256 password verifier.

    Parameters
    ----------
    password:
        Raw browser-user password.

    Returns
    -------
    str
        Encoded verifier containing algorithm, iteration count, salt, and hash.
    """
    if not password:
        raise StudioIdentityRefused("Studio browser-user password must not be empty.")
    salt = secrets.token_hex(MIN_BROWSER_USER_PASSWORD_SALT_BYTES)
    password_hash = _pbkdf2_sha256(password, salt, DEFAULT_BROWSER_USER_PASSWORD_ITERATIONS)
    return f"pbkdf2_sha256${DEFAULT_BROWSER_USER_PASSWORD_ITERATIONS}${salt}${password_hash}"


def verify_browser_user_password(
    password: str, encoded_verifier: str, *, minimum_iterations: int = 0
) -> bool:
    """Verify a stored digest while completing the caller's PBKDF2 work floor.

    Parameters
    ----------
    password:
        Supplied password, including a legacy password awaiting rotation.
    encoded_verifier:
        Stored algorithm, iteration count, salt and expected digest.
    minimum_iterations:
        Total hash work required for this attempt. Extra iterations do not
        change the legacy digest or its password acceptance. Zero preserves
        the stored cost for standalone verifier callers.

    Returns
    -------
    bool
        Whether the supplied password matches a valid stored verifier.
    """
    if minimum_iterations < 0:
        raise StudioIdentityRefused("Studio password verifier work floor must be non-negative.")

    parsed = _parse_password_verifier(encoded_verifier)
    if parsed is None:
        return False
    iterations, salt, expected_hash = parsed
    candidate = _pbkdf2_sha256(password, salt, iterations)
    if iterations < minimum_iterations:
        _pbkdf2_sha256(password, salt, minimum_iterations - iterations)
    return hmac.compare_digest(candidate, expected_hash)


def _pbkdf2_sha256(password: str, salt_hex: str, iterations: int) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt_hex),
        iterations,
    ).hex()


def _parse_password_verifier(value: str) -> tuple[int, str, str] | None:
    parts = value.split("$")
    if len(parts) != 4:
        return None
    algorithm, raw_iterations, salt_hex, digest_hex = parts
    if algorithm != "pbkdf2_sha256":
        return None
    try:
        iterations = int(raw_iterations)
        bytes.fromhex(salt_hex)
    except ValueError:
        return None
    if iterations < MIN_BROWSER_USER_PASSWORD_ITERATIONS:
        return None
    if len(salt_hex) < MIN_BROWSER_USER_PASSWORD_SALT_BYTES * 2:
        return None
    if not _is_sha256_hex(digest_hex):
        return None
    return iterations, salt_hex, digest_hex
