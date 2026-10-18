# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio identity password verifier unit tests

"""Behaviour tests for Studio browser-user password verifiers."""

from __future__ import annotations

import hashlib

import pytest

from sc_neurocore.studio.platform.identity_passwords import (
    DEFAULT_BROWSER_USER_PASSWORD_ITERATIONS,
    _parse_password_verifier,
    make_browser_user_password_verifier,
    verify_browser_user_password,
)


def test_make_and_verify_round_trip() -> None:
    """A verifier accepts the correct password and rejects mismatches."""
    verifier = make_browser_user_password_verifier("correct-horse")
    assert verifier.startswith(f"pbkdf2_sha256${DEFAULT_BROWSER_USER_PASSWORD_ITERATIONS}$")
    assert verify_browser_user_password("correct-horse", verifier)
    assert not verify_browser_user_password("wrong-horse", verifier)


def test_empty_password_rejected() -> None:
    """Empty passwords are refuse-closed at encode time."""
    with pytest.raises(ValueError, match="must not be empty"):
        make_browser_user_password_verifier("")


def test_negative_verifier_work_budget_is_refused() -> None:
    """Reject an invalid verification work floor while preserving the stored digest."""
    verifier = make_browser_user_password_verifier("correct-horse")
    with pytest.raises(ValueError, match="work floor must be non-negative"):
        verify_browser_user_password("correct-horse", verifier, minimum_iterations=-1)
    assert verify_browser_user_password("correct-horse", verifier)


def test_parse_rejects_malformed_verifiers() -> None:
    """Malformed or under-iterated verifiers fail closed."""
    assert _parse_password_verifier("sha256$1$bad") is None
    assert _parse_password_verifier("pbkdf2_sha256$1$aa$bb") is None
    assert not verify_browser_user_password("x", "not-a-verifier")


@pytest.mark.parametrize("floor", [0, DEFAULT_BROWSER_USER_PASSWORD_ITERATIONS, 500_000])
def test_legacy_verifier_preserves_password_acceptance_with_a_work_floor(floor: int) -> None:
    """Retain a real legacy digest while completing current verification work."""
    salt = bytes(range(16))
    digest = hashlib.pbkdf2_hmac("sha256", b"legacy-password", salt, 100_000).hex()
    verifier = f"pbkdf2_sha256$100000${salt.hex()}${digest}"
    assert verify_browser_user_password("legacy-password", verifier, minimum_iterations=floor)
    assert not verify_browser_user_password(
        "incorrect-password", verifier, minimum_iterations=floor
    )
