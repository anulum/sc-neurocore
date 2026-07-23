# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio identity password verifier unit tests

"""Behaviour tests for Studio browser-user password verifiers."""

from __future__ import annotations

import pytest

from sc_neurocore.studio.platform.identity_passwords import (
    DEFAULT_BROWSER_USER_PASSWORD_ITERATIONS,
    make_browser_user_password_verifier,
    verify_browser_user_password,
    _parse_password_verifier,
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


def test_parse_rejects_malformed_verifiers() -> None:
    """Malformed or under-iterated verifiers fail closed."""
    assert _parse_password_verifier("sha256$1$bad") is None
    assert _parse_password_verifier("pbkdf2_sha256$1$aa$bb") is None
    assert not verify_browser_user_password("x", "not-a-verifier")
