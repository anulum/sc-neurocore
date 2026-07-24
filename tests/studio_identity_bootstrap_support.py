# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_studio_identity_bootstrap.py

from __future__ import annotations


import hashlib


import json


import os


import sys


from pathlib import Path


from typing import Any, Literal


import pytest


import sc_neurocore.studio.platform.bootstrap as bootstrap


from sc_neurocore.cli import main


from sc_neurocore.studio.platform.bootstrap import (
    DEFAULT_STUDIO_ADMIN_PRINCIPAL_ID,
    DEFAULT_STUDIO_ADMIN_ROLES,
    MIN_BOOTSTRAP_TOKEN_BYTES,
    bootstrap_studio_admin_identity,
)


from sc_neurocore.studio.platform.identity import (
    StudioIdentityAuthenticator,
    add_studio_browser_user_record,
    load_studio_identity_store,
)


_STUDIO_PREFLIGHT_ENV_KEYS = (
    "SC_NEUROCORE_STUDIO_ALLOW_HEADER_PRINCIPAL",
    "SC_NEUROCORE_STUDIO_AUDIT_LOG_PATH",
    "SC_NEUROCORE_STUDIO_DEPLOYMENT_PROFILE",
    "SC_NEUROCORE_STUDIO_ENFORCE_ROUTE_POLICIES",
    "SC_NEUROCORE_STUDIO_IDENTITY_FILE",
    "SC_NEUROCORE_STUDIO_JOB_ROOT",
)


def _fixed_token(_: int) -> str:
    return "generated-admin-token"


def _configure_release_preflight_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    identity_path: Path,
) -> None:
    for key in tuple(os.environ):
        if key.startswith("SC_NEUROCORE_STUDIO_") and key not in _STUDIO_PREFLIGHT_ENV_KEYS:
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SC_NEUROCORE_STUDIO_ALLOW_HEADER_PRINCIPAL", "false")
    monkeypatch.setenv(
        "SC_NEUROCORE_STUDIO_AUDIT_LOG_PATH",
        str(tmp_path / "audit" / "studio.jsonl"),
    )
    monkeypatch.setenv("SC_NEUROCORE_STUDIO_DEPLOYMENT_PROFILE", "production")
    monkeypatch.setenv("SC_NEUROCORE_STUDIO_ENFORCE_ROUTE_POLICIES", "true")
    monkeypatch.setenv("SC_NEUROCORE_STUDIO_IDENTITY_FILE", str(identity_path))
    monkeypatch.setenv("SC_NEUROCORE_STUDIO_JOB_ROOT", str(tmp_path / "jobs"))


class _StringStdin:
    """Small stdin stand-in for CLI password input tests."""

    def __init__(self, text: str) -> None:
        self._text = text

    def readline(self) -> str:
        """Return the configured input once."""

        return self._text


__all__ = [
    "hashlib",
    "json",
    "os",
    "sys",
    "Path",
    "Any",
    "Literal",
    "pytest",
    "bootstrap",
    "main",
    "DEFAULT_STUDIO_ADMIN_PRINCIPAL_ID",
    "DEFAULT_STUDIO_ADMIN_ROLES",
    "MIN_BOOTSTRAP_TOKEN_BYTES",
    "bootstrap_studio_admin_identity",
    "StudioIdentityAuthenticator",
    "add_studio_browser_user_record",
    "load_studio_identity_store",
    "_STUDIO_PREFLIGHT_ENV_KEYS",
    "_fixed_token",
    "_configure_release_preflight_env",
    "_StringStdin",
]
