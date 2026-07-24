# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio identity store test support

from __future__ import annotations

import hashlib

import json

from datetime import datetime, timezone

from pathlib import Path

import pytest

from sc_neurocore.studio.platform.identity import (
    StudioIdentityAuthenticator,
    add_studio_browser_user_record,
    load_studio_identity_store,
    make_browser_user_password_verifier,
    rotate_studio_browser_user_password,
    update_studio_browser_user_record,
    update_studio_identity_record,
    verify_browser_user_password,
)


def _write_identity_file(path: Path, token: str, *, expires_at_utc: str | None = None) -> None:
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    payload: dict[str, object] = {
        "schema_version": "sc-neurocore.studio.identity.v1",
        "service_accounts": [
            {
                "principal_id": "svc-admin",
                "roles": ["studio.admin", "studio.viewer"],
                "token_sha256": token_hash,
            }
        ],
    }
    if expires_at_utc is not None:
        accounts = payload["service_accounts"]
        assert isinstance(accounts, list)
        account = accounts[0]
        assert isinstance(account, dict)
        account["expires_at_utc"] = expires_at_utc
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_payload(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


__all__ = [
    "annotations",
    "hashlib",
    "json",
    "datetime",
    "timezone",
    "Path",
    "pytest",
    "StudioIdentityAuthenticator",
    "add_studio_browser_user_record",
    "load_studio_identity_store",
    "make_browser_user_password_verifier",
    "rotate_studio_browser_user_password",
    "update_studio_browser_user_record",
    "update_studio_identity_record",
    "verify_browser_user_password",
    "_write_identity_file",
    "_write_payload",
]
