# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio identity admin test support

from __future__ import annotations

import hashlib

import json

from pathlib import Path

from typing import Any

import pytest

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app

from sc_neurocore.studio.platform.identity import (
    StudioIdentityAuthenticator,
    StudioIdentityLifecycleError,
    list_studio_browser_user_public_records,
    list_studio_identity_public_records,
    load_studio_identity_store,
    make_browser_user_password_verifier,
    rotate_studio_browser_user_password,
    update_studio_browser_user_record,
    update_studio_identity_record,
)

from sc_neurocore.studio.platform.settings import StudioRuntimeSettings


def _write_identity_file(path: Path, *, active: bool = True) -> str:
    token = "admin-token"
    path.write_text(
        json.dumps(
            {
                "browser_users": [
                    {
                        "active": True,
                        "expires_at_utc": None,
                        "password_pbkdf2_sha256": make_browser_user_password_verifier(
                            "operator-password"
                        ),
                        "principal_id": "human-operator",
                        "roles": ["studio.admin", "studio.viewer"],
                        "username": "operator",
                    }
                ],
                "schema_version": "sc-neurocore.studio.identity.v1",
                "service_accounts": [
                    {
                        "active": active,
                        "expires_at_utc": None,
                        "principal_id": "svc-admin",
                        "roles": ["studio.admin", "studio.viewer"],
                        "token_sha256": hashlib.sha256(token.encode("utf-8")).hexdigest(),
                    },
                    {
                        "active": True,
                        "principal_id": "svc-viewer",
                        "roles": ["studio.viewer"],
                        "token_sha256": hashlib.sha256(b"viewer-token").hexdigest(),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return token


def _write_single_service_admin_identity_file(path: Path) -> str:
    token = "sole-admin-token"
    path.write_text(
        json.dumps(
            {
                "browser_users": [],
                "schema_version": "sc-neurocore.studio.identity.v1",
                "service_accounts": [
                    {
                        "active": True,
                        "expires_at_utc": None,
                        "principal_id": "svc-sole-admin",
                        "roles": ["studio.admin"],
                        "token_sha256": hashlib.sha256(token.encode("utf-8")).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return token


def _write_single_browser_admin_identity_file(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "browser_users": [
                    {
                        "active": True,
                        "expires_at_utc": None,
                        "password_pbkdf2_sha256": make_browser_user_password_verifier(
                            "operator-password"
                        ),
                        "principal_id": "human-sole-admin",
                        "roles": ["studio.admin"],
                        "username": "operator",
                    }
                ],
                "schema_version": "sc-neurocore.studio.identity.v1",
                "service_accounts": [
                    {
                        "active": True,
                        "principal_id": "svc-viewer",
                        "roles": ["studio.viewer"],
                        "token_sha256": hashlib.sha256(b"viewer-token").hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _client(identity_path: Path, audit_path: Path) -> TestClient:
    app = create_app(
        StudioRuntimeSettings(
            allow_header_principal=False,
            audit_log_path=str(audit_path),
            enforce_route_policies=True,
            identity_file_path=str(identity_path),
        )
    )
    return TestClient(app, base_url="http://127.0.0.1")


def _admin_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "x-request-id": "identity-admin-test"}


def _audit_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


__all__ = [
    "annotations",
    "hashlib",
    "json",
    "Path",
    "Any",
    "pytest",
    "TestClient",
    "create_app",
    "StudioIdentityAuthenticator",
    "StudioIdentityLifecycleError",
    "list_studio_browser_user_public_records",
    "list_studio_identity_public_records",
    "load_studio_identity_store",
    "make_browser_user_password_verifier",
    "rotate_studio_browser_user_password",
    "update_studio_browser_user_record",
    "update_studio_identity_record",
    "StudioRuntimeSettings",
    "_write_identity_file",
    "_write_single_service_admin_identity_file",
    "_write_single_browser_admin_identity_file",
    "_client",
    "_admin_headers",
    "_audit_rows",
]
