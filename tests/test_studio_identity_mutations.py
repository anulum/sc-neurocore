# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio identity store mutations

"""Identity mutation helpers and non-target preservation contracts."""

from __future__ import annotations

from tests.studio_identity_support import *  # noqa: F403

def test_identity_mutations_report_missing_records(tmp_path: Path) -> None:
    identity_path = tmp_path / "studio-identities.json"
    _write_identity_file(identity_path, "admin-token")

    with pytest.raises(KeyError, match="missing-service"):
        update_studio_identity_record(
            identity_path,
            active=True,
            expires_at_utc=None,
            principal_id="missing-service",
            roles=["studio.admin"],
        )
    with pytest.raises(KeyError, match="missing-browser"):
        update_studio_browser_user_record(
            identity_path,
            active=True,
            expires_at_utc=None,
            roles=["studio.admin"],
            username="missing-browser",
        )
    with pytest.raises(KeyError, match="missing-browser"):
        rotate_studio_browser_user_password(
            identity_path,
            password="rotated-password",
            username="missing-browser",
        )
