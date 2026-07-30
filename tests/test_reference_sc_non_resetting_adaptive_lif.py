# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Receipt metadata checks for the retained project recurrence."""

from __future__ import annotations
import json
from pathlib import Path


def test_project_receipt_binds_compatibility_digest() -> None:
    data = json.loads(
        Path(
            "src/sc_neurocore/neurons/reference_trace_data/sc_non_resetting_adaptive_lif_project.json"
        ).read_text()
    )
    assert data["source"]["doi"] is None
    assert data["oracle"]["event_count"] == 5
    assert (
        data["oracle"]["trace_sha256"]
        == "7dd9f76fd1d819bc462460112cfb5906b137935db466bfd60e206f1b4303ae25"
    )
