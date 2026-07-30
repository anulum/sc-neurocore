# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Receipt metadata checks for source MAT(1)."""

from __future__ import annotations
import json
from pathlib import Path


def test_source_receipt_binds_equations_and_digest() -> None:
    data = json.loads(
        Path(
            "src/sc_neurocore/neurons/reference_trace_data/non_resetting_lif_mat1.json"
        ).read_text()
    )
    assert data["source"]["doi"] == "10.3389/neuro.10.009.2009"
    assert data["oracle"]["event_indices"] == [3945]
    assert (
        data["oracle"]["trace_sha256"]
        == "2ac13e42322a3ac6b4059f29190f0936409c9d4bf28f1837e4bee97add2069c6"
    )
