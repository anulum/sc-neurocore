# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestExport from former test_wave4.py

"""Focused suite: TestExport from former test_wave4.py."""

from __future__ import annotations

from sc_neurocore.debug.hil_client import (
    export_csv,
    export_json,
    SpikeEvent,
)


class TestExport:
    """Exporting spike events."""

    def test_csv(self) -> None:
        """The CSV export has a header and the layer."""
        events = [SpikeEvent(timestamp=1, layer_id="L0", precision=0.95)]
        csv_str = export_csv(events)
        assert "timestamp" in csv_str
        assert "L0" in csv_str

    def test_json(self) -> None:
        """The JSON export lists the event with its layer."""
        events = [SpikeEvent(timestamp=1, layer_id="L0")]
        j = export_json(events)
        data = __import__("json").loads(j)
        assert len(data) == 1
        assert data[0]["layer_id"] == "L0"
