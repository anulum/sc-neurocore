# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Thermal SC internal rotation contracts

"""Internal activity-preserving rotation fallback contract."""

from .security_thermal_sc_encoding_support import *


def test_rotation_offset_falls_back_to_zero_without_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the candidate generator yields no offsets the activity-preserving
    search finds nothing and falls back to the safe identity rotation."""
    config = ThermalSCEncodingConfig(bitstream_length=8)
    base = _distribute_ones(4, config.bitstream_length)
    monkeypatch.setattr(
        "sc_neurocore.security.thermal_sc_encoding._candidate_offsets",
        lambda *args, **kwargs: (),
    )
    assert _activity_preserving_rotation_offset(base, config, 0) == 0
