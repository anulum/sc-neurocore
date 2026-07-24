# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDataContractValidation from former test_validation_contracts.py

"""Focused suite: TestDataContractValidation from former test_validation_contracts.py."""

from __future__ import annotations

from tests.test_bioware.validation_contracts_support import *  # noqa: F403


class TestDataContractValidation:
    def test_mea_layout_type_is_explicit(self) -> None:
        with pytest.raises(TypeError, match="layout must be a MEALayout"):
            MEAConfig(layout=cast(Any, "60ch"))
        with pytest.raises(TypeError, match="layout must be a MEALayout"):
            MEAConfig.from_layout(cast(Any, "60ch"))

    @pytest.mark.parametrize(
        "waveform",
        [
            cast(Any, [1.0]),
            np.array([], dtype=float),
            np.ones((1, 1)),
            np.array(["x"]),
            np.array([float("inf")]),
        ],
    )
    def test_detected_spike_rejects_invalid_waveforms(self, waveform: Any) -> None:
        with pytest.raises((TypeError, ValueError)):
            DetectedSpike(0, 0.0, -1.0, waveform=waveform)

    def test_aer_packet_rejects_out_of_range_fields(self) -> None:
        with pytest.raises(ValueError, match="timestamp must fit"):
            AEREvent(0, 0x10000)
        with pytest.raises(TypeError, match="valid must be a bool"):
            AEREvent(0, 0, valid=cast(Any, 1))
        with pytest.raises(ValueError, match="weight must fit"):
            AEREvent(0, 0, weight=0x10000)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("num_spikes", 1),
            ("num_aer_events", 1),
            ("num_bitstreams", 1),
            ("num_opto_pulses", 1),
        ],
    )
    def test_frame_result_rejects_payload_count_mismatch(
        self,
        field: str,
        value: int,
    ) -> None:
        payload: dict[str, Any] = {
            "round": 1,
            "num_spikes": 0,
            "num_aer_events": 0,
            "num_bitstreams": 0,
            "num_opto_pulses": 0,
            "latency_us": 1.0,
            "health": {},
            "spikes": [],
            "aer_events": [],
            "bitstreams": {},
            "opto_pulses": [],
        }
        payload[field] = value
        with pytest.raises(ValueError, match="must equal"):
            BioHybridFrameResult(**payload)
