# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeCodecModuleExports from former test_spike_codec.py

"""Focused suite: TestSpikeCodecModuleExports from former test_spike_codec.py."""

from __future__ import annotations

from tests.spike_codec_support import *  # noqa: F403


class TestSpikeCodecModuleExports:
    def test_lazy_predictive_exports_are_available(self):
        assert spike_codec_module.PredictiveSpikeCodec is not None
        assert spike_codec_module.PredictiveCompressionResult is not None
        assert "SpikeCodec" in spike_codec_module.__all__

    def test_unknown_lazy_export_raises_attribute_error(self):
        with pytest.raises(AttributeError):
            spike_codec_module.__getattr__("not_an_export")
