# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBitstreamGenerationContracts from former test_fault_injection_module.py

"""Focused suite: TestBitstreamGenerationContracts from former test_fault_injection_module.py."""

from __future__ import annotations

from tests.fault_injection_module_support import *  # noqa: F403

class TestBitstreamGenerationContracts:
    def test_generated_stream_is_binary_and_length_preserved(self):
        bench = ResilienceBenchmark(seed=3)
        stream = bench._generate_bitstream(16, 0.25)
        assert stream.shape == (16,)
        assert set(stream.tolist()).issubset({0, 1})

    @pytest.mark.parametrize(
        ("length", "probability", "match"),
        [
            (0, 0.5, "length"),
            (4, -0.1, "probability"),
            (4, 1.1, "probability"),
            (4, float("nan"), "probability"),
        ],
    )
    def test_rejects_invalid_generation_inputs(self, length, probability, match):
        bench = ResilienceBenchmark(seed=3)
        with pytest.raises(ValueError, match=match):
            bench._generate_bitstream(length, probability)
