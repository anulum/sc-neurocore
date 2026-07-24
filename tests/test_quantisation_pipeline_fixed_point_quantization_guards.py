# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFixedPointQuantizationGuards from former test_quantisation_pipeline.py

"""Focused suite: TestFixedPointQuantizationGuards from former test_quantisation_pipeline.py."""

from __future__ import annotations

from tests.quantisation_pipeline_support import *  # noqa: F403


class TestFixedPointQuantizationGuards:
    """Fail-closed branches in the fixed-point quantisation backend."""

    def test_coerce_q_format_rejects_non_format_type(self):
        from sc_neurocore.compiler.fixed_point_quantization import _coerce_q_format

        with pytest.raises(TypeError, match="Expected QFormat or Q-format string"):
            _coerce_q_format(123)  # type: ignore[arg-type]

    def test_quantize_weights_rejects_block_floating_string(self):
        from sc_neurocore.compiler.fixed_point_quantization import quantize_weights as qw

        with pytest.raises(ValueError, match="quantize_block_floating"):
            qw(np.array([0.5]), fmt="BFP16E3X32")

    def test_dequantize_weights_rejects_block_floating_string(self):
        from sc_neurocore.compiler.fixed_point_quantization import (
            dequantize_weights as dqw,
        )

        with pytest.raises(ValueError, match="dequantize_block_floating"):
            dqw(np.array([1]), fmt="BFP16E3X32")

    def test_mixed_precision_scale_must_stay_finite(self):
        from sc_neurocore.compiler.fixed_point_quantization import quantize_weights as qw
        from sc_neurocore.compiler.quantizer import QFormatMixed

        # A near-denormal weight makes the per-tensor scale overflow to inf.
        with pytest.raises(ValueError, match="per-tensor scale must be finite"):
            qw(np.array([1e-308]), fmt=QFormatMixed(scale_per_tensor=True))

    def test_quantization_error_rejects_mixed_format(self):
        from sc_neurocore.compiler.fixed_point_quantization import quantization_error
        from sc_neurocore.compiler.quantizer import QFormatMixed

        with pytest.raises(TypeError, match="not QFormatMixed"):
            quantization_error(np.array([0.5, 0.3]), fmt=QFormatMixed())
