# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFaultInjectorInjectContracts from former test_fault_injection_module.py

"""Focused suite: TestFaultInjectorInjectContracts from former test_fault_injection_module.py."""

from __future__ import annotations

from tests.fault_injection_module_support import *  # noqa: F403

class TestFaultInjectorInjectContracts:
    def test_rejects_invalid_inputs(self):
        import numpy as np

        injector = FaultInjector(seed=1)
        with pytest.raises(ValueError, match="numpy.ndarray"):
            injector.inject([0, 1], FaultModel.BIT_FLIP, 0.1)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="1-D"):
            injector.inject(np.zeros((2, 2), dtype=np.uint8), FaultModel.BIT_FLIP, 0.1)
        with pytest.raises(ValueError, match="non-empty"):
            injector.inject(np.zeros((0,), dtype=np.uint8), FaultModel.BIT_FLIP, 0.1)
        with pytest.raises(ValueError, match="FaultModel"):
            injector.inject(np.zeros((4,), dtype=np.uint8), "bit_flip", 0.1)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="ber"):
            injector.inject(np.zeros((4,), dtype=np.uint8), FaultModel.BIT_FLIP, 1.1)

    def test_discrete_models_reject_non_binary_streams(self):
        import numpy as np

        injector = FaultInjector(seed=1)
        bad = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        with pytest.raises(ValueError, match="binary"):
            injector.inject(bad, FaultModel.BIT_FLIP, 0.1)

    def test_gaussian_noise_rejects_non_finite_or_out_of_range_inputs(self):
        import numpy as np

        injector = FaultInjector(seed=1)
        with pytest.raises(ValueError, match="finite"):
            injector.inject(np.array([0.0, np.nan, 1.0]), FaultModel.GAUSSIAN_NOISE, 0.1)
        with pytest.raises(ValueError, match="within"):
            injector.inject(np.array([0.0, 1.2, 1.0]), FaultModel.GAUSSIAN_NOISE, 0.1)

    @pytest.mark.parametrize(
        "model",
        [
            FaultModel.BIT_FLIP,
            FaultModel.STUCK_AT_0,
            FaultModel.STUCK_AT_1,
            FaultModel.GAUSSIAN_NOISE,
            FaultModel.DROPOUT,
        ],
    )
    def test_zero_ber_is_deterministic_noop(self, model):
        import numpy as np

        injector = FaultInjector(seed=1)
        bitstream = np.array([0, 1, 0, 1], dtype=np.uint8)
        out, affected = injector.inject(bitstream, model, 0.0)
        assert affected == 0
        assert np.array_equal(out, bitstream)
