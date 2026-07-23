# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRunIntegratedStep from former test_scpn_integrated.py

"""Focused suite: TestRunIntegratedStep from former test_scpn_integrated.py."""

from __future__ import annotations

from tests.scpn_integrated_support import *  # noqa: F403

class TestRunIntegratedStep:
    def test_returns_dict(self):
        stack = create_full_stack()
        result = run_integrated_step(stack, dt=0.001)
        assert isinstance(result, dict)

    def test_all_layers_in_result(self):
        stack = create_full_stack()
        result = run_integrated_step(stack, dt=0.001)
        for key in stack:
            assert key in result, f"{key} missing from integrated step result"

    def test_finite_outputs(self):
        """All layer outputs should be finite after one step."""
        stack = create_full_stack()
        result = run_integrated_step(stack, dt=0.001)
        for key, output in result.items():
            if isinstance(output, (dict,)):
                for k, v in output.items():
                    if isinstance(v, (float, int, np.floating)):
                        assert np.isfinite(v), f"{key}.{k} = {v} not finite"
                    elif isinstance(v, np.ndarray):
                        assert np.all(np.isfinite(v)), f"{key}.{k} contains non-finite"

    def test_multiple_steps_stable(self):
        """10 steps should not diverge."""
        stack = create_full_stack()
        for _ in range(10):
            result = run_integrated_step(stack, dt=0.001)
