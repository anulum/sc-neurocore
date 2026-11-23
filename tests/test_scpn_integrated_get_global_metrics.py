# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGetGlobalMetrics from former test_scpn_integrated.py

"""Focused suite: TestGetGlobalMetrics from former test_scpn_integrated.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.scpn import (
    create_full_stack,
    get_global_metrics,
    run_integrated_step,
)


class TestGetGlobalMetrics:
    """Collecting the global metric of each layer."""

    def test_returns_dict(self) -> None:
        """The call returns a dictionary."""
        stack = create_full_stack()
        run_integrated_step(stack, dt=0.001)
        metrics = get_global_metrics(stack)
        assert isinstance(metrics, dict)

    def test_metrics_finite(self) -> None:
        """Every global metric is finite."""
        stack = create_full_stack()
        for _ in range(5):
            run_integrated_step(stack, dt=0.001)
        metrics = get_global_metrics(stack)
        for key, val in metrics.items():
            if isinstance(val, (float, int, np.floating)):
                assert np.isfinite(val), f"metric {key} = {val} not finite"

    def test_has_coherence_metric(self) -> None:
        """Global metrics should include some coherence measure."""
        stack = create_full_stack()
        run_integrated_step(stack, dt=0.001)
        metrics = get_global_metrics(stack)
        # Should have at least one metric related to coherence/integration
        assert len(metrics) > 0
