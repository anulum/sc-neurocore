# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestProbeReturnsTrueOnSecondCall from former test_hierarchical_partitioner_runtime_cache.py

"""Focused suite: TestProbeReturnsTrueOnSecondCall from former test_hierarchical_partitioner_runtime_cache.py."""

from __future__ import annotations


import pytest
from tests.julia_requirement import require_julia


class TestProbeReturnsTrueOnSecondCall:
    """Backend probes short-circuit on their second call.

    Each `_ensure_*_loaded` probe must short-circuit return True
    when called again — covers the `if X is not None: return True`
    branch. We force-load (first call) then assert (second call) so
    the test does not depend on the order of unrelated tests.
    """

    def test_julia_probe_second_call_short_circuits(self) -> None:
        """A loaded Julia probe returns True on its second call."""
        from sc_neurocore.chiplet import hierarchical_backend_runtime as hp_mod

        require_julia()
        assert hp_mod._ensure_julia_kl_refine_loaded(), (
            "kl_refine.jl did not load with a healthy bridge"
        )
        # Now julia is loaded — second call must short-circuit.
        assert hp_mod._ensure_julia_kl_refine_loaded() is True

    def test_go_probe_second_call_short_circuits(self) -> None:
        """A loaded Go probe returns True on its second call."""
        from sc_neurocore.chiplet import hierarchical_backend_runtime as hp_mod

        if not hp_mod._ensure_go_kl_refine_loaded():
            pytest.skip("Go libpartition.so not built")
        assert hp_mod._ensure_go_kl_refine_loaded() is True

    def test_mojo_probe_second_call_short_circuits(self) -> None:
        """A loaded Mojo probe returns True on its second call."""
        from sc_neurocore.chiplet import hierarchical_backend_runtime as hp_mod

        if not hp_mod._ensure_mojo_kl_refine_loaded():
            pytest.skip("Mojo libpartition.so not built")
        assert hp_mod._ensure_mojo_kl_refine_loaded() is True
