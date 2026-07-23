# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRefineBackendValidation from former test_hierarchical_partitioner_core.py

"""Focused suite: TestRefineBackendValidation from former test_hierarchical_partitioner_core.py."""

from __future__ import annotations

from hierarchical_partitioner_core_support import *  # noqa: F403

class TestRefineBackendValidation:
    """The constructor must reject unknown backend names cleanly,
    and missing-tool errors at dispatch time must be informative."""

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="refine_backend must be"):
            HierarchicalPartitioner(refine_backend="cuda")

    def test_known_backends_construct(self) -> None:
        for b in ("auto", "rust", "julia", "go", "mojo", "python"):
            hp = HierarchicalPartitioner(refine_backend=b)
            assert hp.refine_backend == b
