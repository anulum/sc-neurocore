# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — v3 engine thread-pool configuration

"""Rayon thread-pool configuration checks for the v3 engine."""

from __future__ import annotations

import subprocess
import sys


class TestSetNumThreads:
    """Tests for Rayon thread-pool configuration."""

    def test_zero_preserves_uninitialised_default_pool(self) -> None:
        """Zero leaves the pool unset so a later explicit size can initialise it."""
        script = """
import sc_neurocore_engine as v3

assert v3.set_num_threads(0) is None
assert v3.set_num_threads(1) is None
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
