# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio analysis codegen

"""Focused suite: TestCodegen from former test_studio_analysis.py."""

from __future__ import annotations

from tests.studio_analysis_support import *  # noqa: F403

class TestCodegen:
    def test_model_script(self):
        script = generate_model_script("COBALIFNeuron", {"c_m": 200.0}, 100, 10, 0.1)
        assert "COBALIFNeuron" in script
        assert "c_m=200.0" in script
        assert "step(current=" in script

    def test_model_script_runs(self):
        script = generate_model_script("COBALIFNeuron")
        assert "import numpy" in script

