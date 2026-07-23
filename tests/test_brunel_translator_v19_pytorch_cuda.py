# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestV19PytorchCuda from former test_brunel_translator.py

"""Focused suite: TestV19PytorchCuda from former test_brunel_translator.py."""

from __future__ import annotations

from tests.brunel_translator_support import *  # noqa: F403

class TestV19PytorchCuda:
    """V19: CUDA tensor computation."""

    def test_cuda_params(self):
        bp = BrunelParams()
        params = translate_v19_pytorch_cuda(bp)
        assert params["n_total"] == 1000
        assert params["v_threshold"] == 20.0
