# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTransferHelpers from former test_gpu_backend.py

"""Focused suite: TestTransferHelpers from former test_gpu_backend.py."""

from __future__ import annotations

from tests.gpu_backend_support import *  # noqa: F403


class TestTransferHelpers:
    def test_to_device_and_back(self):
        a = np.array([1, 2, 3], dtype=np.float32)
        dev = to_device(a)
        host = to_host(dev)
        np.testing.assert_array_equal(host, a)

    def test_to_host_passthrough(self):
        a = np.array([4, 5, 6])
        assert to_host(a) is a or np.array_equal(to_host(a), a)
