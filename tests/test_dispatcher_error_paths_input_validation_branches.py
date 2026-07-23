# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestInputValidationBranches from former test_dispatcher_error_paths.py

"""Focused suite: TestInputValidationBranches from former test_dispatcher_error_paths.py."""

from __future__ import annotations

from tests.dispatcher_error_paths_support import *  # noqa: F403

class TestInputValidationBranches:
    """The shape-mismatch ValueError paths in every dispatcher."""

    def test_go_wilson_stim_mismatch(self) -> None:
        # Wilson-Cowan has only `ext_input`; there is no stim1/stim2 in its
        # signature. This test is the degenerate case: empty array works.
        # Requires the compiled Go cdylib — skip in environments where
        # `libwilson_cowan.so` was not built (e.g. the default CI image
        # without a Go toolchain step).
        if go_wilson._lib is None:
            pytest.skip(
                "libwilson_cowan.so not built — install Go and run "
                "`go build -buildmode=c-shared -o libwilson_cowan.so wilson_cowan.go` "
                "in src/sc_neurocore/accel/go/wilson_cowan/"
            )
        out = go_wilson.simulate_wilson_cowan(
            0.1,
            0.05,
            10.0,
            6.0,
            10.0,
            1.0,
            1.0,
            2.0,
            1.2,
            4.0,
            0.1,
            np.zeros(0),
        )
        e_trace = out["e"]
        assert isinstance(e_trace, np.ndarray)
        assert e_trace.shape == (0,)
