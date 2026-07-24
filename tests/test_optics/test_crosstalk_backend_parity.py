# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBackendParity from former test_crosstalk.py

"""Focused suite: TestBackendParity from former test_crosstalk.py."""

from __future__ import annotations

from crosstalk_support import *  # noqa: F403


class TestBackendParity:
    """The Rust and Python code paths solve the same coupled-mode equations
    — their outputs must agree to within floating-point tolerance. Any drift
    is a regression in the Rust FFI layer.
    """

    def test_analyze_bank_rust_matches_python(self):
        import sc_neurocore.optics.photonic_emitter as mod

        if not mod._HAS_RUST_PH:
            pytest.skip("Rust photonic bindings not built")
        model_rust = CrosstalkModel()
        r_rust = model_rust.analyze_bank(waveguides=6, gap_nm=250.0, coupling_length_um=12.0)
        assert r_rust["backend"] == "rust"

        # Force Python path via monkey-patched flag.
        import sc_neurocore.optics.photonic_emitter as mod

        orig = mod._HAS_RUST_PH
        try:
            mod._HAS_RUST_PH = False
            r_py = CrosstalkModel().analyze_bank(
                waveguides=6, gap_nm=250.0, coupling_length_um=12.0
            )
        finally:
            mod._HAS_RUST_PH = orig
        assert r_py["backend"] == "python"

        for key in (
            "adjacent_coupling_ratio",
            "adjacent_isolation_db",
            "next_nearest_coupling_ratio",
            "next_nearest_isolation_db",
            "worst_isolation_db",
            "mean_coupling_ratio",
        ):
            assert r_rust[key] == pytest.approx(r_py[key], rel=1e-9, abs=1e-12), (
                f"{key}: rust={r_rust[key]} python={r_py[key]}"
            )

    def test_analyze_pairs_rust_matches_python(self):
        import sc_neurocore.optics.photonic_emitter as mod

        if not mod._HAS_RUST_PH:
            pytest.skip("Rust photonic bindings not built")
        pair_indices = [(0, 1), (1, 2), (0, 2), (3, 4)]
        gaps = [200.0, 300.0, 500.0, 150.0]
        lengths = [10.0, 15.0, 20.0, 5.0]

        r_rust = CrosstalkModel().analyze_pairs(pair_indices, gaps, lengths)
        assert r_rust["backend"] == "rust"

        import sc_neurocore.optics.photonic_emitter as mod

        orig = mod._HAS_RUST_PH
        try:
            mod._HAS_RUST_PH = False
            r_py = CrosstalkModel().analyze_pairs(pair_indices, gaps, lengths)
        finally:
            mod._HAS_RUST_PH = orig

        assert np.allclose(r_rust["isolation_db"], r_py["isolation_db"], rtol=1e-9, atol=1e-12)
        assert np.allclose(r_rust["coupling_ratio"], r_py["coupling_ratio"], rtol=1e-9, atol=1e-12)
