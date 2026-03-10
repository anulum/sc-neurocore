# SPDX-License-Identifier: AGPL-3.0-or-later
"""Verify all __all__ exports are importable and no regressions occur."""

import sc_neurocore


def test_all_symbols_importable():
    for name in sc_neurocore.__all__:
        assert hasattr(sc_neurocore, name), f"Missing export: {name}"


def test_version_string():
    assert sc_neurocore.__version__ == "3.10.0"


def test_all_count():
    assert len(sc_neurocore.__all__) >= 18, "Public API shrank unexpectedly"
