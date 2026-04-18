# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for the @deprecated decorator

"""Tests for the @deprecated decorator."""

from __future__ import annotations

import warnings

from sc_neurocore.utils.deprecation import deprecated


def test_function_warns():
    @deprecated(since="3.10", removal="4.0")
    def old_func():
        return 42

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = old_func()

    assert result == 42
    assert len(w) == 1
    assert issubclass(w[0].category, DeprecationWarning)
    assert "v3.10" in str(w[0].message)
    assert "v4.0" in str(w[0].message)


def test_function_alternative_message():
    @deprecated(since="3.10", removal="4.0", alternative="new_func")
    def old_func():
        return 1

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        old_func()

    assert "Use new_func instead" in str(w[0].message)


def test_class_warns():
    @deprecated(since="3.10", removal="4.0", alternative="NewClass")
    class OldClass:
        def __init__(self, x):
            self.x = x

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        obj = OldClass(7)

    assert obj.x == 7
    assert len(w) == 1
    assert issubclass(w[0].category, DeprecationWarning)
    assert "NewClass" in str(w[0].message)


def test_preserves_function_metadata():
    @deprecated(since="3.10", removal="4.0")
    def some_function():
        """Docstring."""
        return 0

    assert some_function.__name__ == "some_function"
    assert some_function.__doc__ == "Docstring."


def test_no_alternative_no_mention():
    @deprecated(since="1.0", removal="2.0")
    def f():
        pass

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        f()

    assert "Use " not in str(w[0].message)


def test_top_level_import():
    from sc_neurocore import deprecated as d

    assert d is deprecated
