# SPDX-License-Identifier: AGPL-3.0-or-later
"""Deprecation utilities for SC-NeuroCore API evolution."""

from __future__ import annotations

import functools
import warnings


def deprecated(since: str, removal: str, alternative: str | None = None):
    """Mark a function or class as deprecated.

    Example::

        @deprecated(since="3.11", removal="4.0", alternative="new_func")
        def old_func():
            ...

    Parameters
    ----------
    since : str
        Version where deprecation was introduced.
    removal : str
        Version where the function will be removed.
    alternative : str, optional
        Name of the replacement function/class.
    """

    def decorator(obj):
        alt_msg = f" Use {alternative} instead." if alternative else ""
        msg = (
            f"{obj.__qualname__} is deprecated since v{since} "
            f"and will be removed in v{removal}.{alt_msg}"
        )

        if isinstance(obj, type):
            original_init = obj.__init__

            @functools.wraps(original_init)
            def new_init(self, *args, **kwargs):
                warnings.warn(msg, DeprecationWarning, stacklevel=2)
                original_init(self, *args, **kwargs)

            obj.__init__ = new_init
            return obj

        @functools.wraps(obj)
        def wrapper(*args, **kwargs):
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return obj(*args, **kwargs)

        return wrapper

    return decorator
