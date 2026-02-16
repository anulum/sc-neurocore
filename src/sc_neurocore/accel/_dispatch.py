"""
Shared import guards for optional accelerators (Rust / Numba / CuPy).

Usage::

    from sc_neurocore.accel._dispatch import USE_RUST, _engine, njit_or_python, HAS_NUMBA
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Rust engine (PyO3 extension built by maturin)
# ---------------------------------------------------------------------------
try:
    import sc_neurocore_engine as _engine  # type: ignore[import-untyped]  # pragma: no cover

    USE_RUST = True  # pragma: no cover
except ImportError:
    _engine = None  # type: ignore[assignment]
    USE_RUST = False

# ---------------------------------------------------------------------------
# Numba JIT
# ---------------------------------------------------------------------------
try:
    from numba import njit as _njit  # type: ignore[import-untyped]

    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    HAS_NUMBA = False  # pragma: no cover

    def _njit(*args, **kwargs):  # type: ignore[misc]  # pragma: no cover
        """No-op decorator when Numba is absent."""
        if args and callable(args[0]):
            return args[0]
        return lambda f: f


def njit_or_python(**kwargs):
    """Return ``@njit(**kwargs)`` when Numba is available, else identity."""
    return _njit(**kwargs) if HAS_NUMBA else lambda f: f
