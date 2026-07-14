# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Wu et al. 2021 integer quadratic integrate-and-fire model

"""Bit-true 2021 Integer Quadratic Integrate-and-Fire recurrence.

The maintained contract follows the public implementation accompanying Wu
et al. (2021), pinned to ``twetto/iq-neuron`` commit
``a8752eba49dba9ba43a64be74090b91a51044b2f``.  Despite the historical model
name, its hardware-friendly restoring force is piecewise linear rather than a
literal square:

``f(v) = a * (v_rest - v)`` below the branch point and
``f(v) = b * (v - v_threshold)`` otherwise.  ``a`` and ``b`` are Q0.3
numerators, so the source update uses an arithmetic right shift by three.
The source's optional noise is deliberately absent from this soma contract;
the enrolled tutorial configures it to zero and the paper's Eq. 2 receives
the already-computed synaptic current ``I(t)``.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import ClassVar, SupportsIndex, cast

import numpy as np
import numpy.typing as npt

_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1
_MAX_STEPS = (1 << 31) - 1
_BACKENDS = ("auto", "python", "rust", "julia", "go", "mojo")


def _int32(value: object, name: str) -> int:
    """Return one signed-int32 value without accepting lossy coercions."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a signed 32-bit integer")
    try:
        converted = operator.index(cast(SupportsIndex, value))
    except TypeError as exc:
        raise ValueError(f"{name} must be a signed 32-bit integer") from exc
    result = int(converted)
    if not _INT32_MIN <= result <= _INT32_MAX:
        raise ValueError(f"{name} must be in signed 32-bit range")
    return result


def _step_count(value: object) -> int:
    """Return a C-ABI-safe non-negative batch length."""
    if isinstance(value, bool):
        raise ValueError("n_steps must be a non-negative integer")
    try:
        converted = operator.index(cast(SupportsIndex, value))
    except TypeError as exc:
        raise ValueError("n_steps must be a non-negative integer") from exc
    result = int(converted)
    if not 0 <= result <= _MAX_STEPS:
        raise ValueError(f"n_steps must be in [0, {_MAX_STEPS}]")
    return result


def _trunc_div(numerator: int, denominator: int) -> int:
    """Divide integers with C++ truncation toward zero."""
    if denominator <= 0:  # pragma: no cover - guarded by the public contract
        raise ValueError("denominator must be positive")
    if numerator >= 0:
        return numerator // denominator
    return -((-numerator) // denominator)


@dataclass
class IntegerQIFNeuron:
    """Wu et al. (2021) piecewise-linear integer QIF soma.

    Parameters
    ----------
    v : int, default=128
        Live integer membrane state.
    v_rest : int, default=128
        Resting state and state restored by :meth:`reset`.
    v_threshold : int, default=200
        Upper piecewise-force reference, not the spike threshold.
    v_reset : int, default=128
        State committed after a candidate exceeds ``v_max``.
    a, b : int, default=1
        Non-negative Q0.3 numerator coefficients.  The recurrence applies
        ``(coefficient * difference) >> 3``.
    v_max, v_min : int, default=255, 0
        Strict upper event boundary and inclusive lower state clamp.

    Notes
    -----
    The branch point is ``trunc((b*v_threshold + a*v_rest)/(a+b))``.
    One step computes the source force from the pre-step state, applies the
    Q0.3 arithmetic shift and current, emits when the candidate is strictly
    greater than ``v_max``, then hard-resets to ``v_reset``.  Otherwise the
    candidate is clamped only at ``v_min``.

    References
    ----------
    Wu, W.-C., Yeh, C.-F., White, A. J., et al. (2021). Integer Quadratic
    Integrate-and-Fire (IQIF): A Neuron Model for Digital Neuromorphic Systems.
    https://doi.org/10.1109/AICAS51828.2021.9458572
    """

    v: int = 128
    v_rest: int = 128
    v_threshold: int = 200
    v_reset: int = 128
    a: int = 1
    b: int = 1
    v_max: int = 255
    v_min: int = 0

    dt: ClassVar[float] = 1.0
    SLOPE_FRACTION_BITS: ClassVar[int] = 3

    def __post_init__(self) -> None:
        """Normalise and validate the complete source contract."""
        values = self._validated_values()
        (
            self.v,
            self.v_rest,
            self.v_threshold,
            self.v_reset,
            self.a,
            self.b,
            self.v_max,
            self.v_min,
        ) = values

    @property
    def branch_point(self) -> int:
        """Return the source's C++-truncated piecewise boundary."""
        (
            _v,
            v_rest,
            v_threshold,
            _v_reset,
            a,
            b,
            _v_max,
            _v_min,
        ) = self._validated_values()
        return _trunc_div(b * v_threshold + a * v_rest, a + b)

    def _validated_values(self) -> tuple[int, int, int, int, int, int, int, int]:
        values = (
            _int32(self.v, "v"),
            _int32(self.v_rest, "v_rest"),
            _int32(self.v_threshold, "v_threshold"),
            _int32(self.v_reset, "v_reset"),
            _int32(self.a, "a"),
            _int32(self.b, "b"),
            _int32(self.v_max, "v_max"),
            _int32(self.v_min, "v_min"),
        )
        v, v_rest, v_threshold, v_reset, a, b, v_max, v_min = values
        if a < 0 or b < 0 or a + b == 0:
            raise ValueError("a and b must be non-negative and not both zero")
        if not v_min < v_rest < v_threshold < v_max:
            raise ValueError("require v_min < v_rest < v_threshold < v_max")
        if not v_min <= v_reset <= v_max:
            raise ValueError("v_reset must lie in [v_min, v_max]")
        if not v_min <= v <= v_max:
            raise ValueError("v must lie in [v_min, v_max]")
        return values

    @staticmethod
    def _advance(
        v: int,
        current: int,
        *,
        v_rest: int,
        v_threshold: int,
        v_reset: int,
        a: int,
        b: int,
        v_max: int,
        v_min: int,
        branch_point: int,
    ) -> tuple[int, int]:
        """Evaluate one source step without mutating an instance."""
        force = a * (v_rest - v) if v < branch_point else b * (v - v_threshold)
        candidate = v + (force >> IntegerQIFNeuron.SLOPE_FRACTION_BITS) + current
        if candidate > v_max:
            return v_reset, 1
        return max(v_min, candidate), 0

    def step(self, current: int) -> int:
        """Advance one integer tick and return a binary spike indicator.

        Invalid runtime mutations or non-integral input fail before state is
        changed.  All public scalar fields and input are signed int32; products,
        the branch numerator and the shifted candidate fit signed int64 over
        that domain.
        """
        current_i = _int32(current, "current")
        values = self._validated_values()
        v, v_rest, v_threshold, v_reset, a, b, v_max, v_min = values
        branch_point = _trunc_div(b * v_threshold + a * v_rest, a + b)
        self.v, spike = self._advance(
            v,
            current_i,
            v_rest=v_rest,
            v_threshold=v_threshold,
            v_reset=v_reset,
            a=a,
            b=b,
            v_max=v_max,
            v_min=v_min,
            branch_point=branch_point,
        )
        return spike

    def simulate(
        self,
        n_steps: int,
        current: int = 10,
        backend: str = "auto",
    ) -> tuple[npt.NDArray[np.int64], int]:
        """Return an exact post-step trace through one maintained backend.

        A successful run atomically commits only the returned final voltage.
        Explicit unavailable backends raise :class:`RuntimeError`; auto falls
        through availability probes in the measured backend order, while any
        error from the selected available backend propagates.
        """
        steps = _step_count(n_steps)
        current_i = _int32(current, "current")
        values = self._validated_values()
        v, v_rest, v_threshold, v_reset, a, b, v_max, v_min = values
        if backend not in _BACKENDS:
            raise ValueError(f"backend must be auto/python/rust/julia/go/mojo, got {backend!r}")

        from sc_neurocore.accel import iqif as backends

        selected = backends.auto_backend() if backend == "auto" else backend
        if selected == "python":
            trace, spikes, final_v = self._simulate_python(steps, current_i)
        else:
            if not backends.backend_available(selected):
                raise RuntimeError(f"{selected.title()} IQIF backend is unavailable.")
            runner = {
                "rust": backends.simulate_rust,
                "julia": backends.simulate_julia,
                "go": backends.simulate_go,
                "mojo": backends.simulate_mojo,
            }[selected]
            trace, spikes, final_v = runner(
                v,
                v_rest,
                v_threshold,
                v_reset,
                a,
                b,
                v_max,
                v_min,
                steps,
                current_i,
            )

        trace, spikes, final_v = backends.normalise_result(
            trace,
            spikes,
            final_v,
            n_steps=steps,
            v_min=v_min,
            v_max=v_max,
        )
        self.v = final_v
        return trace, spikes

    def _simulate_python(
        self,
        n_steps: int,
        current: int,
    ) -> tuple[npt.NDArray[np.int64], int, int]:
        """Evaluate a mutation-free local batch for atomic public commit."""
        v, v_rest, v_threshold, v_reset, a, b, v_max, v_min = self._validated_values()
        branch_point = _trunc_div(b * v_threshold + a * v_rest, a + b)
        trace = np.empty(n_steps, dtype=np.int64)
        spikes = 0
        for index in range(n_steps):
            v, spike = self._advance(
                v,
                current,
                v_rest=v_rest,
                v_threshold=v_threshold,
                v_reset=v_reset,
                a=a,
                b=b,
                v_max=v_max,
                v_min=v_min,
                branch_point=branch_point,
            )
            trace[index] = v
            spikes += spike
        return trace, spikes, v

    def reset(self) -> None:
        """Restore ``v_rest`` without changing any parameter."""
        values = self._validated_values()
        self.v = values[1]
