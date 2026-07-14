# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — McCulloch & Pitts 1943 all-or-none threshold neuron

"""Source-faithful McCulloch--Pitts excitatory-count neuron.

McCulloch and Pitts' 1943 formal neuron has three relevant single-cell
properties: activity is all-or-none, a fixed number of excitatory afferents
must be active within one synaptic delay, and any active inhibitory afferent
vetoes excitation.  The maintained class exposes that contract directly.  A
general real-valued weighted-threshold unit is a later abstraction and is not
silently substituted here.
"""

from __future__ import annotations

from collections.abc import Iterable
import math
import operator
from dataclasses import dataclass
from typing import SupportsIndex, cast

import numpy as np
import numpy.typing as npt

_INT32_MAX = (1 << 31) - 1
_BACKENDS = ("auto", "python", "rust", "julia", "go", "mojo")


def _bounded_integer(value: object, name: str, *, minimum: int) -> int:
    """Return one exact integer in the public count domain.

    Integer-valued Python and NumPy floats remain accepted because the generic
    network runner carries scalar input through a floating-point accumulator.
    Fractional, Boolean, non-finite and out-of-range values fail closed.
    """
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer in [{minimum}, {_INT32_MAX}]")
    try:
        result = int(operator.index(cast(SupportsIndex, value)))
    except TypeError:
        if not isinstance(value, (float, np.floating)):
            raise ValueError(f"{name} must be an integer in [{minimum}, {_INT32_MAX}]") from None
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"{name} must be an integer in [{minimum}, {_INT32_MAX}]")
        result = int(numeric)
    if not minimum <= result <= _INT32_MAX:
        raise ValueError(f"{name} must be an integer in [{minimum}, {_INT32_MAX}]")
    return result


def _inhibitory_flag(value: object, name: str = "inhibitory_active") -> bool:
    """Return one exact Boolean inhibitory-afferent indicator."""
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a Boolean")
    return bool(value)


def encode_hardware_input(excitatory_count: object, inhibitory_active: object = False) -> int:
    """Encode the two-input source contract on the signed Q32.0 RTL port.

    Non-negative values carry the active excitatory-afferent count.  ``-1`` is
    the sole inhibitory-veto sentinel.  Because ``theta`` is strictly positive,
    the schema condition ``I >= theta`` is equivalent to the public two-input
    rule for every valid input.
    """
    count = _bounded_integer(excitatory_count, "excitatory_count", minimum=0)
    return -1 if _inhibitory_flag(inhibitory_active) else count


def _normalise_batch(
    excitatory_counts: Iterable[object] | npt.NDArray[np.generic],
    inhibitory_flags: Iterable[object] | npt.NDArray[np.generic] | None,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.uint8]]:
    """Validate complete batch inputs before any backend receives them."""
    if isinstance(excitatory_counts, np.ndarray) and excitatory_counts.ndim != 1:
        raise ValueError("excitatory_counts must be one-dimensional")
    try:
        raw_counts = (
            excitatory_counts.tolist()
            if isinstance(excitatory_counts, np.ndarray)
            else list(excitatory_counts)
        )
    except TypeError as exc:
        raise ValueError("excitatory_counts must be a one-dimensional iterable") from exc
    if isinstance(raw_counts, list) and any(isinstance(item, list) for item in raw_counts):
        raise ValueError("excitatory_counts must be one-dimensional")
    if len(raw_counts) > _INT32_MAX:
        raise ValueError(f"batch length must be in [0, {_INT32_MAX}]")

    counts = np.empty(len(raw_counts), dtype=np.int64)
    for index, value in enumerate(raw_counts):
        counts[index] = _bounded_integer(value, f"excitatory_counts[{index}]", minimum=0)

    if inhibitory_flags is None:
        flags = np.zeros(len(raw_counts), dtype=np.uint8)
    else:
        if isinstance(inhibitory_flags, np.ndarray) and inhibitory_flags.ndim != 1:
            raise ValueError("inhibitory_flags must be one-dimensional")
        try:
            raw_flags = (
                inhibitory_flags.tolist()
                if isinstance(inhibitory_flags, np.ndarray)
                else list(inhibitory_flags)
            )
        except TypeError as exc:
            raise ValueError("inhibitory_flags must be a one-dimensional iterable") from exc
        if isinstance(raw_flags, list) and any(isinstance(item, list) for item in raw_flags):
            raise ValueError("inhibitory_flags must be one-dimensional")
        if len(raw_flags) != len(raw_counts):
            raise ValueError("inhibitory_flags must match excitatory_counts length")
        flags = np.empty(len(raw_flags), dtype=np.uint8)
        for index, value in enumerate(raw_flags):
            flags[index] = int(_inhibitory_flag(value, f"inhibitory_flags[{index}]"))
    return np.ascontiguousarray(counts), np.ascontiguousarray(flags)


@dataclass
class McCullochPittsNeuron:
    """McCulloch and Pitts' 1943 all-or-none logical neuron.

    Parameters
    ----------
    theta : int, default=1
        Positive number of simultaneously active excitatory afferents required
        to excite the neuron when no inhibitory afferent is active.

    Notes
    -----
    :meth:`step` maps afferent activity at one network instant to activity one
    synaptic delay later.  The delay belongs to the network scheduler; the
    formal neuron has no evolving membrane state.  Any active inhibitory
    afferent is an absolute veto, independent of the excitatory count.

    References
    ----------
    McCulloch, W. S., & Pitts, W. (1943). A logical calculus of the ideas
    immanent in nervous activity. *Bulletin of Mathematical Biophysics*, 5,
    115--133. https://doi.org/10.1007/BF02478259
    """

    theta: int = 1

    def __post_init__(self) -> None:
        """Normalise the fixed excitatory-count threshold."""
        self.theta = _bounded_integer(self.theta, "theta", minimum=1)

    def _validated_theta(self) -> int:
        """Revalidate a publicly mutable threshold before evaluation."""
        return _bounded_integer(self.theta, "theta", minimum=1)

    def step(self, excitatory_count: object, inhibitory_active: object = False) -> int:
        """Return the source-faithful all-or-none output for one delay.

        Parameters
        ----------
        excitatory_count : int
            Number of active excitatory afferents in the preceding network
            instant.  The accepted domain is signed-ABI-safe ``[0, 2**31-1]``.
        inhibitory_active : bool, default=False
            Whether at least one inhibitory afferent is active.  ``True``
            always returns zero.

        Returns
        -------
        int
            ``1`` exactly when inhibition is absent and the count is at least
            :attr:`theta`; otherwise ``0``.
        """
        count = _bounded_integer(excitatory_count, "excitatory_count", minimum=0)
        inhibited = _inhibitory_flag(inhibitory_active)
        theta = self._validated_theta()
        return int(not inhibited and count >= theta)

    def simulate(
        self,
        excitatory_counts: Iterable[object] | npt.NDArray[np.generic],
        inhibitory_flags: Iterable[object] | npt.NDArray[np.generic] | None = None,
        backend: str = "auto",
    ) -> tuple[npt.NDArray[np.uint8], int]:
        """Evaluate a varying-input batch through one maintained backend.

        All inputs are validated before dispatch.  Explicit unavailable
        backends raise :class:`RuntimeError`; ``auto`` selects the first real
        available native lane and otherwise uses Python.  Returned values must
        be a contiguous binary ``uint8`` vector with an exact event count.
        """
        if backend not in _BACKENDS:
            raise ValueError(f"backend must be auto/python/rust/julia/go/mojo, got {backend!r}")
        theta = self._validated_theta()
        counts, flags = _normalise_batch(excitatory_counts, inhibitory_flags)

        from sc_neurocore.accel import mcculloch_pitts as backends

        selected = backends.auto_backend() if backend == "auto" else backend
        if selected == "python":
            events = np.asarray((flags == 0) & (counts >= theta), dtype=np.uint8)
            result: object = (events, int(np.sum(events, dtype=np.int64)))
        else:
            if not backends.backend_available(selected):
                raise RuntimeError(f"{selected.title()} McCulloch-Pitts backend is unavailable.")
            runner = {
                "rust": backends.evaluate_rust,
                "julia": backends.evaluate_julia,
                "go": backends.evaluate_go,
                "mojo": backends.evaluate_mojo,
            }[selected]
            result = runner(theta, counts, flags)
        return backends.normalise_result(result, expected_length=len(counts))

    def reset(self) -> None:
        """Validate the fixed parameter; the formal neuron has no live state."""
        self._validated_theta()


__all__ = ["McCullochPittsNeuron", "encode_hardware_input"]
