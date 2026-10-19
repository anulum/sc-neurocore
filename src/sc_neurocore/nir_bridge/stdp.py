# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — trace-based STDP for the FPGA network compiler: rule and integer golden

"""Trace-based pair STDP on a network connection, in fixed point.

The rule is the one :meth:`sc_neurocore.network.projection.Projection.update_plasticity`
applies, without its self-projection symmetrisation. Each tick, after every neuron
has stepped:

1. every trace decays and takes this tick's spikes,
   ``x <- x * exp(-1/tau) + s`` (pre traces per source neuron, post traces per
   target neuron, both counting the tick just computed);
2. every synapse ``i -> j`` of the connection subtracts ``a_minus * post[j]`` when
   ``i`` spiked and adds ``a_plus * pre[i]`` when ``j`` spiked;
3. the weight is clipped to ``[w_min, w_max]``.

The next tick's currents use the updated weights. The synapses are the non-zero
entries of the connection's initial weight matrix; an entry that is zero at the
start is no synapse and never learns, as a CSR projection has no entry for it.

In hardware every quantity is a signed word of the network format (``fraction``
bits): the decay factor ``D = round(exp(-1/tau) * 2**fraction)``, the
amplitudes ``A± = round(a± * 2**fraction)``, the bounds, the traces and the
weights. A trace becomes ``sat((x * D) >>> fraction + (spike ? 2**fraction : 0))``
and a weight ``clip(w - ((A- * post) >>> fraction) + ((A+ * pre) >>> fraction))``,
each term added only on its spike, with ``>>>`` the arithmetic shift (floor).
:func:`stdp_tick` is that integer arithmetic; the generated RTL of the direct and
the folded interconnect equals it, which the tests check bit for bit.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["QuantisedSTDP", "STDPRule", "stdp_tick"]


@dataclass(frozen=True, slots=True)
class STDPRule:
    """Pair STDP with exponential traces on one connection.

    Parameters
    ----------
    a_plus : float
        Potentiation amplitude per post spike, times the pre trace; at least 0.
    a_minus : float
        Depression amplitude per pre spike, times the post trace; at least 0.
    tau : float
        Trace time constant in ticks; positive.
    w_min, w_max : float
        Weight bounds, ``w_min <= w_max``.

    Raises
    ------
    ValueError
        For a non-finite value, a negative amplitude, a non-positive ``tau`` or
        bounds in the wrong order.
    """

    a_plus: float = 0.01
    a_minus: float = 0.012
    tau: float = 20.0
    w_min: float = 0.0
    w_max: float = 1.0

    def __post_init__(self) -> None:
        """Validate the rule's domain."""
        for name in ("a_plus", "a_minus", "tau", "w_min", "w_max"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"STDP {name} must be a real number, got {value!r}")
            if not math.isfinite(value):
                raise ValueError(f"STDP {name} must be finite, got {value!r}")
        if self.a_plus < 0.0 or self.a_minus < 0.0:
            raise ValueError("STDP amplitudes must be non-negative")
        if self.tau <= 0.0:
            raise ValueError(f"STDP tau must be positive, got {self.tau!r}")
        if self.w_min > self.w_max:
            raise ValueError(f"STDP needs w_min <= w_max, got {self.w_min!r} > {self.w_max!r}")

    def quantised(self, data_width: int, fraction: int) -> QuantisedSTDP:
        """Return the rule's words at a signed ``data_width``/``fraction`` format.

        Parameters
        ----------
        data_width, fraction : int
            The network's fixed-point format.

        Returns
        -------
        QuantisedSTDP
            The decay factor, amplitudes and bounds as words.

        Raises
        ------
        ValueError
            When a value is outside the format, or a non-zero amplitude or the
            decay factor rounds to zero.
        """
        scale = 1 << fraction
        low, high = -(1 << (data_width - 1)), (1 << (data_width - 1)) - 1
        words = {
            "decay": round(math.exp(-1.0 / self.tau) * scale),
            "a_plus": round(self.a_plus * scale),
            "a_minus": round(self.a_minus * scale),
            "w_min": round(self.w_min * scale),
            "w_max": round(self.w_max * scale),
            "one": scale,
        }
        for name, word in words.items():
            if not low <= word <= high:
                raise ValueError(f"STDP {name} does not fit Q{data_width - fraction}.{fraction}")
        for name in ("a_plus", "a_minus"):
            if getattr(self, name) > 0.0 and words[name] == 0:
                raise ValueError(
                    f"STDP {name}={getattr(self, name)!r} rounds to zero at fraction {fraction}"
                )
        if words["decay"] == 0:
            raise ValueError(
                f"STDP decay exp(-1/{self.tau!r}) rounds to zero at fraction {fraction}"
            )
        return QuantisedSTDP(
            decay=words["decay"],
            a_plus=words["a_plus"],
            a_minus=words["a_minus"],
            w_min=words["w_min"],
            w_max=words["w_max"],
            one=scale,
            fraction=fraction,
            trace_max=high,
        )


@dataclass(frozen=True, slots=True)
class QuantisedSTDP:
    """An :class:`STDPRule` as the words the hardware holds.

    Parameters
    ----------
    decay : int
        ``round(exp(-1/tau) * 2**fraction)``.
    a_plus, a_minus : int
        The amplitudes as words.
    w_min, w_max : int
        The weight bounds as words.
    one : int
        ``2**fraction``, a spike's contribution to a trace.
    fraction : int
        Fractional bits.
    trace_max : int
        The largest word; traces saturate there.
    """

    decay: int
    a_plus: int
    a_minus: int
    w_min: int
    w_max: int
    one: int
    fraction: int
    trace_max: int


def stdp_tick(
    weights: Sequence[Sequence[int]],
    mask: Sequence[Sequence[bool]],
    pre_trace: Sequence[int],
    post_trace: Sequence[int],
    pre_spikes: Sequence[int],
    post_spikes: Sequence[int],
    rule: QuantisedSTDP,
) -> tuple[list[list[int]], list[int], list[int]]:
    """Apply one tick of the integer rule, as the generated RTL does.

    Parameters
    ----------
    weights : sequence of sequence of int
        Weight words, ``weights[j][i]`` for target ``j`` and source ``i``.
    mask : sequence of sequence of bool
        Which entries are synapses (non-zero at the start).
    pre_trace, post_trace : sequence of int
        Trace words before this tick.
    pre_spikes, post_spikes : sequence of int
        This tick's spikes (0 or 1) of the source and target neurons.
    rule : QuantisedSTDP
        The rule's words.

    Returns
    -------
    tuple
        ``(weights, pre_trace, post_trace)`` after the tick.
    """
    f = rule.fraction

    def decay(trace: int, spike: int) -> int:
        return min(((trace * rule.decay) >> f) + (rule.one if spike else 0), rule.trace_max)

    pre = [decay(x, s) for x, s in zip(pre_trace, pre_spikes, strict=True)]
    post = [decay(x, s) for x, s in zip(post_trace, post_spikes, strict=True)]
    updated = []
    for j, row in enumerate(weights):
        new_row = []
        for i, w in enumerate(row):
            if mask[j][i]:
                if pre_spikes[i]:
                    w -= (rule.a_minus * post[j]) >> f
                if post_spikes[j]:
                    w += (rule.a_plus * pre[i]) >> f
                w = min(max(w, rule.w_min), rule.w_max)
            new_row.append(w)
        updated.append(new_row)
    return updated, pre, post
