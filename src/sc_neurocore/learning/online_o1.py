# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bounded O(1) online learning contracts

"""Fixed-point online learning contracts with bounded per-synapse state.

The implementation is intentionally local: one synapse stores only the current
weight, pre trace, post trace, and eligibility trace. Sequence samples are
streamed through ``step`` and are never retained, making the memory proof
independent of sequence length.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Literal

ONLINE_O1_ANNOTATION_SCHEMA_VERSION = "sc-neurocore.online-o1.annotation.v1"
ONLINE_O1_MEMORY_PROOF_SCHEMA_VERSION = "sc-neurocore.online-o1.memory-proof.v1"

_STATE_FIELDS = ("weight", "pre_trace", "post_trace", "eligibility")
_MAX_WEIGHT_BITS = 31
_MAX_TRACE_BITS = 30
_MAX_REWARD_BITS = 30
_MAX_SHIFT = 30


@dataclass(frozen=True, slots=True)
class OnlineO1Config:
    """Hardware-bounded configuration for local reward-modulated STDP."""

    weight_bits: int = 16
    trace_bits: int = 12
    reward_bits: int = 8
    learning_shift: int = 4
    trace_decay_shift: int = 4
    rule_family: Literal["reward_modulated_stdp"] = "reward_modulated_stdp"

    def __post_init__(self) -> None:
        """Validate and normalize fixed-point fields after dataclass initialization."""
        object.__setattr__(
            self,
            "weight_bits",
            _require_integral_range(
                name="weight_bits", value=self.weight_bits, lower=1, upper=_MAX_WEIGHT_BITS
            ),
        )
        object.__setattr__(
            self,
            "trace_bits",
            _require_integral_range(
                name="trace_bits", value=self.trace_bits, lower=2, upper=_MAX_TRACE_BITS
            ),
        )
        object.__setattr__(
            self,
            "reward_bits",
            _require_integral_range(
                name="reward_bits", value=self.reward_bits, lower=1, upper=_MAX_REWARD_BITS
            ),
        )
        object.__setattr__(
            self,
            "learning_shift",
            _require_integral_range(
                name="learning_shift", value=self.learning_shift, lower=0, upper=_MAX_SHIFT
            ),
        )
        object.__setattr__(
            self,
            "trace_decay_shift",
            _require_integral_range(
                name="trace_decay_shift",
                value=self.trace_decay_shift,
                lower=0,
                upper=_MAX_SHIFT,
            ),
        )
        if self.rule_family != "reward_modulated_stdp":
            raise ValueError("rule_family must be 'reward_modulated_stdp'")

    @property
    def max_weight(self) -> int:
        """Maximum unsigned fixed-point weight."""
        return (1 << self.weight_bits) - 1

    @property
    def max_trace(self) -> int:
        """Maximum unsigned trace value."""
        return (1 << self.trace_bits) - 1

    @property
    def min_eligibility(self) -> int:
        """Minimum signed eligibility value."""
        return -(1 << (self.trace_bits - 1))

    @property
    def max_eligibility(self) -> int:
        """Maximum signed eligibility value."""
        return (1 << (self.trace_bits - 1)) - 1

    @property
    def min_reward(self) -> int:
        """Minimum signed reward input."""
        return -(1 << (self.reward_bits - 1))

    @property
    def max_reward(self) -> int:
        """Maximum signed reward input."""
        return (1 << (self.reward_bits - 1)) - 1

    @property
    def per_synapse_state_bits(self) -> int:
        """Stored bits per synapse: weight plus three bounded traces."""
        return self.weight_bits + 3 * self.trace_bits

    def to_scnir_annotation(self, *, rule_id: str) -> dict[str, Any]:
        """Return deterministic SC-NIR metadata for online-learning synapses."""
        if not rule_id:
            raise ValueError("rule_id must be non-empty")
        return {
            "schema_version": ONLINE_O1_ANNOTATION_SCHEMA_VERSION,
            "rule_id": rule_id,
            "rule_family": self.rule_family,
            "state_fields": list(_STATE_FIELDS),
            "per_synapse_state_bits": self.per_synapse_state_bits,
            "weight_bits": self.weight_bits,
            "trace_bits": self.trace_bits,
            "reward_bits": self.reward_bits,
            "learning_shift": self.learning_shift,
            "trace_decay_shift": self.trace_decay_shift,
            "saturation_policy": "signed_eligibility_unsigned_weight",
            "hidden_history_fields": [],
            "sequence_length_independent": True,
        }


@dataclass(frozen=True, slots=True)
class OnlineO1Snapshot:
    """Immutable synapse state snapshot after one online update."""

    weight: int
    pre_trace: int
    post_trace: int
    eligibility: int


@dataclass(slots=True)
class OnlineO1Synapse:
    """One fixed-point reward-modulated STDP synapse with O(1) state."""

    config: OnlineO1Config
    initial_weight: int = 0
    weight: int = 0
    pre_trace: int = 0
    post_trace: int = 0
    eligibility: int = 0

    def __post_init__(self) -> None:
        """Initialize bounded mutable synapse state from a validated configuration."""
        if not isinstance(self.config, OnlineO1Config):
            raise TypeError("config must be an OnlineO1Config")
        self.initial_weight = _require_non_negative_integral(
            name="initial_weight", value=self.initial_weight
        )
        self.weight = _saturate(self.initial_weight, 0, self.config.max_weight)
        self.pre_trace = 0
        self.post_trace = 0
        self.eligibility = 0

    @property
    def state_fields(self) -> tuple[str, ...]:
        """Names of state fields retained between timesteps."""
        return _STATE_FIELDS

    @property
    def state_bit_count(self) -> int:
        """Stored state bits for one synapse."""
        return self.config.per_synapse_state_bits

    def snapshot(self) -> OnlineO1Snapshot:
        """Return the current bounded state."""
        return OnlineO1Snapshot(
            weight=self.weight,
            pre_trace=self.pre_trace,
            post_trace=self.post_trace,
            eligibility=self.eligibility,
        )

    def step(self, *, pre_spike: bool, post_spike: bool, reward: int) -> OnlineO1Snapshot:
        """Advance one streamed timestep and return the bounded state.

        The rule uses pre-before-post eligibility:

        ``eligibility += post_spike * pre_trace - pre_spike * post_trace``

        The reward-gated weight update is an arithmetic right shift of the
        product, then saturated into the unsigned weight range.
        """
        reward = _saturate(
            _require_integral(name="reward", value=reward),
            self.config.min_reward,
            self.config.max_reward,
        )
        previous_pre_trace = self.pre_trace
        previous_post_trace = self.post_trace

        self.pre_trace = _decay_unsigned(
            self.pre_trace, self.config.trace_decay_shift, self.config.max_trace
        )
        self.post_trace = _decay_unsigned(
            self.post_trace, self.config.trace_decay_shift, self.config.max_trace
        )
        if pre_spike:
            self.pre_trace = self.config.max_trace
        if post_spike:
            self.post_trace = self.config.max_trace

        decayed_eligibility = _decay_signed(self.eligibility, self.config.trace_decay_shift)
        potentiation = 0
        if post_spike:
            potentiation = self.config.max_trace if pre_spike else previous_pre_trace
        depression = previous_post_trace if pre_spike else 0
        eligibility_delta = potentiation - depression
        self.eligibility = _saturate(
            decayed_eligibility + eligibility_delta,
            self.config.min_eligibility,
            self.config.max_eligibility,
        )

        weight_delta = (reward * self.eligibility) >> self.config.learning_shift
        self.weight = _saturate(self.weight + weight_delta, 0, self.config.max_weight)
        return self.snapshot()


def build_online_o1_memory_proof(
    *, n_synapses: int, config: OnlineO1Config, sequence_length: int | None = None
) -> dict[str, Any]:
    """Return a sequence-length independent memory proof for the rule."""
    n_synapses = _require_integral(name="n_synapses", value=n_synapses)
    if n_synapses < 0:
        raise ValueError("n_synapses must be >= 0")
    if sequence_length is not None:
        sequence_length = _require_integral(name="sequence_length", value=sequence_length)
        if sequence_length < 0:
            raise ValueError("sequence_length must be >= 0")
    total_state_bits = n_synapses * config.per_synapse_state_bits
    return {
        "schema_version": ONLINE_O1_MEMORY_PROOF_SCHEMA_VERSION,
        "n_synapses": n_synapses,
        "state_fields": list(_STATE_FIELDS),
        "per_synapse_state_bits": config.per_synapse_state_bits,
        "total_state_bits": total_state_bits,
        "sequence_length_independent": True,
        "hidden_history_fields": [],
    }


def _require_integral(*, name: str, value: object) -> int:
    """Return ``value`` as ``int`` after rejecting bool and non-integral input."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer and must not be bool")
    return int(value)


def _require_non_negative_integral(*, name: str, value: object) -> int:
    """Return a non-negative integer for unsigned fixed-point domains."""
    integral = _require_integral(name=name, value=value)
    if integral < 0:
        raise ValueError(f"{name} must be >= 0")
    return integral


def _require_integral_range(*, name: str, value: object, lower: int, upper: int) -> int:
    """Return an integer inside the requested inclusive fixed-point domain."""
    integral = _require_integral(name=name, value=value)
    if integral < lower or integral > upper:
        raise ValueError(f"{name} must be in {lower}..={upper}")
    return integral


def _decay_unsigned(value: int, shift: int, max_value: int) -> int:
    """Apply bounded unsigned trace decay with saturating totality."""
    if shift == 0:
        return _saturate(value, 0, max_value)
    return _saturate(value - (value >> shift), 0, max_value)


def _decay_signed(value: int, shift: int) -> int:
    """Apply arithmetic-magnitude decay to a signed eligibility trace."""
    if shift == 0:
        return value
    if value >= 0:
        return value - (value >> shift)
    magnitude = -value
    return -(magnitude - (magnitude >> shift))


def _saturate(value: int, lower: int, upper: int) -> int:
    """Clamp an integer into inclusive fixed-point bounds."""
    return min(upper, max(lower, int(value)))
