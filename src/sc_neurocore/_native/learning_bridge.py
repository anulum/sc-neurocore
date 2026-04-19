# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Python ctypes bridge for autonomous_learning Rust C-FFI

"""C-FFI bridge to Rust ``libautonomous_learning.so`` for plasticity hot paths.

Provides:
- RustPlasticityRule: STDP(0), BCM(1), RewardSTDP(2), ELIGENT(3)
- RustEligentLearner: Online eligibility-based learning
"""

from __future__ import annotations

import ctypes as _ct
import pathlib as _pl

_LIB_PATH = _pl.Path(__file__).parent / "libautonomous_learning.so"
_HAS_LEARNING = False
_lib = None

if _LIB_PATH.exists():
    try:
        _lib = _ct.CDLL(str(_LIB_PATH))

        # create_rule(rule_type: u32, weight: f32, param_a: f32, param_b: f32) -> *mut RuleHandle
        _lib.create_rule.argtypes = [_ct.c_uint32, _ct.c_float, _ct.c_float, _ct.c_float]
        _lib.create_rule.restype = _ct.c_void_p

        # step_rule(ptr, pre_spike: bool, post_spike: bool, dt: f32, reward: f32)
        _lib.step_rule.argtypes = [_ct.c_void_p, _ct.c_bool, _ct.c_bool, _ct.c_float, _ct.c_float]
        _lib.step_rule.restype = None

        # get_rule_weight(ptr) -> f32
        _lib.get_rule_weight.argtypes = [_ct.c_void_p]
        _lib.get_rule_weight.restype = _ct.c_float

        # reset_rule(ptr)
        _lib.reset_rule.argtypes = [_ct.c_void_p]
        _lib.reset_rule.restype = None

        # destroy_rule(ptr)
        _lib.destroy_rule.argtypes = [_ct.c_void_p]
        _lib.destroy_rule.restype = None

        # create_learner(threshold: f32, target_rate: f32, weight: f32) -> *mut EligentRule
        _lib.create_learner.argtypes = [_ct.c_float, _ct.c_float, _ct.c_float]
        _lib.create_learner.restype = _ct.c_void_p

        # step_learner(ptr, fired: bool, pre_spike: bool, global_reward: f32)
        _lib.step_learner.argtypes = [_ct.c_void_p, _ct.c_bool, _ct.c_bool, _ct.c_float]
        _lib.step_learner.restype = None

        # destroy_learner(ptr)
        _lib.destroy_learner.argtypes = [_ct.c_void_p]
        _lib.destroy_learner.restype = None

        _HAS_LEARNING = True
    except OSError:
        pass

# Rule type constants
RULE_STDP = 0
RULE_BCM = 1
RULE_REWARD_STDP = 2
RULE_ELIGENT = 3


def is_available() -> bool:
    """Return True if the Rust learning engine is loaded."""
    return _HAS_LEARNING


class RustPlasticityRule:
    """RAII wrapper around a Rust plasticity rule handle.

    rule_type: RULE_STDP(0), RULE_BCM(1), RULE_REWARD_STDP(2), RULE_ELIGENT(3)
    """

    __slots__ = ("_ptr", "_rule_type")

    def __init__(
        self,
        rule_type: int = RULE_STDP,
        weight: float = 0.5,
        param_a: float = 0.01,
        param_b: float = 0.012,
    ) -> None:
        if not _HAS_LEARNING:
            raise RuntimeError("libautonomous_learning.so not available")
        self._rule_type = rule_type
        self._ptr = _lib.create_rule(
            _ct.c_uint32(rule_type),
            _ct.c_float(weight),
            _ct.c_float(param_a),
            _ct.c_float(param_b),
        )

    def step(
        self, pre_spike: bool, post_spike: bool, dt: float = 0.001, reward: float = 0.0
    ) -> None:
        _lib.step_rule(self._ptr, pre_spike, post_spike, _ct.c_float(dt), _ct.c_float(reward))

    @property
    def weight(self) -> float:
        return float(_lib.get_rule_weight(self._ptr))

    def reset(self) -> None:
        _lib.reset_rule(self._ptr)

    def __del__(self) -> None:
        if self._ptr and _HAS_LEARNING:
            _lib.destroy_rule(self._ptr)
            self._ptr = None


class RustEligentLearner:
    """RAII wrapper around a Rust ELIGENT learner handle."""

    __slots__ = ("_ptr",)

    def __init__(
        self, threshold: float = 1.0, target_rate: float = 0.1, weight: float = 0.5
    ) -> None:
        if not _HAS_LEARNING:
            raise RuntimeError("libautonomous_learning.so not available")
        self._ptr = _lib.create_learner(
            _ct.c_float(threshold),
            _ct.c_float(target_rate),
            _ct.c_float(weight),
        )

    def step(self, fired: bool, pre_spike: bool, global_reward: float = 0.0) -> None:
        _lib.step_learner(self._ptr, fired, pre_spike, _ct.c_float(global_reward))

    def __del__(self) -> None:
        if self._ptr and _HAS_LEARNING:
            _lib.destroy_learner(self._ptr)
            self._ptr = None
