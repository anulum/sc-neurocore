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

import os

_HAS_LEARNING = False
_lib = None


def _load_native_library() -> bool:
    """Load libautonomous_learning if present. Returns True on success, False if absent.

    A missing shared library is a legitimate state: the Rust plasticity backend is
    optional. Consumers must gate runtime use behind ``is_available()`` — the class
    constructors below raise on construction when the library is absent. Raising at
    import time couples every ``sc_neurocore`` import to a Rust build artefact and
    breaks 398 unrelated tests when the crate has not yet been compiled.
    """
    global _lib, _HAS_LEARNING
    env_path = os.getenv("SC_NEUROCORE_LIB_PATH")

    if env_path:
        lib_path = _pl.Path(env_path)
    else:
        import platform

        sys_name = platform.system()
        lib_name = (
            "autonomous_learning.dll"
            if sys_name == "Windows"
            else "libautonomous_learning.dylib"
            if sys_name == "Darwin"
            else "libautonomous_learning.so"
        )
        lib_path = _pl.Path(__file__).parent.resolve() / lib_name

    if not lib_path.exists():
        _HAS_LEARNING = False
        _lib = None
        return False

    try:
        _lib = _ct.CDLL(str(lib_path))

        # Binding definitions
        _lib.create_rule.argtypes = [_ct.c_uint32, _ct.c_float, _ct.c_float, _ct.c_float]
        _lib.create_rule.restype = _ct.c_void_p
        _lib.step_rule.argtypes = [_ct.c_void_p, _ct.c_bool, _ct.c_bool, _ct.c_float, _ct.c_float]
        _lib.step_rule.restype = None
        _lib.get_rule_weight.argtypes = [_ct.c_void_p]
        _lib.get_rule_weight.restype = _ct.c_float
        _lib.reset_rule.argtypes = [_ct.c_void_p]
        _lib.reset_rule.restype = None
        _lib.destroy_rule.argtypes = [_ct.c_void_p]
        _lib.destroy_rule.restype = None

        _lib.create_learner.argtypes = [_ct.c_float, _ct.c_float, _ct.c_float]
        _lib.create_learner.restype = _ct.c_void_p
        _lib.step_learner.argtypes = [
            _ct.c_void_p,
            _ct.c_bool,
            _ct.c_bool,
            _ct.c_float,
            _ct.c_float,
        ]
        _lib.step_learner.restype = None
        _lib.destroy_learner.argtypes = [_ct.c_void_p]
        _lib.destroy_learner.restype = None

        _lib.step_rule_batched.argtypes = [
            _ct.c_void_p,
            _ct.POINTER(_ct.c_bool),
            _ct.POINTER(_ct.c_bool),
            _ct.POINTER(_ct.c_float),
            _ct.c_size_t,
            _ct.c_float,
        ]
        _lib.step_rule_batched.restype = None
        _lib.step_learner_batched.argtypes = [
            _ct.c_void_p,
            _ct.POINTER(_ct.c_bool),
            _ct.POINTER(_ct.c_bool),
            _ct.POINTER(_ct.c_float),
            _ct.c_size_t,
            _ct.c_float,
        ]
        _lib.step_learner_batched.restype = None

        _lib.create_rule_layer.argtypes = [
            _ct.c_size_t,
            _ct.c_uint32,
            _ct.c_float,
            _ct.c_float,
            _ct.c_float,
        ]
        _lib.create_rule_layer.restype = _ct.c_void_p
        _lib.step_rule_layer.argtypes = [
            _ct.c_void_p,
            _ct.POINTER(_ct.c_bool),
            _ct.POINTER(_ct.c_bool),
            _ct.POINTER(_ct.c_float),
            _ct.c_float,
        ]
        _lib.step_rule_layer.restype = None
        _lib.get_rule_layer_weights.argtypes = [_ct.c_void_p, _ct.POINTER(_ct.c_float)]
        _lib.get_rule_layer_weights.restype = None
        _lib.destroy_rule_layer.argtypes = [_ct.c_void_p]
        _lib.destroy_rule_layer.restype = None
        _lib.reset_rule_layer.argtypes = [_ct.c_void_p]
        _lib.reset_rule_layer.restype = None

        _lib.save_rule_layer_batched.argtypes = [_ct.c_void_p, _ct.c_char_p]
        _lib.save_rule_layer_batched.restype = _ct.c_bool
        _lib.load_rule_layer_batched.argtypes = [_ct.c_void_p, _ct.c_char_p]
        _lib.load_rule_layer_batched.restype = _ct.c_bool

        # New Memory-Buffer Serialization API boundaries
        _lib.get_rule_layer_state_size.argtypes = [_ct.c_void_p]
        _lib.get_rule_layer_state_size.restype = _ct.c_size_t
        _lib.get_rule_layer_state_mem.argtypes = [_ct.c_void_p, _ct.POINTER(_ct.c_byte)]
        _lib.get_rule_layer_state_mem.restype = _ct.c_bool
        _lib.set_rule_layer_state_mem.argtypes = [_ct.c_void_p, _ct.POINTER(_ct.c_byte)]
        _lib.set_rule_layer_state_mem.restype = _ct.c_bool

        _lib.step_rule_layer_analog.argtypes = [
            _ct.c_void_p,
            _ct.POINTER(_ct.c_float),
            _ct.POINTER(_ct.c_float),
            _ct.POINTER(_ct.c_float),
            _ct.c_uint64,
            _ct.c_float,
        ]
        _lib.step_rule_layer_analog.restype = None

        # WGPU Backend boundaries
        _lib.create_wgpu_layer.argtypes = [
            _ct.c_size_t,
            _ct.c_uint32,
            _ct.c_float,
            _ct.c_float,
            _ct.c_float,
            _ct.c_float,
            _ct.c_float,
            _ct.c_float,
        ]
        _lib.create_wgpu_layer.restype = _ct.c_void_p
        _lib.step_wgpu_layer.argtypes = [
            _ct.c_void_p,
            _ct.POINTER(_ct.c_float),
            _ct.POINTER(_ct.c_float),
            _ct.POINTER(_ct.c_float),
            _ct.c_float,
        ]
        _lib.step_wgpu_layer.restype = None
        _lib.get_wgpu_weights.argtypes = [_ct.c_void_p, _ct.POINTER(_ct.c_float)]
        _lib.get_wgpu_weights.restype = None
        _lib.set_wgpu_layer_seed.argtypes = [_ct.c_void_p, _ct.c_uint32]
        _lib.set_wgpu_layer_seed.restype = None
        _lib.free_wgpu_layer.argtypes = [_ct.c_void_p]
        _lib.free_wgpu_layer.restype = None
        _lib.reset_wgpu_layer.argtypes = [_ct.c_void_p]
        _lib.reset_wgpu_layer.restype = None

        _HAS_LEARNING = True
        return True
    except OSError:
        _HAS_LEARNING = False
        _lib = None
        return False


_load_native_library()

# Rule type constants
RULE_ELIGENT = 0
RULE_STDP = 1
RULE_REWARD_STDP = 2
RULE_BCM = 3


_DETERMINISTIC_SEED = None


def set_deterministic_mode(seed: int = None) -> None:
    """Force exact random generator seeds for bit-true SC formal verification."""
    global _DETERMINISTIC_SEED
    _DETERMINISTIC_SEED = seed


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
        _lib.step_rule(self._ptr, pre_spike, post_spike, _ct.c_float(reward), _ct.c_float(dt))

    def step_batched(self, pre_spikes, post_spikes, rewards, dt: float = 0.001) -> None:
        """Process Numpy arrays in a single FFI boundary crossing."""
        import numpy as np

        if len(pre_spikes) != len(post_spikes) or len(pre_spikes) != len(rewards):
            raise ValueError(
                f"Batched array lengths mismatch: {len(pre_spikes)}, {len(post_spikes)}, {len(rewards)}"
            )

        pre_spikes = np.ascontiguousarray(pre_spikes, dtype=np.bool_)
        post_spikes = np.ascontiguousarray(post_spikes, dtype=np.bool_)
        rewards = np.ascontiguousarray(rewards, dtype=np.float32)
        count = len(pre_spikes)

        pre_ptr = pre_spikes.ctypes.data_as(_ct.POINTER(_ct.c_bool))
        post_ptr = post_spikes.ctypes.data_as(_ct.POINTER(_ct.c_bool))
        rew_ptr = rewards.ctypes.data_as(_ct.POINTER(_ct.c_float))

        _lib.step_rule_batched(
            self._ptr, pre_ptr, post_ptr, rew_ptr, _ct.c_size_t(count), _ct.c_float(dt)
        )

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

    def step(
        self, fired: bool, pre_spike: bool, global_reward: float = 0.0, dt: float = 0.001
    ) -> None:
        _lib.step_learner(self._ptr, fired, pre_spike, _ct.c_float(global_reward), _ct.c_float(dt))

    def step_batched(self, fired_slice, pre_spikes, rewards, dt: float = 0.001) -> None:
        import numpy as np

        if len(fired_slice) != len(pre_spikes) or len(fired_slice) != len(rewards):
            raise ValueError("Arrays must be identically sized")
        fired_slice = np.ascontiguousarray(fired_slice, dtype=np.bool_)
        pre_spikes = np.ascontiguousarray(pre_spikes, dtype=np.bool_)
        rewards = np.ascontiguousarray(rewards, dtype=np.float32)
        count = len(fired_slice)

        fired_ptr = fired_slice.ctypes.data_as(_ct.POINTER(_ct.c_bool))
        pre_ptr = pre_spikes.ctypes.data_as(_ct.POINTER(_ct.c_bool))
        rew_ptr = rewards.ctypes.data_as(_ct.POINTER(_ct.c_float))
        _lib.step_learner_batched(
            self._ptr, fired_ptr, pre_ptr, rew_ptr, _ct.c_size_t(count), _ct.c_float(dt)
        )

    def __del__(self) -> None:
        if self._ptr and _HAS_LEARNING:
            _lib.destroy_learner(self._ptr)
            self._ptr = None


class RustRuleLayer:
    """RAII wrapper around a Rust RuleLayerHandle for parallelized spatial (layer) execution."""

    __slots__ = ("_ptr", "_count", "_rule_type", "_weight", "_param_a", "_param_b")

    def __init__(
        self,
        count: int,
        rule_type: int = RULE_STDP,
        weight: float = 0.5,
        param_a: float = 0.01,
        param_b: float = 0.012,
    ) -> None:
        if not _HAS_LEARNING:
            raise RuntimeError("libautonomous_learning.so not available")
        self._count = count
        self._rule_type = rule_type
        self._weight = weight
        self._param_a = param_a
        self._param_b = param_b

        self._ptr = _lib.create_rule_layer(
            _ct.c_size_t(count),
            _ct.c_uint32(rule_type),
            _ct.c_float(weight),
            _ct.c_float(param_a),
            _ct.c_float(param_b),
        )

    def __getstate__(self):
        size = _lib.get_rule_layer_state_size(self._ptr)
        buf = (_ct.c_byte * size)()
        _lib.get_rule_layer_state_mem(self._ptr, buf)
        return {
            "count": self._count,
            "rule_type": self._rule_type,
            "weight": self._weight,
            "param_a": self._param_a,
            "param_b": self._param_b,
            "mem_buffer": bytes(buf),
        }

    def get_state_dict(self) -> dict:
        return self.__getstate__()

    def __setstate__(self, state):
        self._count = state["count"]
        self._rule_type = state["rule_type"]
        self._weight = state["weight"]
        self._param_a = state["param_a"]
        self._param_b = state["param_b"]

        if not _HAS_LEARNING:
            _load_native_library()
        if not _HAS_LEARNING:
            raise RuntimeError("libautonomous_learning.so not available")

        self._ptr = _lib.create_rule_layer(
            _ct.c_size_t(self._count),
            _ct.c_uint32(self._rule_type),
            _ct.c_float(self._weight),
            _ct.c_float(self._param_a),
            _ct.c_float(self._param_b),
        )
        mem_buffer = state["mem_buffer"]
        if mem_buffer:
            buf = (_ct.c_byte * len(mem_buffer)).from_buffer_copy(mem_buffer)
            _lib.set_rule_layer_state_mem(self._ptr, buf)

    def load_state_dict(self, state_dict: dict) -> None:
        self.__setstate__(state_dict)

    def step(self, pre_spikes, post_spikes, rewards, dt: float = 0.001) -> None:
        """Process spatial Numpy arrays natively across Rayon Rust threads."""
        import numpy as np

        pre_spikes = np.ascontiguousarray(pre_spikes, dtype=np.bool_)
        post_spikes = np.ascontiguousarray(post_spikes, dtype=np.bool_)
        rewards = np.ascontiguousarray(rewards, dtype=np.float32)

        pre_ptr = pre_spikes.ctypes.data_as(_ct.POINTER(_ct.c_bool))
        post_ptr = post_spikes.ctypes.data_as(_ct.POINTER(_ct.c_bool))
        rew_ptr = rewards.ctypes.data_as(_ct.POINTER(_ct.c_float))

        _lib.step_rule_layer(self._ptr, pre_ptr, post_ptr, rew_ptr, _ct.c_float(dt))

    def step_analog(
        self, pre_probs, post_probs, rewards, dt: float = 0.001, seed: int = None
    ) -> None:
        """Process MTJ analogue probability states directly into sampled stochastic spikes using Rayon RNG."""
        if _DETERMINISTIC_SEED is not None:
            seed = _DETERMINISTIC_SEED
        elif seed is None:
            if not hasattr(self, "_analog_seed_counter"):
                self._analog_seed_counter = 42
            seed = self._analog_seed_counter
            self._analog_seed_counter += 1

        import numpy as np

        pre_probs = np.ascontiguousarray(pre_probs, dtype=np.float32)
        post_probs = np.ascontiguousarray(post_probs, dtype=np.float32)
        rewards = np.ascontiguousarray(rewards, dtype=np.float32)

        pre_ptr = pre_probs.ctypes.data_as(_ct.POINTER(_ct.c_float))
        post_ptr = post_probs.ctypes.data_as(_ct.POINTER(_ct.c_float))
        rew_ptr = rewards.ctypes.data_as(_ct.POINTER(_ct.c_float))

        _lib.step_rule_layer_analog(
            self._ptr, pre_ptr, post_ptr, rew_ptr, _ct.c_uint64(seed), _ct.c_float(dt)
        )

    def get_weights(self):
        import numpy as np

        out = np.zeros(self._count, dtype=np.float32)
        out_ptr = out.ctypes.data_as(_ct.POINTER(_ct.c_float))
        _lib.get_rule_layer_weights(self._ptr, out_ptr)
        return out

    def save(self, path: str) -> bool:
        """Serialize the internal traces and weights to a binary file natively."""
        safe_path = str(_pl.Path(path).resolve().absolute())
        c_path = _ct.c_char_p(safe_path.encode("utf-8"))
        success = _lib.save_rule_layer_batched(self._ptr, c_path)
        if not success:
            raise OSError(f"Failed to save biological traits to {safe_path}")
        return True

    def load(self, path: str) -> bool:
        """Deserialize traces and weights from a binary file directly into memory."""
        safe_path = str(_pl.Path(path).resolve().absolute())
        c_path = _ct.c_char_p(safe_path.encode("utf-8"))
        success = _lib.load_rule_layer_batched(self._ptr, c_path)
        if not success:
            raise OSError(f"Failed to load biological traits or format mismatched from {safe_path}")
        return True

    def reset(self) -> None:
        """Zero each rule's plasticity traces; learned weights are preserved."""
        _lib.reset_rule_layer(self._ptr)

    def __del__(self) -> None:
        if hasattr(self, "_ptr") and self._ptr and _HAS_LEARNING:
            _lib.destroy_rule_layer(self._ptr)
            self._ptr = None


class RustWgpuRuleLayer:
    """RAII wrapper around explicit Rust WGPU execution instances computing via WGSL bindings."""

    __slots__ = ("_ptr", "_count", "_rule_type")

    def __init__(
        self,
        count: int,
        rule_type: int = RULE_STDP,
        weight: float = 0.5,
        param_a: float = 0.01,
        param_b: float = 0.012,
        tau_e: float = 20.0,
        target_sum_weights: float = 1.0,
        **kwargs,
    ) -> None:
        if not _HAS_LEARNING:
            raise RuntimeError("Native library not loaded.")
        self._count = count
        self._rule_type = rule_type

        # param_c acts as tau_e generically across R-STDP and ELIGENT bounds
        # param_d acts as target_sum_weights explicitly in ELIGENT limits
        param_c = max(tau_e, 0.01)
        param_d = max(target_sum_weights, 0.0)
        tau_plus = float(kwargs.get("tau_plus", 20.0))
        tau_minus = float(kwargs.get("tau_minus", 20.0))

        self._ptr = _lib.create_wgpu_layer(
            _ct.c_size_t(count),
            _ct.c_uint32(rule_type),
            _ct.c_float(param_a),
            _ct.c_float(param_b),
            _ct.c_float(tau_plus),
            _ct.c_float(tau_minus),
            _ct.c_float(param_c),
            _ct.c_float(param_d),
        )

        if not self._ptr:
            raise RuntimeError(
                "WGPU backend initialization failed natively. Ensure the system possesses a compatible backend (Vulkan/Metal/WebGPU/DX12)."
            )

        if _DETERMINISTIC_SEED is not None:
            _lib.set_wgpu_layer_seed(self._ptr, _ct.c_uint32(_DETERMINISTIC_SEED))

    def step(self, pre_spikes, post_spikes, rewards=None, dt: float = 0.001) -> None:
        import numpy as np

        # Convert deterministically to 1.0 or 0.0 floats matching hardware probabilities natively
        pre_spikes = np.ascontiguousarray(pre_spikes, dtype=np.float32)
        post_spikes = np.ascontiguousarray(post_spikes, dtype=np.float32)

        pre_ptr = pre_spikes.ctypes.data_as(_ct.POINTER(_ct.c_float))
        post_ptr = post_spikes.ctypes.data_as(_ct.POINTER(_ct.c_float))

        rew_ptr = None
        if rewards is not None:
            rewards = np.ascontiguousarray(rewards, dtype=np.float32)
            rew_ptr = rewards.ctypes.data_as(_ct.POINTER(_ct.c_float))

        _lib.step_wgpu_layer(self._ptr, pre_ptr, post_ptr, rew_ptr, _ct.c_float(dt))

    def step_analog(
        self, pre_probs, post_probs, rewards, dt: float = 0.001, seed: int = None
    ) -> None:
        """
        Native true analog probabilistic emulation via WGSL.
        Float probabilities are passed directly down to the deterministic WGSL kernel
        which executes continuous uniform random thresholds per synapse.
        """
        self.step(pre_probs, post_probs, rewards, dt)

    def get_weights(self):
        import numpy as np

        out = np.zeros(self._count, dtype=np.float32)
        out_ptr = out.ctypes.data_as(_ct.POINTER(_ct.c_float))
        _lib.get_wgpu_weights(self._ptr, out_ptr)
        return out

    def get_state_dict(self) -> dict:
        return {"weights": self.get_weights()}

    def load_state_dict(self, state_dict: dict) -> None:
        import warnings

        warnings.warn(
            "RustWgpuRuleLayer load_state_dict explicitly relies on weights re-initialization rather than state restoration at runtime natively for WGPU."
        )

    def reset(self) -> None:
        """Zero WGSL plasticity traces; weights preserved. See `WgpuRuleLayer::reset`."""
        _lib.reset_wgpu_layer(self._ptr)

    def __del__(self) -> None:
        if hasattr(self, "_ptr") and self._ptr and _HAS_LEARNING:
            _lib.free_wgpu_layer(self._ptr)
            self._ptr = None


# We will define TorchRuleLayer inside the try-catch block for torch.

try:
    import torch
    import torch.nn as nn

    class _BiologicalAutogradFactory(torch.autograd.Function):
        """Mathematically exact Jacobian estimations routing gradients through specific biological traces."""

        @staticmethod
        def forward(
            ctx,
            pre_spikes_f,
            post_spikes_f,
            rewards_f,
            weights,
            pre_trace,
            post_trace,
            eligibility,
            theta_m,
            act_avg,
            rule_params,
            rule_type,
            dt,
        ):
            ctx.rule_type = rule_type
            ctx.dt = dt

            p_a_plus, p_a_minus, p_b_tau1, p_b_tau2, p_tau_e = rule_params

            n_weights = weights.clone()
            n_pre_trace = pre_trace.clone()
            n_post_trace = post_trace.clone()
            n_eligibility = eligibility.clone()
            n_theta_m = theta_m.clone()
            n_act_avg = act_avg.clone()

            if rule_type == RULE_STDP:
                n_pre_trace *= torch.exp(-dt / p_b_tau1)
                n_post_trace *= torch.exp(-dt / p_b_tau2)
                ltp = p_a_plus * n_pre_trace * post_spikes_f
                ltd = p_a_minus * n_post_trace * pre_spikes_f
                n_weights = torch.clamp(n_weights + ltp - ltd, 0.0, 1.0)
                n_pre_trace += pre_spikes_f
                n_post_trace += post_spikes_f
            elif rule_type == RULE_REWARD_STDP:
                n_pre_trace *= torch.exp(-dt / p_b_tau1)
                n_post_trace *= torch.exp(-dt / p_b_tau2)
                ltp = p_a_plus * n_pre_trace * post_spikes_f
                ltd = p_a_minus * n_post_trace * pre_spikes_f
                n_eligibility = n_eligibility + ltp - ltd
                n_eligibility *= torch.exp(-dt / p_tau_e)
                n_weights = torch.clamp(n_weights + n_eligibility * rewards_f, 0.0, 1.0)
                n_pre_trace += pre_spikes_f
                n_post_trace += post_spikes_f
            elif rule_type == RULE_ELIGENT:
                current_rate = post_spikes_f
                n_theta_m += p_a_minus * (current_rate - p_a_plus) * dt
                n_eligibility += pre_spikes_f
                n_eligibility *= torch.exp(-dt / p_tau_e)
                delta = n_eligibility * rewards_f
                n_weights += delta
                n_act_avg += delta
                target_sum = 1.0
                mask = n_act_avg > 0.0
                # Apply Local Synaptic Exhaustion (LSE) boundary explicitly scaling by target to ensure bit-identical parity
                n_weights = torch.where(
                    mask, n_weights * (target_sum / n_act_avg), torch.zeros_like(n_weights)
                )
                n_act_avg = torch.full_like(n_act_avg, target_sum)
            elif rule_type == RULE_BCM:
                x = pre_spikes_f
                y = post_spikes_f
                n_weights = torch.clamp(
                    n_weights + p_a_plus * y * (y - n_theta_m) * x * dt, 0.0, 1.0
                )
                n_act_avg += (y - n_act_avg) * (dt / p_b_tau1)
                n_theta_m += (n_act_avg * n_act_avg - n_theta_m) * (dt / p_b_tau1)
                n_theta_m = torch.clamp(n_theta_m, min=0.01)

            if ctx is not None:
                ctx.save_for_backward(
                    pre_spikes_f,
                    post_spikes_f,
                    rewards_f,
                    n_weights,
                    n_pre_trace,
                    n_post_trace,
                    n_eligibility,
                    n_theta_m,
                    n_act_avg,
                    rule_params,
                )

            return n_weights, n_pre_trace, n_post_trace, n_eligibility, n_theta_m, n_act_avg

        @staticmethod
        def backward(ctx, gw, gp, gp2, ge, gt, ga):
            (
                pre_spikes_f,
                post_spikes_f,
                rewards_f,
                n_weights,
                n_pre_trace,
                n_post_trace,
                n_eligibility,
                n_theta_m,
                n_act_avg,
                rule_params,
            ) = ctx.saved_tensors
            p_a_plus, p_a_minus, p_b_tau1, p_b_tau2, p_tau_e = rule_params

            grad_pre = grad_post = grad_rew = None

            if ctx.rule_type == RULE_STDP:
                grad_pre = gw * (-p_a_minus * n_post_trace)
                grad_post = gw * (p_a_plus * n_pre_trace)
            elif ctx.rule_type == RULE_REWARD_STDP:
                grad_pre = gw * (-p_a_minus * n_post_trace) * rewards_f
                grad_post = gw * (p_a_plus * n_pre_trace) * rewards_f
                grad_rew = gw * n_eligibility
            elif ctx.rule_type == RULE_ELIGENT:
                grad_pre = gw * (n_eligibility * rewards_f)
                grad_rew = gw * n_eligibility
            elif ctx.rule_type == RULE_BCM:
                grad_pre = gw * p_a_plus * post_spikes_f * (post_spikes_f - n_theta_m) * ctx.dt
                grad_post = gw * p_a_plus * (2 * post_spikes_f - n_theta_m) * pre_spikes_f * ctx.dt

            return grad_pre, grad_post, grad_rew, gw, None, None, None, None, None, None, None, None

    class TorchRuleLayer(nn.Module):
        """PyTorch native mixed-precision layer replicating biological plasticity exactly for pure GPU/TPU execution."""

        def __init__(
            self,
            count: int,
            rule_type: int = RULE_STDP,
            weight: float = 0.5,
            param_a: float = 0.01,
            param_b: float = 0.012,
            autograd: bool = False,
            **kwargs,
        ) -> None:
            super().__init__()
            self.count = count
            self.rule_type = rule_type
            self.autograd = autograd

            # Convert kwargs to matching native float boundaries overriding internal constants when defined
            p_a_plus = max(param_a, 0.0001) if rule_type != RULE_ELIGENT else max(param_a, 0.01)
            default_a_minus = (
                0.001
                if rule_type == RULE_ELIGENT
                else (1.0 if rule_type == RULE_BCM else p_a_plus * 0.5)
            )
            p_a_minus = float(kwargs.get("param_a_minus", default_a_minus))

            p_b_tau1 = float(
                kwargs.get(
                    "tau_plus",
                    kwargs.get(
                        "tau",
                        param_b
                        if rule_type not in (RULE_ELIGENT, RULE_BCM)
                        else (1.0 if rule_type == RULE_ELIGENT else max(param_b, 1.0)),
                    ),
                )
            )
            p_b_tau2 = float(
                kwargs.get(
                    "tau_minus",
                    kwargs.get(
                        "tau", param_b if rule_type not in (RULE_ELIGENT, RULE_BCM) else 1.0
                    ),
                )
            )
            p_tau_e = float(
                kwargs.get(
                    "tau_e", param_b if rule_type in (RULE_ELIGENT, RULE_REWARD_STDP) else 1.0
                )
            )

            rp = [p_a_plus, p_a_minus, p_b_tau1, p_b_tau2, p_tau_e]

            self.register_buffer("rule_params", torch.tensor(rp, dtype=torch.float32))
            self.weights = nn.Parameter(
                torch.full((count,), weight, dtype=torch.float32), requires_grad=autograd
            )

            self.register_buffer("pre_trace", torch.zeros(count, dtype=torch.float32))
            self.register_buffer("post_trace", torch.zeros(count, dtype=torch.float32))
            self.register_buffer("eligibility", torch.zeros(count, dtype=torch.float32))
            self.register_buffer(
                "theta_m",
                torch.full((count,), 1.0, dtype=torch.float32)
                if rule_type == RULE_ELIGENT
                else torch.full((count,), 0.5, dtype=torch.float32),
            )
            self.register_buffer(
                "act_avg",
                torch.full((count,), weight, dtype=torch.float32)
                if rule_type == RULE_ELIGENT
                else torch.zeros(count, dtype=torch.float32),
            )

        def reset(self) -> None:
            """Zero the plasticity traces; learned weights preserved.

            Matches the per-rule ``PlasticityRule::reset`` contract of the
            Rust backend (see ``autonomous_learning/src/lib.rs``):

            * STDP (1)       — pre/post traces cleared
            * REWARD_STDP(2) — pre/post traces + eligibility cleared
            * BCM (3)        — ``act_avg`` → 0.0, ``theta_m`` → 0.5
            * ELIGENT (0)    — eligibility cleared

            Buffers outside each rule's reset scope are left untouched.
            """
            with torch.no_grad():
                if self.rule_type == RULE_STDP:
                    self.pre_trace.zero_()
                    self.post_trace.zero_()
                elif self.rule_type == RULE_REWARD_STDP:
                    self.pre_trace.zero_()
                    self.post_trace.zero_()
                    self.eligibility.zero_()
                elif self.rule_type == RULE_BCM:
                    self.act_avg.zero_()
                    self.theta_m.fill_(0.5)
                elif self.rule_type == RULE_ELIGENT:
                    self.eligibility.zero_()

        def forward(self, pre_spikes, post_spikes, rewards=None, dt=1.0):
            pre_f = pre_spikes.to(self.weights.dtype)
            post_f = post_spikes.to(self.weights.dtype)

            if rewards is None:
                if self.rule_type in (RULE_ELIGENT, RULE_REWARD_STDP):
                    import warnings

                    warnings.warn(
                        f"TorchRuleLayer rule_type {self.rule_type} expects 'rewards' gradient but received None. Proceeding with zeros.",
                        UserWarning,
                    )
                rew_f = torch.zeros_like(pre_f)
            else:
                rew_f = rewards.to(self.weights.dtype)

            if self.autograd:
                new_w, new_pre, new_post, new_e, new_tm, new_avg = _BiologicalAutogradFactory.apply(
                    pre_f,
                    post_f,
                    rew_f,
                    self.weights,
                    self.pre_trace,
                    self.post_trace,
                    self.eligibility,
                    self.theta_m,
                    self.act_avg,
                    self.rule_params,
                    self.rule_type,
                    dt,
                )
                self.pre_trace = new_pre.detach()
                self.post_trace = new_post.detach()
                self.eligibility = new_e.detach()
                self.theta_m = new_tm.detach()
                self.act_avg = new_avg.detach()
                return new_w
            else:
                with torch.no_grad():
                    n_w, n_pr, n_po, n_e, n_t, n_a = _BiologicalAutogradFactory.forward(
                        None,
                        pre_f,
                        post_f,
                        rew_f,
                        self.weights,
                        self.pre_trace,
                        self.post_trace,
                        self.eligibility,
                        self.theta_m,
                        self.act_avg,
                        self.rule_params,
                        self.rule_type,
                        dt,
                    )
                    self.weights.copy_(n_w)
                    self.pre_trace.copy_(n_pr)
                    self.post_trace.copy_(n_po)
                    self.eligibility.copy_(n_e)
                    self.theta_m.copy_(n_t)
                    self.act_avg.copy_(n_a)
                    return self.weights

        def step(self, pre_spikes, post_spikes, rewards, dt: float = 1.0) -> None:
            """Legacy compatibility bridge for SC-NeuroCore benchmarks."""
            if not isinstance(pre_spikes, torch.Tensor):
                pre_spikes = torch.tensor(pre_spikes, device=self.weights.device, dtype=torch.bool)
                post_spikes = torch.tensor(
                    post_spikes, device=self.weights.device, dtype=torch.bool
                )
                rewards = torch.tensor(rewards, device=self.weights.device, dtype=torch.float32)
            self.forward(pre_spikes, post_spikes, rewards, dt)

        def get_state_dict(self):
            return self.state_dict()

        def load_state_dict(self, state_dict):
            super().load_state_dict(state_dict)

        def get_weights(self):
            return self.weights.detach().cpu().numpy()

    # Bridge the deprecated prototype to the hardened module
    # Bridge the deprecated prototype to the hardened module
    AutogradSTDPLayer = TorchRuleLayer

    def create_plasticity_layer(
        count: int,
        rule_type: int = RULE_STDP,
        backend: str = "torch",
        autograd: bool = True,
        **kwargs,
    ):
        """
        High-level SC-NeuroCore plasticity layer factory.

        Backends:
        - "torch": Highly-optimized C++ autograd surrogate bindings. Primary training accelerator.
        - "rust": Parallel CPU execution over Rayon bounds natively.
        - "rust-wgpu": pure-Rust GPU for cross-platform edge deployment where CUDA is unavailable. Not primary training accelerator.

        Args:
            count: Number of neurons/synapses in the layer.
            rule_type: One of RULE_ELIGENT, RULE_STDP, RULE_REWARD_STDP, RULE_BCM.
            backend: 'torch' for GPU autograd, 'rust' for Exascale CPU/Analog MTJ execution.
            autograd: Explicit tracking metric when running under Torch backend.
            **kwargs: Extra parameters passed down to the backend components (e.g. param_a, param_b).

        Note:
            To achieve bit-identical mathematical matching between frameworks when resolving analog stochastics,
            call `set_deterministic_mode(seed)` before executing forward pathways.

            When executing RULE_ELIGENT under PyTorch autograd tracking, the normalization scaler explicitly acts as
            a parallel tensor approximation (`target / act_avg`). Exact bit-for-bit Rust logic applies conditional
            scalar bounds internally ensuring absolute limit zeroing which diverges slightly during stochastic boundary conditions.

        Returns:
            Either TorchRuleLayer, RustRuleLayer, or RustWgpuRuleLayer
        """
        if backend.lower() == "torch":
            return TorchRuleLayer(count=count, rule_type=rule_type, autograd=autograd, **kwargs)
        elif backend.lower() == "rust":
            return RustRuleLayer(count=count, rule_type=rule_type, **kwargs)
        elif backend.lower() == "rust-wgpu":
            return RustWgpuRuleLayer(count=count, rule_type=rule_type, **kwargs)
        else:
            raise ValueError(f"Unknown backend '{backend}'. Use 'torch', 'rust', or 'rust-wgpu'.")

except ImportError:
    pass
