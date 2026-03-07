# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Brian2 ↔ SC-NeuroCore Brunel Network Parameter Translator
==========================================================

Maps Brian2 ``iaf_psc_delta`` Brunel balanced network parameters to 20
SC-NeuroCore neuron/synapse/layer variants for comprehensive characterization.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BrunelParams:
    """Brian2-compatible Brunel balanced network parameters."""

    n_exc: int = 800
    n_inh: int = 200
    conn_prob: float = 0.1
    weight_exc: float = 0.1  # mV (Brian2 delta-PSC jump)
    g_inh: float = 5.0
    sim_ms: float = 1000.0
    dt: float = 0.1  # ms
    v_threshold: float = 20.0  # mV
    v_reset: float = 10.0  # mV
    v_rest: float = 0.0  # mV
    tau_mem: float = 20.0  # ms
    external_rate_hz: float = 20.0
    seed: int = 42

    @property
    def n_total(self) -> int:
        return self.n_exc + self.n_inh

    @property
    def weight_inh(self) -> float:
        return self.g_inh * self.weight_exc


def translate_v1_stochastic_lif(bp: BrunelParams) -> dict:
    """Map Brian2 params to StochasticLIFNeuron with delta-PSC semantics.

    Key fixes vs old benchmark:
    - v_reset actually passed (was defaulting to 0.0)
    - Synaptic events applied as direct v += w (not R*I*dt)
    - Poisson drive as voltage kicks (not steady current)
    """
    return dict(
        neuron_kwargs=dict(
            v_threshold=bp.v_threshold,
            v_reset=bp.v_reset,
            v_rest=bp.v_rest,
            tau_mem=bp.tau_mem,
            dt=bp.dt,
            resistance=1.0,
            noise_std=0.0,
        ),
        weight_exc=bp.weight_exc,
        weight_inh=bp.weight_inh,
        external_rate_hz=bp.external_rate_hz,
        ext_weight=bp.weight_exc,
        delta_psc=True,  # apply as v += w, not current
    )


def translate_v2_rate_matched(bp: BrunelParams) -> dict:
    """Map Brian2 params to VectorizedSCLayer probability-domain params.

    Weight probability = w / v_threshold (fraction of threshold per spike).
    """
    bitstream_length = 4096
    weight_prob = bp.weight_exc / bp.v_threshold
    # External drive as input probability: rate * dt / 1000 per step
    ext_prob = bp.external_rate_hz * bp.dt / 1000.0
    return dict(
        n_inputs=bp.n_total,
        n_neurons=bp.n_total,
        bitstream_length=bitstream_length,
        weight_prob=np.clip(weight_prob, 0.0, 1.0),
        ext_prob=np.clip(ext_prob, 0.0, 1.0),
        conn_prob=bp.conn_prob,
        g_inh=bp.g_inh,
    )


def translate_v3_fixed_point(bp: BrunelParams) -> dict:
    """Map Brian2 params to FixedPointLIFNeuron Q8.8 representation.

    Q8.8: multiply by 256 to get fixed-point integer.
    leak_k derived from exp(-dt/tau_mem) ≈ dt/tau_mem for small dt.
    """
    frac = 8
    scale = 1 << frac  # 256

    threshold_q = int(bp.v_threshold * scale)
    reset_q = int(bp.v_reset * scale)
    # leak_k: fraction of (v_rest - v) applied per step
    # Euler: dv/dt = -(v-v_rest)/tau → leak_k = dt/tau_mem in Q8.8
    leak_k = max(1, int((bp.dt / bp.tau_mem) * scale))
    # gain_k = 1.0 in Q8.8 (identity, since we apply weights directly)
    gain_k = scale
    # Synaptic weight in Q8.8
    j_exc_q = int(bp.weight_exc * scale)
    j_inh_q = int(bp.weight_inh * scale)

    if threshold_q > 32767 or threshold_q < -32768:
        raise OverflowError(f"v_threshold={bp.v_threshold} overflows Q8.8 (max ±127.996)")

    return dict(
        data_width=16,
        fraction=frac,
        v_threshold_q=threshold_q,
        v_reset_q=reset_q,
        leak_k=leak_k,
        gain_k=gain_k,
        j_exc_q=j_exc_q,
        j_inh_q=j_inh_q,
        refractory_period=2,
    )


def translate_v4_hybrid(bp: BrunelParams) -> dict:
    """Map Brian2 params to BitstreamSynapse AND + StochasticLIFNeuron hybrid.

    Spike trains encoded as bitstreams → AND gate with weight bitstream →
    popcount → scale to current → StochasticLIFNeuron.
    """
    bitstream_length = 256
    # Weight as probability for BitstreamSynapse: w / v_threshold
    w_prob = np.clip(bp.weight_exc / bp.v_threshold, 0.001, 0.999)
    # Popcount-to-current scale: reconstruct mV from bit fraction
    popcount_scale = bp.v_threshold / bitstream_length

    return dict(
        neuron_kwargs=dict(
            v_threshold=bp.v_threshold,
            v_reset=bp.v_reset,
            v_rest=bp.v_rest,
            tau_mem=bp.tau_mem,
            dt=bp.dt,
            resistance=1.0,
            noise_std=0.0,
        ),
        synapse_kwargs=dict(
            w_min=0.0,
            w_max=1.0,
            length=bitstream_length,
            w=w_prob,
        ),
        bitstream_length=bitstream_length,
        popcount_scale=popcount_scale,
        external_rate_hz=bp.external_rate_hz,
    )


def translate_v5_izhikevich(bp: BrunelParams) -> dict:
    """Map Brian2 LIF to Izhikevich regular-spiking params.

    Izhikevich dynamics: v'=0.04v²+5v+140-u+I, fires at v>=30.
    The model needs ~10 pA sustained current to fire from rest (v=-65).
    Weight scaled to provide comparable excitatory drive.
    """
    # Izhikevich needs ~10 current units for regular spiking from rest.
    # LIF weight of 5mV with threshold 20mV → 25% of threshold.
    # For Izh: 25% of the ~40 current units needed → ~10 per spike.
    scale = 10.0 / max(bp.weight_exc, 0.01)
    return dict(
        neuron_kwargs=dict(
            a=0.02,
            b=0.2,
            c=-65.0,
            d=8.0,
            dt=0.25,  # Izhikevich needs finer dt for stability
            noise_std=0.0,
        ),
        weight_exc=bp.weight_exc * scale,
        weight_inh=bp.weight_inh * scale,
        external_rate_hz=bp.external_rate_hz,
        ext_weight=bp.weight_exc * scale,
        delta_psc=True,
    )


def translate_v6_homeostatic(bp: BrunelParams) -> dict:
    """Map Brian2 params to HomeostaticLIFNeuron.

    target_rate derived from expected Brian2 mean firing rate (~30 Hz → 0.03
    probability per ms step at dt=0.1ms).
    """
    target_rate = 0.003  # ~30 Hz * dt/1000
    return dict(
        neuron_kwargs=dict(
            v_threshold=bp.v_threshold,
            v_reset=bp.v_reset,
            v_rest=bp.v_rest,
            tau_mem=bp.tau_mem,
            dt=bp.dt,
            resistance=1.0,
            noise_std=0.0,
            target_rate=target_rate,
            adaptation_rate=0.001,
        ),
        weight_exc=bp.weight_exc,
        weight_inh=bp.weight_inh,
        external_rate_hz=bp.external_rate_hz,
        ext_weight=bp.weight_exc,
        delta_psc=True,
    )


def translate_v7_noisy(bp: BrunelParams) -> dict:
    """V1 + Gaussian membrane noise (noise_std=1.0 mV)."""
    base = translate_v1_stochastic_lif(bp)
    base["neuron_kwargs"]["noise_std"] = 1.0
    return base


def translate_v8_refractory(bp: BrunelParams) -> dict:
    """V1 + refractory period of 5 steps (0.5 ms at dt=0.1)."""
    base = translate_v1_stochastic_lif(bp)
    base["neuron_kwargs"]["refractory_period"] = 5
    return base


def translate_v9_post_kick(bp: BrunelParams) -> dict:
    """V1 but delta-PSC applied AFTER step() to match Brian2 timing."""
    base = translate_v1_stochastic_lif(bp)
    base["kick_after_step"] = True
    return base


def translate_v10_exact_leak(bp: BrunelParams) -> dict:
    """V1 but using exact exponential leak exp(-dt/tau) instead of Euler."""
    base = translate_v1_stochastic_lif(bp)
    base["exact_leak"] = True
    base["leak_factor"] = float(np.exp(-bp.dt / bp.tau_mem))
    return base


def translate_v11_q16(bp: BrunelParams) -> dict:
    """Map Brian2 params to FixedPointLIFNeuron Q16.12 (12 fractional bits, 32-bit).

    Higher precision than V3 Q8.8 with wider data path to avoid overflow.
    """
    frac = 12
    scale = 1 << frac  # 4096

    threshold_q = int(bp.v_threshold * scale)
    reset_q = int(bp.v_reset * scale)
    leak_k = max(1, int((bp.dt / bp.tau_mem) * scale))
    gain_k = scale
    j_exc_q = int(bp.weight_exc * scale)
    j_inh_q = int(bp.weight_inh * scale)

    max_val = (1 << 31) - 1
    if threshold_q > max_val or threshold_q < -max_val:
        raise OverflowError(f"v_threshold={bp.v_threshold} overflows Q16.12 32-bit")

    return dict(
        data_width=32,
        fraction=frac,
        v_threshold_q=threshold_q,
        v_reset_q=reset_q,
        leak_k=leak_k,
        gain_k=gain_k,
        j_exc_q=j_exc_q,
        j_inh_q=j_inh_q,
        refractory_period=2,
    )


def translate_v12_stdp(bp: BrunelParams) -> dict:
    """V1 + StochasticSTDPSynapse for online weight learning."""
    base = translate_v1_stochastic_lif(bp)
    base["stdp"] = True
    base["stdp_kwargs"] = dict(
        w_min=0.0,
        w_max=1.0,
        length=256,
        w=np.clip(bp.weight_exc / bp.v_threshold, 0.001, 0.999),
        learning_rate=0.001,
        window_size=5,
    )
    return base


def translate_v13_dot_product(bp: BrunelParams) -> dict:
    """Map to BitstreamDotProduct multi-channel SC summation."""
    w_prob = np.clip(bp.weight_exc / bp.v_threshold, 0.001, 0.999)
    return dict(
        neuron_kwargs=dict(
            v_threshold=bp.v_threshold,
            v_reset=bp.v_reset,
            v_rest=bp.v_rest,
            tau_mem=bp.tau_mem,
            dt=bp.dt,
            resistance=1.0,
            noise_std=0.0,
        ),
        synapse_w_prob=w_prob,
        bitstream_length=256,
        external_rate_hz=bp.external_rate_hz,
        ext_weight=bp.weight_exc,
        delta_psc=True,
    )


def translate_v14_sobol(bp: BrunelParams) -> dict:
    """V4 hybrid but with Sobol low-discrepancy bitstream encoding."""
    base = translate_v4_hybrid(bp)
    base["encoding_mode"] = "sobol"
    return base


def translate_v15_jax(bp: BrunelParams) -> dict:
    """Map to JaxSCDenseLayer params."""
    return dict(
        n_neurons=bp.n_total,
        n_inputs=bp.n_total,
        bitstream_length=1024,
        neuron_params=dict(
            v_rest=bp.v_rest,
            v_reset=bp.v_reset,
            v_threshold=bp.v_threshold,
            tau_mem=bp.tau_mem,
            resistance=1.0,
            noise_std=0.0,
        ),
        weight_exc=bp.weight_exc,
        weight_inh=bp.weight_inh,
        external_rate_hz=bp.external_rate_hz,
        conn_prob=bp.conn_prob,
        g_inh=bp.g_inh,
    )


def translate_v16_recurrent(bp: BrunelParams) -> dict:
    """Map to SCRecurrentLayer reservoir computing params."""
    return dict(
        n_inputs=bp.n_total,
        n_neurons=bp.n_total,
        feedback_strength=0.3,
        input_strength=bp.weight_exc / bp.v_threshold,
        spectral_radius=0.9,
        length=256,
        external_rate_hz=bp.external_rate_hz,
    )


def translate_v17_memristive(bp: BrunelParams) -> dict:
    """Map to MemristiveDenseLayer with device defects."""
    return dict(
        n_inputs=bp.n_total,
        n_neurons=bp.n_total,
        length=256,
        stuck_rate=0.01,
        variability=0.05,
        weight_prob=np.clip(bp.weight_exc / bp.v_threshold, 0.0, 1.0),
        ext_prob=np.clip(bp.external_rate_hz * bp.dt / 1000.0, 0.0, 1.0),
        conn_prob=bp.conn_prob,
        g_inh=bp.g_inh,
    )


def translate_v18_numba(bp: BrunelParams) -> dict:
    """V1 params with numba JIT flag for inner loop."""
    base = translate_v1_stochastic_lif(bp)
    base["numba_jit"] = True
    return base


def translate_v19_pytorch_cuda(bp: BrunelParams) -> dict:
    """Map to PyTorch CUDA tensors for GPU LIF simulation."""
    return dict(
        v_threshold=bp.v_threshold,
        v_reset=bp.v_reset,
        v_rest=bp.v_rest,
        tau_mem=bp.tau_mem,
        dt=bp.dt,
        weight_exc=bp.weight_exc,
        weight_inh=bp.weight_inh,
        external_rate_hz=bp.external_rate_hz,
        conn_prob=bp.conn_prob,
        n_exc=bp.n_exc,
        n_total=bp.n_total,
        g_inh=bp.g_inh,
    )


def translate_v20_vectorized_numpy(bp: BrunelParams) -> dict:
    """V1 params for batch NumPy neuron update (no per-neuron loop)."""
    return dict(
        v_threshold=bp.v_threshold,
        v_reset=bp.v_reset,
        v_rest=bp.v_rest,
        tau_mem=bp.tau_mem,
        dt=bp.dt,
        weight_exc=bp.weight_exc,
        weight_inh=bp.weight_inh,
        external_rate_hz=bp.external_rate_hz,
        conn_prob=bp.conn_prob,
        n_exc=bp.n_exc,
        n_total=bp.n_total,
        g_inh=bp.g_inh,
    )
