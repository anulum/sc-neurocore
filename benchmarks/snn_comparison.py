# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""
SNN Simulator Comparison Benchmark — 20 SC-NeuroCore Variants
==============================================================

Head-to-head comparison of 20 SC-NeuroCore neuron/synapse/layer variants
against Brian2's ``iaf_psc_delta`` on the Brunel balanced network.

Usage::

    python benchmarks/snn_comparison.py              # V1 only (quick)
    python benchmarks/snn_comparison.py --all         # all 20 variants + Brian2
    python benchmarks/snn_comparison.py --json results.json --markdown
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from brunel_translator import (
    BrunelParams,
    translate_v1_stochastic_lif,
    translate_v2_rate_matched,
    translate_v3_fixed_point,
    translate_v4_hybrid,
    translate_v5_izhikevich,
    translate_v6_homeostatic,
    translate_v7_noisy,
    translate_v8_refractory,
    translate_v9_post_kick,
    translate_v10_exact_leak,
    translate_v11_q16,
    translate_v12_stdp,
    translate_v13_dot_product,
    translate_v14_sobol,
    translate_v15_jax,
    translate_v16_recurrent,
    translate_v17_memristive,
    translate_v18_numba,
    translate_v19_pytorch_cuda,
    translate_v20_vectorized_numpy,
)


@dataclass
class VariantResult:
    variant: str
    total_spikes: int
    mean_rate_hz: float
    wall_time_s: float
    brian2_spikes: int = 0
    brian2_rate_hz: float = 0.0
    rate_ratio: float = 0.0
    metric_note: str = ""
    params: dict | None = None
    domain: str = "spike"
    mean_output_prob: float | None = None
    status: str = "ok"
    reason: str = ""


# Brunel (2000) AI regime: g=5, eta=2
_ETA = 2.0


def _brunel_ext_lambda(bp: BrunelParams) -> float:
    """Poisson lambda per neuron per timestep for Brunel external drive.

    Brunel (2000): nu_thr = V_th / (J * C_E * tau_m), nu_ext = eta * nu_thr.
    Total events per neuron per step = C_E * nu_ext * dt_s.
    """
    c_e = bp.conn_prob * bp.n_exc
    if c_e == 0:
        return 0.0
    nu_thr = bp.v_threshold / (bp.weight_exc * c_e * bp.tau_mem * 1e-3)
    nu_ext = _ETA * nu_thr
    return c_e * nu_ext * bp.dt / 1000.0


# ---------------------------------------------------------------------------
# Brian2 reference
# ---------------------------------------------------------------------------
def run_brian2_reference(bp: BrunelParams) -> VariantResult | None:
    try:
        import brian2
    except ImportError:
        return None

    brian2.start_scope()

    eqs = """
    dv/dt = -v / (tau * ms) : 1
    tau : 1
    """
    G = brian2.NeuronGroup(
        bp.n_total,
        eqs,
        threshold="v > v_th",
        reset="v = v_reset",
        method="euler",
        dt=bp.dt * brian2.ms,
    )
    G.v = 0
    G.tau = bp.tau_mem
    G.namespace["v_th"] = bp.v_threshold
    G.namespace["v_reset"] = bp.v_reset

    S_exc = brian2.Synapses(G[: bp.n_exc], G, on_pre="v_post += w", dt=bp.dt * brian2.ms)
    S_exc.connect(p=bp.conn_prob)
    S_exc.namespace["w"] = bp.weight_exc

    S_inh = brian2.Synapses(G[bp.n_exc :], G, on_pre="v_post -= w", dt=bp.dt * brian2.ms)
    S_inh.connect(p=bp.conn_prob)
    S_inh.namespace["w"] = bp.weight_inh

    # Independent Poisson input per neuron (Brunel 2000).
    # PoissonInput avoids correlated-drive artifact of shared PoissonGroup.
    c_ext = int(bp.conn_prob * bp.n_exc)
    c_e = bp.conn_prob * bp.n_exc
    nu_thr = bp.v_threshold / (bp.weight_exc * c_e * bp.tau_mem * 1e-3) if c_e > 0 else 20.0
    nu_ext = _ETA * nu_thr
    _P_ext = brian2.PoissonInput(G, "v", N=c_ext, rate=nu_ext * brian2.Hz, weight=bp.weight_exc)

    mon = brian2.SpikeMonitor(G)

    t0 = time.perf_counter()
    brian2.run(bp.sim_ms * brian2.ms)
    wall = time.perf_counter() - t0

    rate = mon.num_spikes / (bp.sim_ms / 1000.0) / bp.n_total
    return VariantResult(
        variant="brian2_reference",
        total_spikes=mon.num_spikes,
        mean_rate_hz=rate,
        wall_time_s=wall,
    )


# ---------------------------------------------------------------------------
# V1: StochasticLIF — delta-PSC bug fix
# ---------------------------------------------------------------------------
def run_v1_stochastic_lif(bp: BrunelParams) -> VariantResult:
    from sc_neurocore import StochasticLIFNeuron

    params = translate_v1_stochastic_lif(bp)
    rng = np.random.default_rng(bp.seed)

    neurons = [StochasticLIFNeuron(**params["neuron_kwargs"]) for _ in range(bp.n_total)]

    conn_mask = rng.random((bp.n_total, bp.n_total)) < bp.conn_prob
    np.fill_diagonal(conn_mask, False)
    weights = np.where(conn_mask, params["weight_exc"], 0.0)
    weights[bp.n_exc :, :] *= -bp.g_inh

    steps = int(bp.sim_ms / bp.dt)
    spike_count = 0
    prev_spikes = np.zeros(bp.n_total, dtype=bool)

    t0 = time.perf_counter()
    for _ in range(steps):
        # Poisson external drive as delta-PSC voltage kicks
        ext_events = rng.poisson(_brunel_ext_lambda(bp), bp.n_total)

        # Synaptic input from previous spikes (delta-PSC)
        if prev_spikes.any():
            syn_dv = weights[prev_spikes].sum(axis=0)
        else:
            syn_dv = np.zeros(bp.n_total)

        spikes = np.zeros(bp.n_total, dtype=bool)
        for i, n in enumerate(neurons):
            # Apply delta-PSC: direct voltage manipulation
            n.v += ext_events[i] * params["ext_weight"] + syn_dv[i]
            # Leak-only step (no input current)
            spikes[i] = n.step(0.0)

        prev_spikes = spikes
        spike_count += int(spikes.sum())

    wall = time.perf_counter() - t0
    rate = spike_count / (bp.sim_ms / 1000.0) / bp.n_total

    return VariantResult(
        variant="v1_stochastic_lif",
        total_spikes=spike_count,
        mean_rate_hz=rate,
        wall_time_s=wall,
        params=params["neuron_kwargs"],
    )


# ---------------------------------------------------------------------------
# V2: Rate-matched SC bitstream (probability domain)
# ---------------------------------------------------------------------------
def run_v2_rate_matched(bp: BrunelParams) -> VariantResult:
    from sc_neurocore import VectorizedSCLayer

    params = translate_v2_rate_matched(bp)
    # Use a representative 100-neuron subset (full N² bitstream allocation is too slow)
    n_sub = min(100, bp.n_total)
    bl = min(1024, params["bitstream_length"])
    layer = VectorizedSCLayer(n_inputs=n_sub, n_neurons=n_sub, length=bl)

    rng = np.random.default_rng(bp.seed)
    conn = rng.random((n_sub, n_sub)) < params["conn_prob"]
    np.fill_diagonal(conn, False)
    layer.weights = np.where(conn, params["weight_prob"], 0.0)
    layer._refresh_packed_weights()

    steps = int(bp.sim_ms / bp.dt)
    input_probs = np.full(n_sub, params["ext_prob"])

    t0 = time.perf_counter()
    total_output = np.zeros(n_sub)
    for _ in range(steps):
        out = layer.forward(input_probs)
        total_output += out
    wall = time.perf_counter() - t0

    mean_out_prob = total_output.mean() / steps

    return VariantResult(
        variant="v2_rate_matched",
        total_spikes=0,
        mean_rate_hz=0.0,
        wall_time_s=wall,
        metric_note=f"n_sub={n_sub}, bl={bl}",
        domain="probability",
        mean_output_prob=mean_out_prob,
    )


# ---------------------------------------------------------------------------
# V3: FixedPointLIF Q8.8 (hardware-faithful)
# ---------------------------------------------------------------------------
def run_v3_fixed_point(bp: BrunelParams) -> VariantResult:
    from sc_neurocore import FixedPointLIFNeuron

    params = translate_v3_fixed_point(bp)
    neurons = [
        FixedPointLIFNeuron(
            data_width=params["data_width"],
            fraction=params["fraction"],
            v_threshold=params["v_threshold_q"],
            v_reset=params["v_reset_q"],
            refractory_period=params["refractory_period"],
        )
        for _ in range(bp.n_total)
    ]

    rng = np.random.default_rng(bp.seed)
    conn_mask = rng.random((bp.n_total, bp.n_total)) < bp.conn_prob
    np.fill_diagonal(conn_mask, False)
    # Weights in Q8.8: exc positive, inh negative
    w_q = np.where(conn_mask, params["j_exc_q"], 0)
    w_q[bp.n_exc :, :] = np.where(conn_mask[bp.n_exc :, :], -params["j_inh_q"], 0)

    steps = int(bp.sim_ms / bp.dt)
    spike_count = 0
    prev_spikes = np.zeros(bp.n_total, dtype=bool)

    t0 = time.perf_counter()
    for _ in range(steps):
        ext_events = rng.poisson(_brunel_ext_lambda(bp), bp.n_total)

        if prev_spikes.any():
            syn_q = w_q[prev_spikes].sum(axis=0)
        else:
            syn_q = np.zeros(bp.n_total, dtype=int)

        spikes = np.zeros(bp.n_total, dtype=bool)
        for i, n in enumerate(neurons):
            # External Poisson as Q8.8 current
            I_ext = int(ext_events[i]) * params["j_exc_q"]
            I_total = int(syn_q[i]) + I_ext
            spike, _ = n.step(
                leak_k=params["leak_k"],
                gain_k=params["gain_k"],
                I_t=I_total,
            )
            spikes[i] = bool(spike)

        prev_spikes = spikes
        spike_count += int(spikes.sum())

    wall = time.perf_counter() - t0
    rate = spike_count / (bp.sim_ms / 1000.0) / bp.n_total

    return VariantResult(
        variant="v3_fixed_point_q88",
        total_spikes=spike_count,
        mean_rate_hz=rate,
        wall_time_s=wall,
    )


# ---------------------------------------------------------------------------
# V4: Hybrid SC synapses + analog LIF
# ---------------------------------------------------------------------------
def run_v4_hybrid(bp: BrunelParams) -> VariantResult:
    from sc_neurocore import StochasticLIFNeuron, BitstreamSynapse

    params = translate_v4_hybrid(bp)
    neurons = [StochasticLIFNeuron(**params["neuron_kwargs"]) for _ in range(bp.n_total)]

    rng = np.random.default_rng(bp.seed)
    conn_mask = rng.random((bp.n_total, bp.n_total)) < bp.conn_prob
    np.fill_diagonal(conn_mask, False)

    syn_exc = BitstreamSynapse(**params["synapse_kwargs"])
    inh_w = min(bp.g_inh * params["synapse_kwargs"]["w"], 0.999)
    syn_inh = BitstreamSynapse(**{**params["synapse_kwargs"], "w": inh_w})

    # Pre-compute effective weight per connection via bitstream AND.
    # For each active connection, the expected current contribution is:
    # P(AND=1) * popcount_scale = P(pre) * P(weight) * bitstream_length * popcount_scale
    # We precompute this as a weight matrix to vectorize the inner loop.
    bl = params["bitstream_length"]
    eff_exc = syn_exc.effective_weight_probability() * bl * params["popcount_scale"]
    eff_inh = syn_inh.effective_weight_probability() * bl * params["popcount_scale"]
    eff_weights = np.where(conn_mask, eff_exc, 0.0)
    eff_weights[bp.n_exc :, :] = np.where(conn_mask[bp.n_exc :, :], -eff_inh, 0.0)

    steps = int(bp.sim_ms / bp.dt)
    spike_count = 0
    prev_spikes = np.zeros(bp.n_total, dtype=bool)

    t0 = time.perf_counter()
    for _ in range(steps):
        ext_events = rng.poisson(_brunel_ext_lambda(bp), bp.n_total)

        # Vectorized synaptic input (stochastic modulation via random scaling)
        if prev_spikes.any():
            # Modulate each active synapse by a Bernoulli(0.5) pre-spike bitstream
            noise = rng.random(prev_spikes.sum())
            syn_dv = (eff_weights[prev_spikes] * noise[:, None]).sum(axis=0)
        else:
            syn_dv = np.zeros(bp.n_total)

        spikes = np.zeros(bp.n_total, dtype=bool)
        for i, n in enumerate(neurons):
            n.v += ext_events[i] * bp.weight_exc
            n.v += syn_dv[i]
            spikes[i] = bool(n.step(0.0))

        prev_spikes = spikes
        spike_count += int(spikes.sum())

    wall = time.perf_counter() - t0
    rate = spike_count / (bp.sim_ms / 1000.0) / bp.n_total

    return VariantResult(
        variant="v4_hybrid_sc_lif",
        total_spikes=spike_count,
        mean_rate_hz=rate,
        wall_time_s=wall,
    )


# ---------------------------------------------------------------------------
# V5: Izhikevich neuron — burst dynamics
# ---------------------------------------------------------------------------
def run_v5_izhikevich(bp: BrunelParams) -> VariantResult:
    from sc_neurocore import SCIzhikevichNeuron

    params = translate_v5_izhikevich(bp)
    rng = np.random.default_rng(bp.seed)

    izh_dt = params["neuron_kwargs"]["dt"]  # 0.25 ms
    neurons = [SCIzhikevichNeuron(**params["neuron_kwargs"]) for _ in range(bp.n_total)]

    conn_mask = rng.random((bp.n_total, bp.n_total)) < bp.conn_prob
    np.fill_diagonal(conn_mask, False)
    weights = np.where(conn_mask, params["weight_exc"], 0.0)
    weights[bp.n_exc :, :] *= -bp.g_inh

    # Run at Izhikevich dt, not Brian2 dt
    steps = int(bp.sim_ms / izh_dt)
    spike_count = 0
    prev_spikes = np.zeros(bp.n_total, dtype=bool)

    t0 = time.perf_counter()
    for _ in range(steps):
        ext_events = rng.poisson(bp.external_rate_hz * izh_dt / 1000.0, bp.n_total)

        if prev_spikes.any():
            syn_I = weights[prev_spikes].sum(axis=0)
        else:
            syn_I = np.zeros(bp.n_total)

        spikes = np.zeros(bp.n_total, dtype=bool)
        for i, n in enumerate(neurons):
            # Izhikevich needs baseline tonic current (~5) to stay near threshold
            I_total = 5.0 + ext_events[i] * params["ext_weight"] + syn_I[i]
            spikes[i] = n.step(I_total)

        prev_spikes = spikes
        spike_count += int(spikes.sum())

    wall = time.perf_counter() - t0
    rate = spike_count / (bp.sim_ms / 1000.0) / bp.n_total
    return VariantResult(
        variant="v5_izhikevich", total_spikes=spike_count, mean_rate_hz=rate, wall_time_s=wall
    )


# ---------------------------------------------------------------------------
# V6: Homeostatic LIF — self-regulating threshold
# ---------------------------------------------------------------------------
def run_v6_homeostatic(bp: BrunelParams) -> VariantResult:
    from sc_neurocore import HomeostaticLIFNeuron

    params = translate_v6_homeostatic(bp)
    rng = np.random.default_rng(bp.seed)

    neurons = [HomeostaticLIFNeuron(**params["neuron_kwargs"]) for _ in range(bp.n_total)]

    conn_mask = rng.random((bp.n_total, bp.n_total)) < bp.conn_prob
    np.fill_diagonal(conn_mask, False)
    weights = np.where(conn_mask, params["weight_exc"], 0.0)
    weights[bp.n_exc :, :] *= -bp.g_inh

    steps = int(bp.sim_ms / bp.dt)
    spike_count = 0
    prev_spikes = np.zeros(bp.n_total, dtype=bool)

    t0 = time.perf_counter()
    for _ in range(steps):
        ext_events = rng.poisson(_brunel_ext_lambda(bp), bp.n_total)

        if prev_spikes.any():
            syn_dv = weights[prev_spikes].sum(axis=0)
        else:
            syn_dv = np.zeros(bp.n_total)

        spikes = np.zeros(bp.n_total, dtype=bool)
        for i, n in enumerate(neurons):
            n.v += ext_events[i] * params["ext_weight"] + syn_dv[i]
            spikes[i] = n.step(0.0)

        prev_spikes = spikes
        spike_count += int(spikes.sum())

    wall = time.perf_counter() - t0
    rate = spike_count / (bp.sim_ms / 1000.0) / bp.n_total
    return VariantResult(
        variant="v6_homeostatic_lif", total_spikes=spike_count, mean_rate_hz=rate, wall_time_s=wall
    )


# ---------------------------------------------------------------------------
# V7: Noisy LIF — stochastic membrane noise
# ---------------------------------------------------------------------------
def run_v7_noisy(bp: BrunelParams) -> VariantResult:
    from sc_neurocore import StochasticLIFNeuron

    params = translate_v7_noisy(bp)
    rng = np.random.default_rng(bp.seed)

    neurons = [StochasticLIFNeuron(**params["neuron_kwargs"]) for _ in range(bp.n_total)]

    conn_mask = rng.random((bp.n_total, bp.n_total)) < bp.conn_prob
    np.fill_diagonal(conn_mask, False)
    weights = np.where(conn_mask, params["weight_exc"], 0.0)
    weights[bp.n_exc :, :] *= -bp.g_inh

    steps = int(bp.sim_ms / bp.dt)
    spike_count = 0
    prev_spikes = np.zeros(bp.n_total, dtype=bool)

    t0 = time.perf_counter()
    for _ in range(steps):
        ext_events = rng.poisson(_brunel_ext_lambda(bp), bp.n_total)

        if prev_spikes.any():
            syn_dv = weights[prev_spikes].sum(axis=0)
        else:
            syn_dv = np.zeros(bp.n_total)

        spikes = np.zeros(bp.n_total, dtype=bool)
        for i, n in enumerate(neurons):
            n.v += ext_events[i] * params["ext_weight"] + syn_dv[i]
            spikes[i] = n.step(0.0)

        prev_spikes = spikes
        spike_count += int(spikes.sum())

    wall = time.perf_counter() - t0
    rate = spike_count / (bp.sim_ms / 1000.0) / bp.n_total
    return VariantResult(
        variant="v7_noisy_lif", total_spikes=spike_count, mean_rate_hz=rate, wall_time_s=wall
    )


# ---------------------------------------------------------------------------
# V8: Refractory LIF — 5-step refractory period
# ---------------------------------------------------------------------------
def run_v8_refractory(bp: BrunelParams) -> VariantResult:
    from sc_neurocore import StochasticLIFNeuron

    params = translate_v8_refractory(bp)
    rng = np.random.default_rng(bp.seed)

    neurons = [StochasticLIFNeuron(**params["neuron_kwargs"]) for _ in range(bp.n_total)]

    conn_mask = rng.random((bp.n_total, bp.n_total)) < bp.conn_prob
    np.fill_diagonal(conn_mask, False)
    weights = np.where(conn_mask, params["weight_exc"], 0.0)
    weights[bp.n_exc :, :] *= -bp.g_inh

    steps = int(bp.sim_ms / bp.dt)
    spike_count = 0
    prev_spikes = np.zeros(bp.n_total, dtype=bool)

    t0 = time.perf_counter()
    for _ in range(steps):
        ext_events = rng.poisson(_brunel_ext_lambda(bp), bp.n_total)

        if prev_spikes.any():
            syn_dv = weights[prev_spikes].sum(axis=0)
        else:
            syn_dv = np.zeros(bp.n_total)

        spikes = np.zeros(bp.n_total, dtype=bool)
        for i, n in enumerate(neurons):
            n.v += ext_events[i] * params["ext_weight"] + syn_dv[i]
            spikes[i] = n.step(0.0)

        prev_spikes = spikes
        spike_count += int(spikes.sum())

    wall = time.perf_counter() - t0
    rate = spike_count / (bp.sim_ms / 1000.0) / bp.n_total
    return VariantResult(
        variant="v8_refractory_lif", total_spikes=spike_count, mean_rate_hz=rate, wall_time_s=wall
    )


# ---------------------------------------------------------------------------
# V9: Post-kick LIF — delta-PSC AFTER step() (Brian2 timing)
# ---------------------------------------------------------------------------
def run_v9_post_kick(bp: BrunelParams) -> VariantResult:
    from sc_neurocore import StochasticLIFNeuron

    params = translate_v9_post_kick(bp)
    rng = np.random.default_rng(bp.seed)

    neurons = [StochasticLIFNeuron(**params["neuron_kwargs"]) for _ in range(bp.n_total)]

    conn_mask = rng.random((bp.n_total, bp.n_total)) < bp.conn_prob
    np.fill_diagonal(conn_mask, False)
    weights = np.where(conn_mask, params["weight_exc"], 0.0)
    weights[bp.n_exc :, :] *= -bp.g_inh

    steps = int(bp.sim_ms / bp.dt)
    spike_count = 0
    prev_spikes = np.zeros(bp.n_total, dtype=bool)

    t0 = time.perf_counter()
    for _ in range(steps):
        ext_events = rng.poisson(_brunel_ext_lambda(bp), bp.n_total)

        if prev_spikes.any():
            syn_dv = weights[prev_spikes].sum(axis=0)
        else:
            syn_dv = np.zeros(bp.n_total)

        spikes = np.zeros(bp.n_total, dtype=bool)
        for i, n in enumerate(neurons):
            # Step FIRST, then apply delta-PSC (Brian2 ordering)
            spikes[i] = n.step(0.0)
            n.v += ext_events[i] * params["ext_weight"] + syn_dv[i]

        prev_spikes = spikes
        spike_count += int(spikes.sum())

    wall = time.perf_counter() - t0
    rate = spike_count / (bp.sim_ms / 1000.0) / bp.n_total
    return VariantResult(
        variant="v9_post_kick_lif", total_spikes=spike_count, mean_rate_hz=rate, wall_time_s=wall
    )


# ---------------------------------------------------------------------------
# V10: Exact-leak LIF — exp(-dt/tau) instead of Euler
# ---------------------------------------------------------------------------
def run_v10_exact_leak(bp: BrunelParams) -> VariantResult:
    from sc_neurocore import StochasticLIFNeuron

    params = translate_v10_exact_leak(bp)
    leak_factor = params["leak_factor"]
    rng = np.random.default_rng(bp.seed)

    neurons = [StochasticLIFNeuron(**params["neuron_kwargs"]) for _ in range(bp.n_total)]

    conn_mask = rng.random((bp.n_total, bp.n_total)) < bp.conn_prob
    np.fill_diagonal(conn_mask, False)
    weights = np.where(conn_mask, params["weight_exc"], 0.0)
    weights[bp.n_exc :, :] *= -bp.g_inh

    steps = int(bp.sim_ms / bp.dt)
    spike_count = 0
    prev_spikes = np.zeros(bp.n_total, dtype=bool)

    t0 = time.perf_counter()
    for _ in range(steps):
        ext_events = rng.poisson(_brunel_ext_lambda(bp), bp.n_total)

        if prev_spikes.any():
            syn_dv = weights[prev_spikes].sum(axis=0)
        else:
            syn_dv = np.zeros(bp.n_total)

        spikes = np.zeros(bp.n_total, dtype=bool)
        for i, n in enumerate(neurons):
            n.v += ext_events[i] * params["ext_weight"] + syn_dv[i]
            # Exact exponential leak instead of Euler
            n.v = bp.v_rest + (n.v - bp.v_rest) * leak_factor
            # Threshold check
            if n.v >= bp.v_threshold:
                spikes[i] = True
                n.v = bp.v_reset
            else:
                spikes[i] = False

        prev_spikes = spikes
        spike_count += int(spikes.sum())

    wall = time.perf_counter() - t0
    rate = spike_count / (bp.sim_ms / 1000.0) / bp.n_total
    return VariantResult(
        variant="v10_exact_leak_lif", total_spikes=spike_count, mean_rate_hz=rate, wall_time_s=wall
    )


# ---------------------------------------------------------------------------
# V11: Q4.12 fixed-point — higher precision
# ---------------------------------------------------------------------------
def run_v11_q16(bp: BrunelParams) -> VariantResult:
    from sc_neurocore import FixedPointLIFNeuron

    params = translate_v11_q16(bp)
    neurons = [
        FixedPointLIFNeuron(
            data_width=params["data_width"],
            fraction=params["fraction"],
            v_threshold=params["v_threshold_q"],
            v_reset=params["v_reset_q"],
            refractory_period=params["refractory_period"],
        )
        for _ in range(bp.n_total)
    ]

    rng = np.random.default_rng(bp.seed)
    conn_mask = rng.random((bp.n_total, bp.n_total)) < bp.conn_prob
    np.fill_diagonal(conn_mask, False)
    w_q = np.where(conn_mask, params["j_exc_q"], 0)
    w_q[bp.n_exc :, :] = np.where(conn_mask[bp.n_exc :, :], -params["j_inh_q"], 0)

    steps = int(bp.sim_ms / bp.dt)
    spike_count = 0
    prev_spikes = np.zeros(bp.n_total, dtype=bool)

    t0 = time.perf_counter()
    for _ in range(steps):
        ext_events = rng.poisson(_brunel_ext_lambda(bp), bp.n_total)

        if prev_spikes.any():
            syn_q = w_q[prev_spikes].sum(axis=0)
        else:
            syn_q = np.zeros(bp.n_total, dtype=int)

        spikes = np.zeros(bp.n_total, dtype=bool)
        for i, n in enumerate(neurons):
            I_ext = int(ext_events[i]) * params["j_exc_q"]
            I_total = int(syn_q[i]) + I_ext
            spike, _ = n.step(
                leak_k=params["leak_k"],
                gain_k=params["gain_k"],
                I_t=I_total,
            )
            spikes[i] = bool(spike)

        prev_spikes = spikes
        spike_count += int(spikes.sum())

    wall = time.perf_counter() - t0
    rate = spike_count / (bp.sim_ms / 1000.0) / bp.n_total
    return VariantResult(
        variant="v11_q16_fixed_point", total_spikes=spike_count, mean_rate_hz=rate, wall_time_s=wall
    )


# ---------------------------------------------------------------------------
# V12: STDP — online weight learning
# ---------------------------------------------------------------------------
def run_v12_stdp(bp: BrunelParams) -> VariantResult:
    from sc_neurocore import StochasticLIFNeuron, StochasticSTDPSynapse

    params = translate_v12_stdp(bp)
    rng = np.random.default_rng(bp.seed)

    neurons = [StochasticLIFNeuron(**params["neuron_kwargs"]) for _ in range(bp.n_total)]

    conn_mask = rng.random((bp.n_total, bp.n_total)) < bp.conn_prob
    np.fill_diagonal(conn_mask, False)

    # STDP synapses for a subset of excitatory connections (cap at 2000 for speed)
    stdp_syns = {}
    max_stdp = 2000
    count = 0
    for i in range(bp.n_exc):
        for j in range(bp.n_total):
            if conn_mask[i, j]:
                stdp_syns[(i, j)] = StochasticSTDPSynapse(**params["stdp_kwargs"])
                count += 1
                if count >= max_stdp:
                    break
        if count >= max_stdp:
            break

    weights = np.where(conn_mask, params["weight_exc"], 0.0)
    weights[bp.n_exc :, :] *= -bp.g_inh

    steps = int(bp.sim_ms / bp.dt)
    spike_count = 0
    prev_spikes = np.zeros(bp.n_total, dtype=bool)

    t0 = time.perf_counter()
    for _ in range(steps):
        ext_events = rng.poisson(_brunel_ext_lambda(bp), bp.n_total)

        if prev_spikes.any():
            syn_dv = weights[prev_spikes].sum(axis=0)
        else:
            syn_dv = np.zeros(bp.n_total)

        spikes = np.zeros(bp.n_total, dtype=bool)
        for i, n in enumerate(neurons):
            n.v += ext_events[i] * params["ext_weight"] + syn_dv[i]
            spikes[i] = n.step(0.0)

        # STDP updates (process a subset to keep wall time reasonable)
        for (pre, post), syn in stdp_syns.items():
            syn.process_step(int(prev_spikes[pre]), int(spikes[post]))
            weights[pre, post] = syn.effective_weight_probability() * bp.v_threshold

        prev_spikes = spikes
        spike_count += int(spikes.sum())

    wall = time.perf_counter() - t0
    rate = spike_count / (bp.sim_ms / 1000.0) / bp.n_total
    n_stdp = len(stdp_syns)
    return VariantResult(
        variant="v12_stdp_lif",
        total_spikes=spike_count,
        mean_rate_hz=rate,
        wall_time_s=wall,
        metric_note=f"stdp_synapses={n_stdp}",
    )


# ---------------------------------------------------------------------------
# V13: Dot-product — multi-channel SC summation
# ---------------------------------------------------------------------------
def run_v13_dot_product(bp: BrunelParams) -> VariantResult:
    from sc_neurocore import StochasticLIFNeuron, BitstreamSynapse, BitstreamDotProduct

    params = translate_v13_dot_product(bp)
    rng = np.random.default_rng(bp.seed)

    # Use subset for speed (dot product is O(n²·L))
    n_sub = min(50, bp.n_total)
    neurons = [StochasticLIFNeuron(**params["neuron_kwargs"]) for _ in range(n_sub)]

    bl = params["bitstream_length"]
    n_exc_sub = int(n_sub * bp.n_exc / bp.n_total)

    # One dot-product per postsynaptic neuron
    dot_products = []
    for _ in range(n_sub):
        syns = [
            BitstreamSynapse(w_min=0.0, w_max=1.0, length=bl, w=params["synapse_w_prob"])
            for _ in range(n_sub)
        ]
        dot_products.append(BitstreamDotProduct(synapses=syns))

    steps = int(bp.sim_ms / bp.dt)
    spike_count = 0
    prev_spikes = np.zeros(n_sub, dtype=bool)

    t0 = time.perf_counter()
    for _ in range(steps):
        ext_events = rng.poisson(_brunel_ext_lambda(bp), n_sub)

        # Encode previous spikes as bitstreams
        pre_matrix = np.zeros((n_sub, bl), dtype=np.uint8)
        for i in range(n_sub):
            if prev_spikes[i]:
                pre_matrix[i] = (rng.random(bl) < 0.8).astype(np.uint8)

        spikes = np.zeros(n_sub, dtype=bool)
        for j, (n, dp) in enumerate(zip(neurons, dot_products)):
            _, y_scalar = dp.apply(pre_matrix, y_min=0.0, y_max=bp.v_threshold)
            n.v += ext_events[j] * params["ext_weight"] + y_scalar
            spikes[j] = n.step(0.0)

        prev_spikes = spikes
        spike_count += int(spikes.sum())

    wall = time.perf_counter() - t0
    rate = spike_count / (bp.sim_ms / 1000.0) / n_sub
    return VariantResult(
        variant="v13_dot_product_lif",
        total_spikes=spike_count,
        mean_rate_hz=rate,
        wall_time_s=wall,
        metric_note=f"n_sub={n_sub}, bl={bl}",
    )


# ---------------------------------------------------------------------------
# V14: Sobol bitstream — low-discrepancy encoding
# ---------------------------------------------------------------------------
def run_v14_sobol(bp: BrunelParams) -> VariantResult:
    from sc_neurocore import StochasticLIFNeuron, BitstreamSynapse
    from sc_neurocore.utils.bitstreams import generate_sobol_bitstream

    params = translate_v14_sobol(bp)
    neurons = [StochasticLIFNeuron(**params["neuron_kwargs"]) for _ in range(bp.n_total)]

    rng = np.random.default_rng(bp.seed)
    conn_mask = rng.random((bp.n_total, bp.n_total)) < bp.conn_prob
    np.fill_diagonal(conn_mask, False)

    syn_exc = BitstreamSynapse(**params["synapse_kwargs"])
    inh_w = min(bp.g_inh * params["synapse_kwargs"]["w"], 0.999)
    syn_inh = BitstreamSynapse(**{**params["synapse_kwargs"], "w": inh_w})

    bl = params["bitstream_length"]
    eff_exc = syn_exc.effective_weight_probability() * bl * params["popcount_scale"]
    eff_inh = syn_inh.effective_weight_probability() * bl * params["popcount_scale"]
    eff_weights = np.where(conn_mask, eff_exc, 0.0)
    eff_weights[bp.n_exc :, :] = np.where(conn_mask[bp.n_exc :, :], -eff_inh, 0.0)

    steps = int(bp.sim_ms / bp.dt)
    spike_count = 0
    prev_spikes = np.zeros(bp.n_total, dtype=bool)

    t0 = time.perf_counter()
    for _ in range(steps):
        ext_events = rng.poisson(_brunel_ext_lambda(bp), bp.n_total)

        if prev_spikes.any():
            # Sobol-modulated noise instead of uniform
            noise = np.array(
                [generate_sobol_bitstream(0.5, 1, seed=s)[0] for s in range(prev_spikes.sum())],
                dtype=float,
            )
            syn_dv = (eff_weights[prev_spikes] * noise[:, None]).sum(axis=0)
        else:
            syn_dv = np.zeros(bp.n_total)

        spikes = np.zeros(bp.n_total, dtype=bool)
        for i, n in enumerate(neurons):
            n.v += ext_events[i] * bp.weight_exc + syn_dv[i]
            spikes[i] = bool(n.step(0.0))

        prev_spikes = spikes
        spike_count += int(spikes.sum())

    wall = time.perf_counter() - t0
    rate = spike_count / (bp.sim_ms / 1000.0) / bp.n_total
    return VariantResult(
        variant="v14_sobol_bitstream", total_spikes=spike_count, mean_rate_hz=rate, wall_time_s=wall
    )


# ---------------------------------------------------------------------------
# V15: JAX vectorized — JIT-compiled LIF layer
# ---------------------------------------------------------------------------
def run_v15_jax(bp: BrunelParams) -> VariantResult:
    try:
        from sc_neurocore import JaxSCDenseLayer
        from sc_neurocore.accel.jax_backend import jnp, HAS_JAX

        if not HAS_JAX:
            raise ImportError("JAX not available")
    except (ImportError, RuntimeError):
        return VariantResult(
            variant="v15_jax_vectorized",
            total_spikes=0,
            mean_rate_hz=0.0,
            wall_time_s=0.0,
            status="skipped",
            reason="JAX not installed",
        )

    params = translate_v15_jax(bp)
    rng = np.random.default_rng(bp.seed)

    layer = JaxSCDenseLayer(
        n_neurons=params["n_neurons"],
        n_inputs=params["n_inputs"],
        bitstream_length=params["bitstream_length"],
        neuron_params=params["neuron_params"],
        seed=bp.seed,
    )

    conn_mask = rng.random((bp.n_total, bp.n_total)) < params["conn_prob"]
    np.fill_diagonal(conn_mask, False)
    w_mat = np.where(conn_mask, params["weight_exc"], 0.0)
    w_mat[bp.n_exc :, :] *= -params["g_inh"]
    w_jax = jnp.array(w_mat)

    steps = int(bp.sim_ms / bp.dt)
    spike_count = 0

    t0 = time.perf_counter()
    prev_spikes = jnp.zeros(bp.n_total)
    for _ in range(steps):
        ext_events = jnp.array(rng.poisson(_brunel_ext_lambda(bp), bp.n_total).astype(float))
        I_syn = jnp.dot(prev_spikes, w_jax)
        I_total = ext_events * params["weight_exc"] + I_syn
        spikes_jax = layer.step(I_total)
        prev_spikes = spikes_jax.astype(float)
        spike_count += int(jnp.sum(spikes_jax))

    wall = time.perf_counter() - t0
    rate = spike_count / (bp.sim_ms / 1000.0) / bp.n_total
    return VariantResult(
        variant="v15_jax_vectorized", total_spikes=spike_count, mean_rate_hz=rate, wall_time_s=wall
    )


# ---------------------------------------------------------------------------
# V16: Recurrent reservoir — SCRecurrentLayer
# ---------------------------------------------------------------------------
def run_v16_recurrent(bp: BrunelParams) -> VariantResult:
    from sc_neurocore import SCRecurrentLayer

    params = translate_v16_recurrent(bp)
    n_sub = min(100, bp.n_total)
    layer = SCRecurrentLayer(
        n_inputs=n_sub,
        n_neurons=n_sub,
        feedback_strength=params["feedback_strength"],
        input_strength=params["input_strength"],
        spectral_radius=params["spectral_radius"],
        length=params["length"],
        seed=bp.seed,
    )

    rng = np.random.default_rng(bp.seed)
    steps = int(bp.sim_ms / bp.dt)
    ext_prob = _brunel_ext_lambda(bp)

    t0 = time.perf_counter()
    total_output = np.zeros(n_sub)
    for _ in range(steps):
        inp = rng.poisson(ext_prob, n_sub).astype(float)
        inp = np.clip(inp / 10.0, 0.0, 1.0)  # normalize to [0,1]
        out = layer.step(inp)
        total_output += out

    wall = time.perf_counter() - t0
    mean_out_prob = total_output.mean() / steps
    return VariantResult(
        variant="v16_recurrent_reservoir",
        total_spikes=0,
        mean_rate_hz=0.0,
        wall_time_s=wall,
        metric_note=f"n_sub={n_sub}",
        domain="probability",
        mean_output_prob=mean_out_prob,
    )


# ---------------------------------------------------------------------------
# V17: Memristive — hardware defects
# ---------------------------------------------------------------------------
def run_v17_memristive(bp: BrunelParams) -> VariantResult:
    from sc_neurocore import MemristiveDenseLayer

    params = translate_v17_memristive(bp)
    n_sub = min(100, bp.n_total)
    bl = params["length"]

    layer = MemristiveDenseLayer(
        n_inputs=n_sub,
        n_neurons=n_sub,
        length=bl,
        stuck_rate=params["stuck_rate"],
        variability=params["variability"],
    )

    rng = np.random.default_rng(bp.seed)
    conn = rng.random((n_sub, n_sub)) < params["conn_prob"]
    np.fill_diagonal(conn, False)
    layer.weights = np.where(conn, params["weight_prob"], 0.0)
    layer._refresh_packed_weights()

    steps = int(bp.sim_ms / bp.dt)
    input_probs = np.full(n_sub, params["ext_prob"])

    t0 = time.perf_counter()
    total_output = np.zeros(n_sub)
    for _ in range(steps):
        out = layer.forward(input_probs)
        total_output += out
    wall = time.perf_counter() - t0

    mean_out_prob = total_output.mean() / steps
    equiv_rate = mean_out_prob * 1000.0
    return VariantResult(
        variant="v17_memristive_defects",
        total_spikes=0,
        mean_rate_hz=equiv_rate,
        wall_time_s=wall,
        metric_note=f"prob={mean_out_prob:.4f}, stuck={params['stuck_rate']}",
    )


# ---------------------------------------------------------------------------
# V18: Numba JIT — compiled inner loop
# ---------------------------------------------------------------------------
def run_v18_numba(bp: BrunelParams) -> VariantResult:
    params = translate_v18_numba(bp)
    rng = np.random.default_rng(bp.seed)

    try:
        from numba import njit

        HAS_NUMBA = True
    except ImportError:
        HAS_NUMBA = False

    # Vectorized state arrays
    n = bp.n_total
    v = np.full(n, bp.v_rest)

    conn_mask = rng.random((n, n)) < bp.conn_prob
    np.fill_diagonal(conn_mask, False)
    weights = np.where(conn_mask, params["weight_exc"], 0.0)
    weights[bp.n_exc :, :] *= -bp.g_inh

    alpha = bp.dt / bp.tau_mem
    steps = int(bp.sim_ms / bp.dt)

    if HAS_NUMBA:

        @njit(cache=True)
        def _run_loop(
            v, weights, alpha, v_rest, v_threshold, v_reset, ext_weight, ext_rate_dt, n, steps, seed
        ):
            np.random.seed(seed)
            spike_count = 0
            prev_spikes = np.zeros(n, dtype=np.bool_)
            for _ in range(steps):
                ext_events = np.random.poisson(ext_rate_dt, n)
                syn_dv = np.zeros(n)
                for j in range(n):
                    if prev_spikes[j]:
                        for k in range(n):
                            syn_dv[k] += weights[j, k]
                new_spikes = np.zeros(n, dtype=np.bool_)
                for i in range(n):
                    v[i] += ext_events[i] * ext_weight + syn_dv[i]
                    v[i] += alpha * (v_rest - v[i])
                    if v[i] >= v_threshold:
                        new_spikes[i] = True
                        v[i] = v_reset
                        spike_count += 1
                prev_spikes = new_spikes
            return spike_count

    else:
        _run_loop = None

    ext_rate_dt = _brunel_ext_lambda(bp)

    t0 = time.perf_counter()
    if _run_loop is not None:
        spike_count = _run_loop(
            v,
            weights,
            alpha,
            bp.v_rest,
            bp.v_threshold,
            bp.v_reset,
            params["ext_weight"],
            ext_rate_dt,
            n,
            steps,
            bp.seed,
        )
    else:
        # Pure Python fallback
        spike_count = 0
        prev_spikes = np.zeros(n, dtype=bool)
        for _ in range(steps):
            ext_events = rng.poisson(ext_rate_dt, n)
            syn_dv = weights[prev_spikes].sum(axis=0) if prev_spikes.any() else np.zeros(n)
            v += ext_events * params["ext_weight"] + syn_dv
            v += alpha * (bp.v_rest - v)
            fired = v >= bp.v_threshold
            spike_count += int(fired.sum())
            v[fired] = bp.v_reset
            prev_spikes = fired

    wall = time.perf_counter() - t0
    rate = spike_count / (bp.sim_ms / 1000.0) / n
    note = "numba JIT" if (_run_loop is not None and HAS_NUMBA) else "pure Python fallback"
    return VariantResult(
        variant="v18_numba_jit",
        total_spikes=spike_count,
        mean_rate_hz=rate,
        wall_time_s=wall,
        metric_note=note,
    )


# ---------------------------------------------------------------------------
# V19: PyTorch CUDA — GPU-accelerated LIF
# ---------------------------------------------------------------------------
def run_v19_pytorch_cuda(bp: BrunelParams) -> VariantResult:
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA")
    except (ImportError, RuntimeError):
        return VariantResult(
            variant="v19_pytorch_cuda",
            total_spikes=0,
            mean_rate_hz=0.0,
            wall_time_s=0.0,
            metric_note="SKIPPED (no CUDA)",
        )

    params = translate_v19_pytorch_cuda(bp)
    device = torch.device("cuda")
    n = params["n_total"]

    rng = np.random.default_rng(bp.seed)
    conn_mask = rng.random((n, n)) < params["conn_prob"]
    np.fill_diagonal(conn_mask, False)
    w_np = np.where(conn_mask, params["weight_exc"], 0.0)
    w_np[params["n_exc"] :, :] *= -params["g_inh"]

    w = torch.tensor(w_np, dtype=torch.float32, device=device)
    v = torch.full((n,), params["v_rest"], dtype=torch.float32, device=device)
    alpha = params["dt"] / params["tau_mem"]
    steps = int(bp.sim_ms / params["dt"])

    t0 = time.perf_counter()
    spike_count = 0
    prev_spikes = torch.zeros(n, dtype=torch.float32, device=device)

    for _ in range(steps):
        ext_events = torch.tensor(
            rng.poisson(_brunel_ext_lambda(bp), n),
            dtype=torch.float32,
            device=device,
        )
        I_syn = torch.matmul(prev_spikes, w)
        v += ext_events * params["weight_exc"] + I_syn
        v += alpha * (params["v_rest"] - v)
        fired = v >= params["v_threshold"]
        spike_count += int(fired.sum().item())
        v = torch.where(fired, torch.tensor(params["v_reset"], device=device), v)
        prev_spikes = fired.float()

    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    rate = spike_count / (bp.sim_ms / 1000.0) / n
    gpu_name = torch.cuda.get_device_name(0)
    return VariantResult(
        variant="v19_pytorch_cuda",
        total_spikes=spike_count,
        mean_rate_hz=rate,
        wall_time_s=wall,
        metric_note=f"GPU: {gpu_name}",
    )


# ---------------------------------------------------------------------------
# V20: Vectorized NumPy — batch neuron update (no per-neuron loop)
# ---------------------------------------------------------------------------
def run_v20_vectorized_numpy(bp: BrunelParams) -> VariantResult:
    params = translate_v20_vectorized_numpy(bp)
    rng = np.random.default_rng(bp.seed)
    n = params["n_total"]

    conn_mask = rng.random((n, n)) < params["conn_prob"]
    np.fill_diagonal(conn_mask, False)
    weights = np.where(conn_mask, params["weight_exc"], 0.0)
    weights[params["n_exc"] :, :] *= -params["g_inh"]

    v = np.full(n, params["v_rest"])
    alpha = params["dt"] / params["tau_mem"]
    steps = int(bp.sim_ms / params["dt"])
    spike_count = 0
    prev_spikes = np.zeros(n, dtype=bool)

    t0 = time.perf_counter()
    for _ in range(steps):
        ext_events = rng.poisson(_brunel_ext_lambda(bp), n)

        # Vectorized synaptic input
        if prev_spikes.any():
            I_syn = weights[prev_spikes].sum(axis=0)
        else:
            I_syn = np.zeros(n)

        # Batch update all neurons
        v += ext_events * params["weight_exc"] + I_syn
        v += alpha * (params["v_rest"] - v)

        fired = v >= params["v_threshold"]
        spike_count += int(fired.sum())
        v[fired] = params["v_reset"]
        prev_spikes = fired

    wall = time.perf_counter() - t0
    rate = spike_count / (bp.sim_ms / 1000.0) / n
    return VariantResult(
        variant="v20_vectorized_numpy",
        total_spikes=spike_count,
        mean_rate_hz=rate,
        wall_time_s=wall,
    )


# ---------------------------------------------------------------------------
# V21: Sparse Numba — CSR connectivity for N>=1K
# ---------------------------------------------------------------------------
def run_v21_sparse_numba(bp: BrunelParams) -> VariantResult:
    """Numba JIT with CSR sparse connectivity.

    Stores the weight matrix as CSR (indptr, indices, data) so the inner
    synapse loop touches only connected entries — O(N*C_E) instead of O(N^2).
    """
    try:
        from numba import njit
    except ImportError:
        return VariantResult(
            variant="v21_sparse_numba",
            total_spikes=0,
            mean_rate_hz=0.0,
            wall_time_s=0.0,
            status="skipped",
            reason="Numba not installed",
        )

    try:
        import scipy.sparse as sp_mod
    except ImportError:
        return VariantResult(
            variant="v21_sparse_numba",
            total_spikes=0,
            mean_rate_hz=0.0,
            wall_time_s=0.0,
            status="skipped",
            reason="scipy not installed",
        )

    params = translate_v18_numba(bp)
    rng = np.random.default_rng(bp.seed)

    n = bp.n_total
    v = np.full(n, bp.v_rest)

    conn_mask = rng.random((n, n)) < bp.conn_prob
    np.fill_diagonal(conn_mask, False)
    weights_dense = np.where(conn_mask, params["weight_exc"], 0.0)
    weights_dense[bp.n_exc :, :] *= -bp.g_inh

    csr = sp_mod.csr_matrix(weights_dense)
    indptr = csr.indptr.astype(np.int64)
    indices = csr.indices.astype(np.int64)
    data = csr.data.astype(np.float64)

    alpha = bp.dt / bp.tau_mem
    steps = int(bp.sim_ms / bp.dt)
    ext_rate_dt = _brunel_ext_lambda(bp)

    @njit(cache=True)
    def _run_sparse(
        v,
        indptr,
        indices,
        data,
        alpha,
        v_rest,
        v_threshold,
        v_reset,
        ext_weight,
        ext_rate_dt,
        n,
        steps,
        seed,
    ):
        np.random.seed(seed)
        spike_count = 0
        prev_spikes = np.zeros(n, dtype=np.bool_)
        for _ in range(steps):
            ext_events = np.random.poisson(ext_rate_dt, n)
            syn_dv = np.zeros(n)
            for j in range(n):
                if prev_spikes[j]:
                    for idx in range(indptr[j], indptr[j + 1]):
                        syn_dv[indices[idx]] += data[idx]
            new_spikes = np.zeros(n, dtype=np.bool_)
            for i in range(n):
                v[i] += ext_events[i] * ext_weight + syn_dv[i]
                v[i] += alpha * (v_rest - v[i])
                if v[i] >= v_threshold:
                    new_spikes[i] = True
                    v[i] = v_reset
                    spike_count += 1
            prev_spikes = new_spikes
        return spike_count

    t0 = time.perf_counter()
    spike_count = _run_sparse(
        v,
        indptr,
        indices,
        data,
        alpha,
        bp.v_rest,
        bp.v_threshold,
        bp.v_reset,
        params["ext_weight"],
        ext_rate_dt,
        n,
        steps,
        bp.seed,
    )
    wall = time.perf_counter() - t0
    rate = spike_count / (bp.sim_ms / 1000.0) / n
    nnz_pct = 100.0 * csr.nnz / (n * n)
    return VariantResult(
        variant="v21_sparse_numba",
        total_spikes=spike_count,
        mean_rate_hz=rate,
        wall_time_s=wall,
        metric_note=f"CSR nnz={csr.nnz} ({nnz_pct:.1f}%)",
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
VARIANTS = {
    "brian2": run_brian2_reference,
    "v1": run_v1_stochastic_lif,
    "v2": run_v2_rate_matched,
    "v3": run_v3_fixed_point,
    "v4": run_v4_hybrid,
    "v5": run_v5_izhikevich,
    "v6": run_v6_homeostatic,
    "v7": run_v7_noisy,
    "v8": run_v8_refractory,
    "v9": run_v9_post_kick,
    "v10": run_v10_exact_leak,
    "v11": run_v11_q16,
    "v12": run_v12_stdp,
    "v13": run_v13_dot_product,
    "v14": run_v14_sobol,
    "v15": run_v15_jax,
    "v16": run_v16_recurrent,
    "v17": run_v17_memristive,
    "v18": run_v18_numba,
    "v19": run_v19_pytorch_cuda,
    "v20": run_v20_vectorized_numpy,
    "v21": run_v21_sparse_numba,
}


def _serialize(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"{type(obj).__name__} not JSON serializable")


def run_all(
    bp: BrunelParams,
    variants: list[str] | None = None,
    brian2_result: VariantResult | None = None,
) -> list[VariantResult]:
    targets = variants or ["v1", "v2", "v3", "v4"]
    results: list[VariantResult] = []

    for name in targets:
        fn = VARIANTS[name]
        print(f"  {name}...", end=" ", flush=True)
        r = fn(bp)
        if r is None:
            print("SKIPPED (not installed)")
            continue
        if brian2_result and name != "brian2":
            r.brian2_spikes = brian2_result.total_spikes
            r.brian2_rate_hz = brian2_result.mean_rate_hz
            if brian2_result.mean_rate_hz > 0 and r.mean_rate_hz > 0:
                r.rate_ratio = r.mean_rate_hz / brian2_result.mean_rate_hz
        if r.status == "skipped":
            print(f"SKIPPED ({r.reason})")
        elif r.domain == "probability":
            print(f"{r.wall_time_s:.2f}s, prob={r.mean_output_prob:.4f}")
        else:
            print(f"{r.wall_time_s:.2f}s, {r.total_spikes:,} spikes, {r.mean_rate_hz:.1f} Hz")
        results.append(r)

    return results


def format_markdown(results: list[VariantResult]) -> str:
    lines = [
        "| Variant | Spikes | Rate (Hz) | Brian2 Ratio | Wall (s) | Note |",
        "|---------|-------:|----------:|-------------:|---------:|------|",
    ]
    for r in results:
        if r.status == "skipped":
            lines.append(f"| {r.variant} | — | — | — | — | skipped: {r.reason} |")
            continue
        ratio = f"{r.rate_ratio:.2f}" if r.rate_ratio > 0 else "—"
        note = r.metric_note or ""
        if r.domain == "probability":
            rate_str = f"{r.mean_output_prob:.4f} (prob)"
        else:
            rate_str = f"{r.mean_rate_hz:.1f}"
        lines.append(
            f"| {r.variant} | {r.total_spikes:,} | {rate_str} "
            f"| {ratio} | {r.wall_time_s:.2f} | {note} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="SNN comparison: 20 SC-NeuroCore variants vs Brian2")
    ap.add_argument("--all", action="store_true", help="run all variants + Brian2")
    ap.add_argument("--variant", choices=list(VARIANTS.keys()), action="append")
    ap.add_argument("--json", type=str, help="write results to JSON file")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--sim-ms", type=float, default=1000.0)
    args = ap.parse_args()

    bp = BrunelParams(sim_ms=args.sim_ms)
    ext_lambda = _brunel_ext_lambda(bp)

    print(f"\nBrunel Network: {bp.n_total} neurons, sim={bp.sim_ms}ms")
    print(f"  weight_exc={bp.weight_exc}, ext_lambda={ext_lambda:.3f}/step (Brunel AI)")
    print("=" * 60)

    # Run Brian2 reference first if available
    brian2_result = None
    if args.all or (args.variant and "brian2" in args.variant):
        print("  brian2...", end=" ", flush=True)
        brian2_result = run_brian2_reference(bp)
        if brian2_result:
            print(
                f"{brian2_result.wall_time_s:.2f}s, "
                f"{brian2_result.total_spikes:,} spikes, "
                f"{brian2_result.mean_rate_hz:.1f} Hz"
            )
        else:
            print("SKIPPED (not installed)")

    # Determine variants to run
    if args.all:
        variants = [f"v{i}" for i in range(1, 22)]
    elif args.variant:
        variants = [v for v in args.variant if v != "brian2"]
    else:
        variants = ["v1"]

    results = run_all(bp, variants, brian2_result)
    if brian2_result:
        results.insert(0, brian2_result)

    if args.markdown:
        print("\n" + format_markdown(results))

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps([asdict(r) for r in results], indent=2, default=_serialize)
        )
        print(f"\nResults written to {args.json}")


if __name__ == "__main__":
    main()
