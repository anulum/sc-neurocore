# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Network simulation for Studio (E-I balance demo)

from __future__ import annotations

import numpy as np


def simulate_ei_network(
    n_exc: int = 80,
    n_inh: int = 20,
    w_ee: float = 0.1,
    w_ei: float = 0.4,
    w_ie: float = 0.1,
    w_ii: float = 0.4,
    p_conn: float = 0.2,
    ext_rate: float = 5.0,
    duration: float = 200.0,
    dt: float = 0.1,
) -> dict:
    """Simulate a basic E-I balanced network of LIF neurons.

    Returns spike raster data and population firing rates.
    """
    n_total = n_exc + n_inh
    n_steps = min(int(duration / dt), 50_000)

    # LIF parameters
    tau_m = 20.0
    v_rest = -65.0
    v_threshold = -50.0
    v_reset = -65.0
    tau_ref = 2.0

    # State
    rng = np.random.default_rng(42)
    v = np.full(n_total, v_rest)
    refractory = np.zeros(n_total)

    # Connectivity (sparse random)
    W = np.zeros((n_total, n_total))
    for i in range(n_total):
        for j in range(n_total):
            if i == j:
                continue
            if rng.random() < p_conn:
                i_is_exc = i < n_exc
                j_is_exc = j < n_exc
                if i_is_exc and j_is_exc:
                    W[j, i] = w_ee
                elif i_is_exc and not j_is_exc:
                    W[j, i] = w_ie
                elif not i_is_exc and j_is_exc:
                    W[j, i] = -w_ei
                else:
                    W[j, i] = -w_ii

    # Recording
    spike_times: list[float] = []
    spike_neurons: list[int] = []
    exc_rates = np.zeros(n_steps)
    inh_rates = np.zeros(n_steps)

    bin_size = max(1, n_steps // 100)
    exc_bin_spikes = 0
    inh_bin_spikes = 0

    for t in range(n_steps):
        refractory = np.maximum(refractory - dt, 0)

        # External Poisson input
        ext_input = rng.poisson(ext_rate * dt / 1000.0, n_total).astype(float) * 5.0

        # Synaptic input from spikes in previous step
        syn_input = np.zeros(n_total)
        if t > 0:
            prev_spikes = np.where(
                (refractory > tau_ref - dt - 0.001) & (refractory <= tau_ref + 0.001)
            )[0]
            if len(prev_spikes) > 0:
                syn_input = W[:, prev_spikes].sum(axis=1)

        # Update voltage (Euler)
        active = refractory <= 0
        dv = (-(v - v_rest) / tau_m + ext_input + syn_input) * dt
        v[active] += dv[active]

        # Spike detection
        spiking = (v >= v_threshold) & active
        spike_idx = np.where(spiking)[0]

        for idx in spike_idx:
            spike_times.append(t * dt)
            spike_neurons.append(int(idx))
            if idx < n_exc:
                exc_bin_spikes += 1
            else:
                inh_bin_spikes += 1

        v[spiking] = v_reset
        refractory[spiking] = tau_ref

        if (t + 1) % bin_size == 0:
            bin_idx = t // bin_size
            if bin_idx < n_steps:
                bin_t = bin_size * dt / 1000.0
                exc_rates[bin_idx] = exc_bin_spikes / max(n_exc, 1) / max(bin_t, 0.001)
                inh_rates[bin_idx] = inh_bin_spikes / max(n_inh, 1) / max(bin_t, 0.001)
            exc_bin_spikes = 0
            inh_bin_spikes = 0

    # Downsample rates to ~100 points
    rate_time = np.arange(n_steps // bin_size) * bin_size * dt
    exc_r = exc_rates[: n_steps // bin_size]
    inh_r = inh_rates[: n_steps // bin_size]

    return {
        "spike_times": spike_times,
        "spike_neurons": spike_neurons,
        "n_exc": n_exc,
        "n_inh": n_inh,
        "n_total": n_total,
        "n_spikes": len(spike_times),
        "rate_time": rate_time.tolist(),
        "exc_rates": exc_r.tolist(),
        "inh_rates": inh_r.tolist(),
        "duration": duration,
        "dt": dt,
        "mean_exc_rate": round(float(np.mean(exc_r[exc_r > 0])) if np.any(exc_r > 0) else 0, 1),
        "mean_inh_rate": round(float(np.mean(inh_r[inh_r > 0])) if np.any(inh_r > 0) else 0, 1),
    }
