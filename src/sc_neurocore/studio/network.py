# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Network simulation for Studio (Rust engine backend)

"""Balanced E-I network simulation helpers for Studio dashboards.

The network is a Brunel (2000, J. Comput. Neurosci. 8:183) style LIF network
with delta synapses: a presynaptic spike moves the postsynaptic membrane by
its weight in millivolts, one step later. Every neuron also receives
:data:`EXTERNAL_SYNAPSES` independent excitatory Poisson inputs at ``ext_rate``
Hz each, every one moving the membrane by :data:`EXTERNAL_WEIGHT_MV`; that
drive alone reaches threshold at about 9.4 Hz per input. The weights follow
the post-pre convention: ``w_ei`` is the inhibitory-to-excitatory weight and
``w_ie`` the excitatory-to-inhibitory one.

The Rust engine and the NumPy fallback implement the same model. An earlier
version multiplied each jump by the step size and gave each neuron a single
external input, so no setting of the Studio's controls produced a spike.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: Independent external excitatory Poisson inputs per neuron (same as the engine).
EXTERNAL_SYNAPSES = 800.0

#: Membrane jump caused by one external input spike, in millivolts.
EXTERNAL_WEIGHT_MV = 0.1

#: Default rate of each external input, in hertz: 1.3x the drive's threshold rate.
DEFAULT_EXTERNAL_RATE_HZ = 12.0

try:
    from sc_neurocore_engine.studio import get_ei_network_simulator
except ImportError:

    def get_ei_network_simulator() -> object:
        """Return the optional Rust E-I network simulator or raise when unavailable."""
        raise ImportError("Studio Rust E-I network simulator unavailable")


def simulate_ei_network(
    n_exc: int = 80,
    n_inh: int = 20,
    w_ee: float = 0.1,
    w_ei: float = 0.4,
    w_ie: float = 0.1,
    w_ii: float = 0.4,
    p_conn: float = 0.2,
    ext_rate: float = DEFAULT_EXTERNAL_RATE_HZ,
    duration: float = 200.0,
    dt: float = 0.1,
) -> dict[str, Any]:
    """Simulate a balanced E-I network. Uses Rust engine when available."""
    try:
        return _simulate_rust(
            n_exc,
            n_inh,
            w_ee,
            w_ei,
            w_ie,
            w_ii,
            p_conn,
            ext_rate,
            duration,
            dt,
        )
    except ImportError:
        return _simulate_numpy(
            n_exc,
            n_inh,
            w_ee,
            w_ei,
            w_ie,
            w_ii,
            p_conn,
            ext_rate,
            duration,
            dt,
        )


def _simulate_rust(
    n_exc: int,
    n_inh: int,
    w_ee: float,
    w_ei: float,
    w_ie: float,
    w_ii: float,
    p_conn: float,
    ext_rate: float,
    duration: float,
    dt: float,
) -> dict[str, Any]:
    """Entire simulation in Rust — connectivity, Poisson input, stepping, recording."""
    simulate = get_ei_network_simulator()
    result = simulate(
        n_exc=n_exc,
        n_inh=n_inh,
        w_ee=w_ee,
        w_ei=w_ei,
        w_ie=w_ie,
        w_ii=w_ii,
        p_conn=p_conn,
        ext_rate=ext_rate,
        duration=duration,
        dt=dt,
        seed=42,
    )
    # Convert numpy arrays to lists for JSON serialisation
    return {
        "spike_times": result["spike_times"].tolist(),
        "spike_neurons": result["spike_neurons"].tolist(),
        "n_exc": int(result["n_exc"]),
        "n_inh": int(result["n_inh"]),
        "n_total": int(result["n_total"]),
        "n_spikes": int(result["n_spikes"]),
        "rate_time": result["rate_time"].tolist(),
        "exc_rates": result["exc_rates"].tolist(),
        "inh_rates": result["inh_rates"].tolist(),
        "duration": duration,
        "dt": dt,
        "mean_exc_rate": float(result["mean_exc_rate"]),
        "mean_inh_rate": float(result["mean_inh_rate"]),
    }


def _simulate_numpy(
    n_exc: int,
    n_inh: int,
    w_ee: float,
    w_ei: float,
    w_ie: float,
    w_ii: float,
    p_conn: float,
    ext_rate: float,
    duration: float,
    dt: float,
) -> dict[str, Any]:
    """Pure NumPy fallback (no Rust engine)."""
    n_total = n_exc + n_inh
    n_steps = min(int(duration / dt), 50_000)
    rng = np.random.default_rng(42)

    tau_m = 20.0
    v_rest = -65.0
    v_threshold = -50.0
    v_reset = -65.0
    tau_ref = 2.0

    v = np.full(n_total, v_rest)
    refractory = np.zeros(n_total)

    W = np.zeros((n_total, n_total))
    mask = rng.random((n_total, n_total)) < p_conn
    np.fill_diagonal(mask, False)
    for i in range(n_total):
        for j in range(n_total):
            if not mask[i, j]:
                continue
            i_exc = i < n_exc
            j_exc = j < n_exc
            if i_exc and j_exc:
                W[j, i] = w_ee
            elif i_exc and not j_exc:
                W[j, i] = w_ie
            elif not i_exc and j_exc:
                W[j, i] = -w_ei
            else:
                W[j, i] = -w_ii

    spike_times: list[float] = []
    spike_neurons: list[int] = []
    bin_size = max(1, n_steps // 100)
    exc_rates = np.zeros(n_steps)
    inh_rates = np.zeros(n_steps)
    exc_bin = 0
    inh_bin = 0
    exc_spikes = 0
    inh_spikes = 0
    prev_spiked = np.zeros(n_total, dtype=bool)

    for t in range(n_steps):
        refractory = np.maximum(refractory - dt, 0)
        ext_input = (
            rng.poisson(EXTERNAL_SYNAPSES * max(ext_rate, 0.0) * dt / 1000.0, n_total).astype(float)
            * EXTERNAL_WEIGHT_MV
        )
        syn_input = W[:, prev_spiked].sum(axis=1) if prev_spiked.any() else np.zeros(n_total)

        active = refractory <= 0
        dv = -(v - v_rest) / tau_m * dt + ext_input + syn_input
        v[active] += dv[active]

        spiking = (v >= v_threshold) & active
        spike_idx = np.where(spiking)[0]

        for idx in spike_idx:
            spike_times.append(t * dt)
            spike_neurons.append(int(idx))
            if idx < n_exc:
                exc_bin += 1
                exc_spikes += 1
            else:
                inh_bin += 1
                inh_spikes += 1

        v[spiking] = v_reset
        refractory[spiking] = tau_ref
        prev_spiked = spiking

        if (t + 1) % bin_size == 0:
            bi = t // bin_size
            bin_t = bin_size * dt / 1000.0
            if bi < len(exc_rates):
                exc_rates[bi] = exc_bin / max(n_exc, 1) / max(bin_t, 0.001)
                inh_rates[bi] = inh_bin / max(n_inh, 1) / max(bin_t, 0.001)
            exc_bin = 0
            inh_bin = 0

    n_bins = n_steps // bin_size
    rate_time = np.arange(n_bins) * bin_size * dt
    # Mean rate per neuron over the simulated time, silent time included.
    simulated_s = max(n_steps * dt / 1000.0, np.finfo(float).tiny)
    exc_r = exc_rates[:n_bins]
    inh_r = inh_rates[:n_bins]

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
        "mean_exc_rate": round(exc_spikes / max(n_exc, 1) / simulated_s, 1),
        "mean_inh_rate": round(inh_spikes / max(n_inh, 1) / simulated_s, 1),
    }
