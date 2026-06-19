# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spike train visualization (pure matplotlib)

"""Spike train visualization: raster, voltage, ISI, cross-correlogram, PSD, and more."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:  # pragma: no cover
    HAS_MPL = False


def _require_mpl() -> None:
    if not HAS_MPL:
        raise ImportError("matplotlib is required for sc_neurocore.viz.plots")


def _get_ax(ax: Any) -> Any:
    """Return existing axes or create a new figure+axes pair."""
    if ax is not None:
        return ax
    _, ax = plt.subplots()
    return ax


def raster_plot(
    spike_monitor: Any,
    ax: Any = None,
    color: str = "k",
    marker: str = ".",
    s: float = 1,
) -> Any:
    """Spike raster from a SpikeMonitor."""
    _require_mpl()
    ax = _get_ax(ax)
    times, ids = spike_monitor.raster_data()
    ax.scatter(times, ids, c=color, marker=marker, s=s, linewidths=0)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Neuron ID")
    ax.set_title("Spike Raster")
    return ax


def voltage_trace(state_monitor: Any, neuron_ids: Any = None, ax: Any = None) -> Any:
    """Membrane voltage traces from a StateMonitor."""
    _require_mpl()
    ax = _get_ax(ax)
    traces = state_monitor.traces
    v = traces.get("v", np.empty((0, 0)))
    if v.size == 0:
        return ax
    t = state_monitor.t
    if neuron_ids is None:
        neuron_ids = range(v.shape[1])
    for nid in neuron_ids:
        ax.plot(t, v[:, nid], label=f"N{nid}")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Membrane voltage")
    ax.set_title("Voltage Traces")
    ax.legend(fontsize="small")
    return ax


def firing_rate_plot(spike_monitor: Any, bin_ms: int = 10, ax: Any = None) -> Any:
    """Population firing rate histogram (spikes per bin)."""
    _require_mpl()
    ax = _get_ax(ax)
    times, _ = spike_monitor.raster_data()
    if times.size == 0:
        ax.set_title("Firing Rate (no spikes)")
        return ax
    t_max = int(times.max()) + 1
    bins = np.arange(0, t_max + bin_ms, bin_ms)
    ax.hist(times, bins=bins, color="steelblue", edgecolor="none")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Spike count")
    ax.set_title("Population Firing Rate")
    return ax


def isi_histogram(spike_monitor: Any, neuron_id: int, bins: int = 50, ax: Any = None) -> Any:
    """Inter-spike interval distribution for a single neuron."""
    _require_mpl()
    ax = _get_ax(ax)
    intervals = spike_monitor.isi(neuron_id)
    if intervals.size == 0:
        ax.set_title(f"ISI N{neuron_id} (no spikes)")
        return ax
    ax.hist(intervals, bins=bins, color="coral", edgecolor="none")
    ax.set_xlabel("ISI (timesteps)")
    ax.set_ylabel("Count")
    ax.set_title(f"ISI Distribution — Neuron {neuron_id}")
    return ax


def cross_correlogram(
    spike_monitor: Any, i: int, j: int, max_lag_ms: int = 50, ax: Any = None
) -> Any:
    """Spike cross-correlation between neurons *i* and *j*."""
    _require_mpl()
    ax = _get_ax(ax)
    corr, lags = spike_monitor.cross_correlation(i, j, max_lag=int(max_lag_ms))
    ax.bar(lags, corr, width=1.0, color="teal", edgecolor="none")
    ax.set_xlabel("Lag (timesteps)")
    ax.set_ylabel("Correlation")
    ax.set_title(f"Cross-Correlogram N{i}–N{j}")
    return ax


def population_activity(spike_monitor: Any, bin_ms: int = 5, ax: Any = None) -> Any:
    """Heatmap of binned spike counts per neuron."""
    _require_mpl()
    ax = _get_ax(ax)
    times, ids = spike_monitor.raster_data()
    if times.size == 0:
        ax.set_title("Population Activity (no spikes)")
        return ax
    n_neurons = spike_monitor.population.n
    t_max = int(times.max()) + 1
    n_bins = max(1, t_max // bin_ms)
    mat = np.zeros((n_neurons, n_bins), dtype=np.float64)
    for t, nid in zip(times, ids):
        b = min(int(t) // bin_ms, n_bins - 1)
        mat[nid, b] += 1
    ax.imshow(mat, aspect="auto", origin="lower", interpolation="nearest")
    ax.set_xlabel(f"Time bin ({bin_ms} steps)")
    ax.set_ylabel("Neuron ID")
    ax.set_title("Population Activity")
    return ax


def phase_portrait(
    state_monitor: Any,
    var_x: str = "v",
    var_y: str = "w",
    neuron_id: int = 0,
    ax: Any = None,
) -> Any:
    """2-D phase-plane trajectory for a single neuron."""
    _require_mpl()
    ax = _get_ax(ax)
    traces = state_monitor.traces
    x = traces.get(var_x, np.empty((0, 0)))
    y = traces.get(var_y, np.empty((0, 0)))
    if x.size == 0 or y.size == 0:
        ax.set_title("Phase Portrait (no data)")
        return ax
    ax.plot(x[:, neuron_id], y[:, neuron_id], linewidth=0.8)
    ax.set_xlabel(var_x)
    ax.set_ylabel(var_y)
    ax.set_title(f"Phase Portrait — Neuron {neuron_id}")
    return ax


def weight_matrix(projection: Any, ax: Any = None, cmap: str = "RdBu_r") -> Any:
    """Connectivity weight heatmap from a Projection's CSR data."""
    _require_mpl()
    ax = _get_ax(ax)
    n_src = projection.source.n
    n_tgt = projection.target.n
    dense = np.zeros((n_src, n_tgt), dtype=np.float64)
    for i in range(n_src):
        for k in range(projection.indptr[i], projection.indptr[i + 1]):
            dense[i, projection.indices[k]] = projection.data[k]
    im = ax.imshow(dense, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xlabel("Target neuron")
    ax.set_ylabel("Source neuron")
    ax.set_title("Weight Matrix")
    return ax


def network_graph(network: Any, ax: Any = None) -> Any:
    """Node/edge diagram of populations and projections."""
    _require_mpl()
    ax = _get_ax(ax)
    pops = network.populations
    n = len(pops)
    if n == 0:
        ax.set_title("Network (empty)")
        return ax
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs, ys = np.cos(angles), np.sin(angles)
    for idx, pop in enumerate(pops):
        ax.scatter(xs[idx], ys[idx], s=200, zorder=3)
        ax.annotate(pop.label, (xs[idx], ys[idx]), ha="center", va="bottom", fontsize=8)
    pop_index = {id(p): idx for idx, p in enumerate(pops)}
    for proj in network.projections:
        si = pop_index.get(id(proj.source))
        ti = pop_index.get(id(proj.target))
        if si is not None and ti is not None:
            ax.annotate(
                "",
                xy=(xs[ti], ys[ti]),
                xytext=(xs[si], ys[si]),
                arrowprops=dict(arrowstyle="->", color="gray"),
            )
    ax.set_aspect("equal")
    ax.set_title("Network Graph")
    ax.axis("off")
    return ax


def psd_plot(spike_monitor: Any, neuron_id: int, ax: Any = None) -> Any:
    """Power spectral density of a single neuron's spike train."""
    _require_mpl()
    ax = _get_ax(ax)
    trains = spike_monitor.spike_trains
    ts = trains.get(neuron_id, np.array([], dtype=np.int64))
    if ts.size < 2:
        ax.set_title(f"PSD N{neuron_id} (too few spikes)")
        return ax
    t_max = int(ts.max()) + 1
    binary = np.zeros(t_max, dtype=np.float64)
    binary[ts] = 1.0
    freqs = np.fft.rfftfreq(t_max)
    power = np.abs(np.fft.rfft(binary - binary.mean())) ** 2
    ax.semilogy(freqs[1:], power[1:], linewidth=0.7)
    ax.set_xlabel("Frequency (cycles/step)")
    ax.set_ylabel("Power")
    ax.set_title(f"PSD — Neuron {neuron_id}")
    return ax


def instantaneous_rate_plot(
    spike_monitor: Any, neuron_id: int, sigma_ms: int = 20, ax: Any = None
) -> Any:
    """Gaussian-kernel-smoothed instantaneous firing rate."""
    _require_mpl()
    ax = _get_ax(ax)
    trains = spike_monitor.spike_trains
    ts = trains.get(neuron_id, np.array([], dtype=np.int64))
    if ts.size == 0:
        ax.set_title(f"Rate N{neuron_id} (no spikes)")
        return ax
    t_max = int(ts.max()) + 1
    binary = np.zeros(t_max, dtype=np.float64)
    binary[ts] = 1.0
    kernel_width = int(4 * sigma_ms)
    kernel_x = np.arange(-kernel_width, kernel_width + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (kernel_x / sigma_ms) ** 2)
    kernel /= kernel.sum()
    rate = np.convolve(binary, kernel, mode="same")
    ax.plot(np.arange(t_max), rate, linewidth=0.8)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Rate (spikes/step)")
    ax.set_title(f"Instantaneous Rate — Neuron {neuron_id}")
    return ax


def spike_train_comparison(
    trains: list[np.ndarray[Any, Any]], labels: list[str] | None = None, ax: Any = None
) -> Any:
    """Overlay multiple spike trains as event plots.

    Parameters
    ----------
    trains : list of np.ndarray[Any, Any]
        Each element is a 1-D array of spike timesteps.
    labels : list of str, optional
        Labels for each train.
    """
    _require_mpl()
    ax = _get_ax(ax)
    for idx, tr in enumerate(trains):
        lbl = labels[idx] if labels else f"Train {idx}"
        y_vals = np.full_like(tr, idx, dtype=np.float64)
        ax.scatter(tr, y_vals, s=2, label=lbl, linewidths=0)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Train index")
    ax.set_title("Spike Train Comparison")
    ax.legend(fontsize="small")
    return ax
